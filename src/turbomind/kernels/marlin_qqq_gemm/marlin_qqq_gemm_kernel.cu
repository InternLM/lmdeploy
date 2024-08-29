/*
 * Adapted from
 * https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu
 * https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda.cpp
 * Modified by HandH1998
 * Copyright (C) 2024 HandH1998
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.h"
#include "marlin_qqq_gemm_kernel.h"
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace turbomind {
namespace marlin_qqq {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template<const int threads,          // number of threads in a threadblock
         const int thread_m_blocks,  // number of 16x16 blocks in the m
                                     // dimension (batchsize) of the
                                     // threadblock
         const int thread_n_blocks,  // same for n dimension (output)
         const int thread_k_blocks,  // same for k dimension (reduction)
         const int stages,           // number of stages for the async global->shared
                                     // fetch pipeline
         const int group_blocks      // number of consecutive 16x16 blocks
                                     // with a separate quantization scale
         >
__global__ void Marlin(const int4* __restrict__ A,    // int8 input matrix of shape mxk
                       const int4* __restrict__ B,    // 4bit quantized weight matrix of shape kxn
                       int4* __restrict__ C,          // int32 global_reduce buffer of shape
                                                      // (max_par*16*4)xn, as int8 tensor core's output is
                                                      // int32 dtype
                       int4* __restrict__ D,          // fp16 output buffer of shape mxn
                       const float* __restrict__ s1,  // fp32 activation per-token quantization
                                                      // scales of shape mx1
                       const int4* __restrict__ s2,   // fp32 weight per-channel quantization scales
                                                      // of shape 1xn
                       const int4* __restrict__ s3,   // fp16 weight per-group quantization scales
                                                      // of shape (k/groupsize)xn, when
                                                      // group_blocks=-1, it should be nullptr
                       int  prob_m,                   // batch dimension m
                       int  prob_n,                   // output dimension n
                       int  prob_k,                   // reduction dimension k
                       int* locks                     // extra global storage for barrier synchronization
)
{
    // Each threadblock processes one "stripe" of the B matrix with (roughly) the
    // same size, which might involve multiple column "slices" (of width 16 *
    // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
    // example:
    //   0 1 3
    //   0 2 3
    //   1 2 4
    // While this kind of partitioning makes things somewhat more complicated, it
    // ensures good utilization of all SMs for many kinds of shape and GPU
    // configurations, while requiring as few slow global cross-threadblock
    // reductions as possible.

    // For larger GEMMs we run multiple batchsize 64 versions in parallel for a
    // better partitioning with less reductions
    int parallel = 1;
    if (prob_m > 16 * thread_m_blocks) {
        parallel = prob_m / (16 * thread_m_blocks);
        prob_m   = 16 * thread_m_blocks;
    }

    int k_tiles = prob_k / 16 / thread_k_blocks;
    int n_tiles = prob_n / 16 / thread_n_blocks;
    int iters   = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);
    // Ensure that the number of tiles in each stripe is a multiple of the
    // groupsize; this avoids an annoying special case where a stripe starts in
    // the middle of group.
    if constexpr (group_blocks != -1)
        iters = (group_blocks / thread_k_blocks) * ceildiv(iters, (group_blocks / thread_k_blocks));

    int slice_row     = (iters * blockIdx.x) % k_tiles;
    int slice_col_par = (iters * blockIdx.x) / k_tiles;
    int slice_col     = slice_col_par;
    int slice_iters;      // number of threadblock tiles in the current slice
    int slice_count = 0;  // total number of active threadblocks in the current slice
    int slice_idx;        // index of threadblock in current slice; numbered bottom to
                          // top

    // We can easily implement parallel problem execution by just remapping
    // indices and advancing global pointers
    if (slice_col_par >= n_tiles) {
        A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 16;
        C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 4;
        D += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
        s1 += (slice_col_par / n_tiles) * 16 * thread_m_blocks;
        locks += (slice_col_par / n_tiles) * n_tiles;
        slice_col = slice_col_par % n_tiles;
    }

    // Compute all information about the current slice which is required for
    // synchronization.
    auto init_slice = [&]() {
        slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
        if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
            slice_iters = 0;
        if (slice_iters == 0)
            return;
        if (slice_row + slice_iters > k_tiles)
            slice_iters = k_tiles - slice_row;
        slice_count   = 1;
        slice_idx     = 0;
        int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
        if (col_first <= k_tiles * (slice_col_par + 1)) {
            int col_off = col_first - k_tiles * slice_col_par;
            slice_count = ceildiv(k_tiles - col_off, iters);
            if (col_off > 0)
                slice_count++;
            int delta_first = iters * blockIdx.x - col_first;
            if (delta_first < 0 || (col_off == 0 && delta_first == 0))
                slice_idx = slice_count - 1;
            else {
                slice_idx = slice_count - 1 - delta_first / iters;
                if (col_off > 0)
                    slice_idx--;
            }
        }
        if (slice_col == n_tiles) {
            A += 16 * thread_m_blocks * prob_k / 16;
            C += 16 * thread_m_blocks * prob_n / 4;
            D += 16 * thread_m_blocks * prob_n / 8;
            s1 += 16 * thread_m_blocks;
            locks += n_tiles;
            slice_col = 0;
        }
    };
    init_slice();

    int a_gl_stride = prob_k / 16;  // stride of the A matrix in global memory
    // We typically use `constexpr` to indicate that this value is a compile-time
    // constant
    constexpr int a_sh_stride     = 16 * thread_k_blocks / 16;  // stride of an A matrix tile in shared memory
    constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 16;  // delta between subsequent A tiles in global memory
    int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);  // between subsequent accesses within a tile
    constexpr int a_sh_wr_delta   = a_sh_stride * (threads / a_gl_rd_delta_o);     // between shared memory writes
    constexpr int a_sh_rd_delta_o = 1 * ((threads / 32) / (thread_n_blocks / 4));  // between shared memory tile reads
    constexpr int a_sh_rd_delta_i = a_sh_stride * 16;                              // within a shared memory tile
    constexpr int a_sh_stage      = a_sh_stride * (16 * thread_m_blocks);          // overall size of a tile
    constexpr int a_sh_wr_iters   = ceildiv(a_sh_stage,
                                          a_sh_wr_delta);  // number of shared write iterations for a tile

    int           b_gl_stride     = 16 * prob_n / 32;
    constexpr int b_sh_stride     = 32 * thread_n_blocks / 4;
    int           b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
    int           b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
    constexpr int b_sh_wr_delta   = threads;
    constexpr int b_sh_rd_delta   = threads;
    constexpr int b_sh_stage      = b_sh_stride * thread_k_blocks;
    constexpr int b_sh_wr_iters   = b_sh_stage / b_sh_wr_delta;

    constexpr int s1_sh_stride = 16 * thread_m_blocks;

    constexpr int s2_sh_stride = 16 * thread_n_blocks / 4;

    int           s3_gl_stride   = prob_n / 8;
    constexpr int s3_sh_stride   = 16 * thread_n_blocks / 8;
    constexpr int s3_sh_stage    = s3_sh_stride;
    int           s3_gl_rd_delta = s3_gl_stride;

    // Global A read index of current thread.
    int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
    a_gl_rd += a_gl_rd_delta_o * slice_row;
    // Shared write index of current thread.
    int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
    // Shared read index.
    // NOTE(HandH1998): int8 input a only need 16 threads to load 16x16 matrix
    int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16);
    a_sh_rd += 1 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

    int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
    b_gl_rd += b_sh_stride * slice_col;
    b_gl_rd += b_gl_rd_delta_o * slice_row;
    int b_sh_wr = threadIdx.x;
    int b_sh_rd = threadIdx.x;

    int s1_gl_rd = threadIdx.x;
    // NOTE(HandH1998): activation scale s1 need shuffle to [0, 8, 1, 9, 2, 10, 3,
    // 11, 4, 12, 5, 13, 6, 14, 7, 15] for example, 0, 8 row scales serve for
    // thread 0, 1, 2, 3. For more details, refer to mma operand A layout as s1's
    // size is not fixed, we can not shuffle before inference we shuffle it when
    // fetching s1 from global memory to shared memory, that's why s1_sh_wr is
    // like this
    int  s1_sh_wr      = (threadIdx.x / 16) * 16 + (threadIdx.x % 8) * 2 + (threadIdx.x % 16) / 8;
    int  s1_sh_rd      = (threadIdx.x % 32) / 4;
    bool s1_sh_wr_pred = threadIdx.x < prob_m;

    int  s2_gl_rd      = s2_sh_stride * slice_col + threadIdx.x;
    int  s2_sh_wr      = threadIdx.x;
    int  s2_sh_rd      = 16 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + 2 * ((threadIdx.x % 32) % 4);
    bool s2_sh_wr_pred = threadIdx.x < s2_sh_stride;

    int  s3_gl_rd, s3_sh_wr, s3_sh_rd;
    bool s3_sh_wr_pred;
    if constexpr (group_blocks != -1) {
        s3_gl_rd =
            s3_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s3_sh_stride * slice_col + threadIdx.x;
        s3_sh_wr = threadIdx.x;
        // NOTE(HandH1998): s3_sh_rd is related to mma output C
        s3_sh_rd      = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
        s3_sh_wr_pred = threadIdx.x < s3_sh_stride;
    }

    // Precompute which thread should not read memory in which iterations; this is
    // needed if there are more threads than required for a certain tilesize or
    // when the batchsize is not a multiple of 16.
    bool a_sh_wr_pred[a_sh_wr_iters];
#pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++)
        a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

    // To ensure that writing and reading A tiles to/from shared memory, the
    // latter in fragment format, is fully bank conflict free, we need to use a
    // rather fancy XOR-based layout. The key here is that neither reads nor
    // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
    // same shared memory banks. Further, it seems (based on NSight-Compute) that
    // each warp must also write a consecutive memory segment?
    auto transform_a = [&](int i) {
        int row = i / a_gl_rd_delta_o;
        return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
    };
    // Since the computation of this remapping is non-trivial and, due to our main
    // loop unrolls, all shared memory accesses are static, we simply precompute
    // both transformed reads and writes.
    int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++)
        a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
    int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
#pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
        for (int j = 0; j < thread_m_blocks; j++)
            a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
    }

    // Since B-accesses have non-constant stride they have to be computed at
    // runtime; we break dependencies between subsequent accesses with a tile by
    // maintining multiple pointers (we have enough registers), a tiny
    // optimization.
    const int4* B_ptr[b_sh_wr_iters];
#pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++)
        B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

    extern __shared__ int4 sh[];
    // Shared memory storage for global fetch pipelines.
    // NOTE(HandH1998): stages need >= 4, otherwise, sh_s1 = sh + max(stages *
    // a_sh_stage + stages * b_sh_stage, 4 * stages * a_sh_stage)
    int4* sh_a  = sh;
    int4* sh_b  = sh_a + (stages * a_sh_stage);
    int4* sh_s1 = sh_b + (stages * b_sh_stage);
    int4* sh_s2 = sh_s1 + s1_sh_stride;
    int4* sh_s3 = sh_s2 + s2_sh_stride;

    // Register storage for double buffer of shared memory reads.
    FragA         frag_a[2][thread_m_blocks];
    I4            frag_b_quant[2];
    FragC         frag_c[thread_m_blocks][4][2];
    FragS_GROUP   frag_s3[2][4];
    FragS_CHANNEL frag_s1[thread_m_blocks];
    FragS_CHANNEL frag_s2[2][4];

    // Zero accumulators.
    auto zero_accums = [&]() {
#pragma unroll
        for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
            reinterpret_cast<int*>(frag_c)[i] = 0;
    };

    // Asynchronously fetch the next A, B and s tile from global to the next
    // shared memory pipeline location.
    auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
        if (pred) {
            int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
            for (int i = 0; i < a_sh_wr_iters; i++) {
                cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]],
                               &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
                               a_sh_wr_pred[i]);
            }
            int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
            for (int i = 0; i < b_sh_wr_iters; i++) {
                cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
                B_ptr[i] += b_gl_rd_delta_o;
            }
            // Only fetch scales if this tile starts a new group
            if constexpr (group_blocks != -1) {
                if (pipe % (group_blocks / thread_k_blocks) == 0) {
                    int4* sh_s3_stage = sh_s3 + s3_sh_stage * pipe;
                    if (s3_sh_wr_pred)
                        cp_async4(&sh_s3_stage[s3_sh_wr], &s3[s3_gl_rd]);
                    s3_gl_rd += s3_gl_rd_delta;
                }
            }
        }
        // Insert a fence even when we are winding down the pipeline to ensure that
        // waiting is also correct at this point.
        cp_async_fence();
    };

    // Wait until the next thread tile has been loaded to shared memory.
    auto wait_for_stage = [&]() {
        // We only have `stages - 2` active fetches since we are double buffering
        // and can only issue the next fetch when it is guaranteed that the previous
        // shared memory load is fully complete (as it may otherwise be
        // overwritten).
        cp_async_wait<stages - 2>();
        __syncthreads();
    };

    // Load the next sub-tile from the current location in the shared memory pipe
    // into the current register buffer.
    auto fetch_to_registers = [&](int k, int pipe) {
        // It may seem inefficient that we reload the groups for every sub-tile;
        // however, this does not seem to be a significant bottleneck, while some
        // theoretically better attempts have lead to bad instruction ordering by
        // the compiler and correspondingly a noticeable drop in performance.
        if constexpr (group_blocks != -1) {
            int4* sh_s3_stage =
                sh_s3 + s3_sh_stage * ((group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks)));
            reinterpret_cast<int4*>(&frag_s3[k % 2])[0] = sh_s3_stage[s3_sh_rd];
        }
        int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
        for (int i = 0; i < thread_m_blocks; i++)
            ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
        int4* sh_b_stage    = sh_b + b_sh_stage * pipe;
        frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
    };

    // Execute the actual tensor core matmul of a sub-tile.
    auto matmul = [&](int k) {
// We have the m dimension as the inner loop in order to encourage overlapping
// dequantization and matmul operations.
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int b_quant = frag_b_quant[k % 2][j];
            // int b_quant_shift = b_quant << 4;
            FragB frag_b0, frag_b1;
            // If there are no groups, we can just scale the final output once and can
            // avoid doing so for each weight.
            if constexpr (group_blocks != -1) {
                int b_quant_shift = b_quant >> 8;
                frag_b0           = dequant_per_group(b_quant, frag_s3[k % 2][j], 0);
                frag_b1           = dequant_per_group(b_quant_shift, frag_s3[k % 2][j], 1);
            }
            else {
                int b_quant_shift = b_quant << 4;
                frag_b0           = dequant_per_channel(b_quant);
                frag_b1           = dequant_per_channel(b_quant_shift);
            }
#pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
                mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
                mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
            }
        }
    };

    // Since we slice across the k dimension of a tile in order to increase the
    // number of warps while keeping the n dimension of a tile reasonable, we have
    // multiple warps that accumulate their partial sums of the same output
    // location; which we have to reduce over in the end. We do in shared memory.
    auto thread_block_reduce = [&]() {
        constexpr int red_off = threads / b_sh_stride / 2;
        if (red_off >= 1) {
            int           red_idx       = threadIdx.x / b_sh_stride;
            constexpr int red_sh_stride = b_sh_stride * 4 * 2;
            constexpr int red_sh_delta  = b_sh_stride;
            int           red_sh_rd     = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);

            // Parallel logarithmic shared memory reduction. We make sure to avoid any
            // unnecessary read or write iterations, e.g., for two warps we write only
            // once by warp 1 and read only once by warp 0.

#pragma unroll
            for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
#pragma unroll
                for (int i = red_off; i > 0; i /= 2) {
                    if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
                        for (int j = 0; j < 4 * 2; j++) {
                            int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
                            if (i < red_off) {
                                int* c_rd = reinterpret_cast<int*>(&sh[red_sh_delta * j + red_sh_rd]);
                                int* c_wr = reinterpret_cast<int*>(&sh[red_sh_wr]);
#pragma unroll
                                for (int k = 0; k < 4; k++)
                                    reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
                            }
                            sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
                        }
                    }
                    __syncthreads();
                }
                if (red_idx == 0) {
#pragma unroll
                    for (int i = 0; i < 4 * 2; i++) {
                        int* c_rd = reinterpret_cast<int*>(&sh[red_sh_delta * i + red_sh_rd]);
#pragma unroll
                        for (int j = 0; j < 4; j++)
                            reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
                    }
                }
                __syncthreads();
            }
        }
    };

    // Since multiple threadblocks may process parts of the same column slice, we
    // finally have to globally reduce over the results. As the striped
    // partitioning minimizes the number of such reductions and our outputs are
    // usually rather small, we perform this reduction serially in L2 cache.
    // global_reduce works on INT32 elements, which are the results of INT8 GEMM.
    // This is why we need another INT32 maxtrix `C` to reduce instead of the
    // original half matrix `D`.
    auto global_reduce = [&](bool first = false, bool last = false) {
        // We are very careful here to reduce directly in the output buffer to
        // maximize L2 cache utilization in this step. To do this, we write out
        // results in FP16 (but still reduce with FP32 compute).
        constexpr int active_threads = 32 * thread_n_blocks / 4;
        if (threadIdx.x < active_threads) {
            int c_gl_stride     = prob_n / 4;
            int c_gl_wr_delta_o = 8 * c_gl_stride;
            int c_gl_wr_delta_i = 8 * (active_threads / 32);
            int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 8 * (threadIdx.x / 32) + (threadIdx.x % 4) * 2;
            c_gl_wr += (4 * thread_n_blocks) * slice_col;
            constexpr int c_sh_wr_delta = active_threads * 2;
            int           c_sh_wr       = 2 * threadIdx.x;

            int row = (threadIdx.x % 32) / 4;

            if (!first) {
// Interestingly, doing direct global accesses here really seems to mess up
// the compiler and lead to slowdowns, hence we also use async-copies even
// though these fetches are not actually asynchronous.
#pragma unroll
                for (int i = 0; i < thread_m_blocks * 4; i++) {
                    cp_async4_pred(&sh[c_sh_wr + c_sh_wr_delta * i],
                                   &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)],
                                   i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
                    cp_async4_pred(&sh[c_sh_wr + c_sh_wr_delta * i + 1],
                                   &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2) + 1],
                                   i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
                }
                cp_async_fence();
                cp_async_wait<0>();
            }

#pragma unroll
            for (int i = 0; i < thread_m_blocks * 4; i++) {
                if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
                    if (!first) {
                        int4 d_red1 = sh[c_sh_wr + i * c_sh_wr_delta];
                        int4 d_red2 = sh[c_sh_wr + i * c_sh_wr_delta + 1];
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            reinterpret_cast<int*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] +=
                                reinterpret_cast<int*>(&d_red1)[j];
                        }
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            reinterpret_cast<int*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * (j + 4) + (i % 4)] +=
                                reinterpret_cast<int*>(&d_red2)[j];
                        }
                    }
                    if (!last) {
                        int4 d1, d2;
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            reinterpret_cast<int*>(&d1)[j] =
                                reinterpret_cast<int*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)];
                        }
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            reinterpret_cast<int*>(&d2)[j] =
                                reinterpret_cast<int*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * (j + 4) + (i % 4)];
                        }
                        C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)]     = d1;
                        C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2) + 1] = d2;
                    }
                }
            }
        }
    };

    // Write out the reduce final result in the correct layout. We only actually
    // reshuffle matrix fragments in this step, the reduction above is performed
    // in fragment layout.
    auto write_result = [&]() {
        int           d_gl_stride   = prob_n / 8;
        constexpr int d_sh_stride   = 2 * thread_n_blocks + 1;
        int           d_gl_wr_delta = d_gl_stride * (threads / (2 * thread_n_blocks));
        constexpr int d_sh_rd_delta = d_sh_stride * (threads / (2 * thread_n_blocks));

        int d_gl_wr = d_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
        d_gl_wr += (2 * thread_n_blocks) * slice_col;
        int d_sh_wr = (4 * d_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
        d_sh_wr += 32 * (threadIdx.x / 32);
        int d_sh_rd = d_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

        int d_gl_wr_end = d_gl_stride * prob_m;

        // We first reorder in shared memory to guarantee the most efficient final
        // global write patterns
        auto write = [&](int idx, int c0, int c1, float a_s, FragS_CHANNEL& w_s) {
            float2 deq_res;
            deq_res.x         = int32_to_float(c0) * w_s[0] * a_s;
            deq_res.y         = int32_to_float(c1) * w_s[1] * a_s;
            ((half2*)sh)[idx] = float2_to_half2(deq_res);
        };

        if (threadIdx.x / 32 < thread_n_blocks / 4) {
#pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    int wr = d_sh_wr + 8 * j;
                    write(wr + (4 * d_sh_stride) * 0 + 0,
                          frag_c[i][j][0][0],
                          frag_c[i][j][0][1],
                          frag_s1[i][0],
                          frag_s2[j / 2][2 * (j % 2) + 0]);
                    write(wr + (4 * d_sh_stride) * 8 + 0,
                          frag_c[i][j][0][2],
                          frag_c[i][j][0][3],
                          frag_s1[i][1],
                          frag_s2[j / 2][2 * (j % 2) + 0]);
                    write(wr + (4 * d_sh_stride) * 0 + 4,
                          frag_c[i][j][1][0],
                          frag_c[i][j][1][1],
                          frag_s1[i][0],
                          frag_s2[j / 2][2 * (j % 2) + 1]);
                    write(wr + (4 * d_sh_stride) * 8 + 4,
                          frag_c[i][j][1][2],
                          frag_c[i][j][1][3],
                          frag_s1[i][1],
                          frag_s2[j / 2][2 * (j % 2) + 1]);
                }
                d_sh_wr += 16 * (4 * d_sh_stride);
            }
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
            if (d_gl_wr < d_gl_wr_end) {
                D[d_gl_wr] = sh[d_sh_rd];
                d_gl_wr += d_gl_wr_delta;
                d_sh_rd += d_sh_rd_delta;
            }
        }
    };

    // Start global fetch and register load pipelines.
    auto start_pipes = [&]() {
#pragma unroll
        for (int i = 0; i < stages - 1; i++)
            fetch_to_shared(i, i, i < slice_iters);
        zero_accums();
        wait_for_stage();
        fetch_to_registers(0, 0);
        a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    };
    start_pipes();

    // Main loop.
    while (slice_iters) {
// We unroll over both the global fetch and the register load pipeline to
// ensure all shared memory accesses are static. Note that both pipelines have
// even length meaning that the next iteration will always start at index 0.
#pragma unroll
        for (int pipe = 0; pipe < stages;) {
#pragma unroll
            for (int k = 0; k < b_sh_wr_iters; k++) {
                fetch_to_registers(k + 1, pipe % stages);
                if (k == b_sh_wr_iters - 2) {
                    fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
                    pipe++;
                    wait_for_stage();
                }
                matmul(k);
            }
            slice_iters--;
            if (slice_iters == 0)
                break;
        }
        a_gl_rd += a_gl_rd_delta_o * stages;

        // Process results and, if necessary, proceed to the next column slice.
        // While this pattern may not be the most readable, other ways of writing
        // the loop seemed to noticeably worse performance after compilation.
        if (slice_iters == 0) {
            cp_async_wait<0>();
            bool last = slice_idx == slice_count - 1;
            // For per-column scales, we only fetch them here in the final step before
            // write-out
            if (last) {
                if (s1_sh_wr_pred) {
                    cp_async1(&sh_s1[s1_sh_wr], &s1[s1_gl_rd]);
                }
                if (s2_sh_wr_pred) {
                    cp_async4(&sh_s2[s2_sh_wr], &s2[s2_gl_rd]);
                }
                cp_async_fence();
            }
            thread_block_reduce();
            if (last) {
                cp_async_wait<0>();
                __syncthreads();
                if (threadIdx.x / 32 < thread_n_blocks / 4) {
#pragma unroll
                    for (int i = 0; i < thread_m_blocks; i++) {
                        frag_s1[i][0] = *reinterpret_cast<float*>(&sh_s1[16 * i + 2 * s1_sh_rd]);
                        frag_s1[i][1] = *reinterpret_cast<float*>(&sh_s1[16 * i + 2 * s1_sh_rd + 1]);
                    }
                    reinterpret_cast<int4*>(&frag_s2)[0] = sh_s2[s2_sh_rd + 0];
                    reinterpret_cast<int4*>(&frag_s2)[1] = sh_s2[s2_sh_rd + 1];
                    reinterpret_cast<int4*>(&frag_s2)[2] = sh_s2[s2_sh_rd + 8];
                    reinterpret_cast<int4*>(&frag_s2)[3] = sh_s2[s2_sh_rd + 9];
                }
            }
            if (slice_count > 1) {  // only globally reduce if there is more than one
                                    // block in a slice
                barrier_acquire(&locks[slice_col], slice_idx);
                global_reduce(slice_idx == 0, last);
                barrier_release(&locks[slice_col], last);
            }
            if (last)  // only the last block in a slice actually writes the result
                write_result();
            slice_row = 0;
            slice_col_par++;
            slice_col++;
            init_slice();
            if (slice_iters) {
                a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
#pragma unroll
                for (int i = 0; i < b_sh_wr_iters; i++)
                    B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
                if (slice_col == 0) {
#pragma unroll
                    for (int i = 0; i < b_sh_wr_iters; i++)
                        B_ptr[i] -= b_gl_stride;
                }
                s3_gl_rd = s3_sh_stride * slice_col + threadIdx.x;
                s2_gl_rd = s2_sh_stride * slice_col + threadIdx.x;
                start_pipes();
            }
        }
    }
}

#else

template<const int threads,          // number of threads in a threadblock
         const int thread_m_blocks,  // number of 16x16 blocks in the m
                                     // dimension (batchsize) of the
                                     // threadblock
         const int thread_n_blocks,  // same for n dimension (output)
         const int thread_k_blocks,  // same for k dimension (reduction)
         const int stages,           // number of stages for the async global->shared
                                     // fetch pipeline
         const int group_blocks      // number of consecutive 16x16 blocks
                                     // with a separate quantization scale
         >
__global__ void Marlin(const int4* __restrict__ A,    // int8 input matrix of shape mxk
                       const int4* __restrict__ B,    // 4bit quantized weight matrix of shape kxn
                       int4* __restrict__ C,          // int32 global_reduce buffer of shape
                                                      // (max_par*16*4)xn, as int8 tensor core's output is
                                                      // int32 dtype
                       int4* __restrict__ D,          // fp16 output buffer of shape mxn
                       const float* __restrict__ s1,  // fp32 activation per-token quantization
                                                      // scales of shape mx1
                       const int4* __restrict__ s2,   // fp32 weight per-channel quantization scales
                                                      // of shape 1xn
                       const int4* __restrict__ s3,   // fp16 weight per-group quantization scales
                                                      // of shape (k/groupsize)xn, when
                                                      // group_blocks=-1, it should be nullptr
                       int  prob_m,                   // batch dimension m
                       int  prob_n,                   // output dimension n
                       int  prob_k,                   // reduction dimension k
                       int* locks                     // extra global storage for barrier synchronization
)
{
    // Marlin is not implemented yet for SM < 8.0
    assert(false);
    return;
}

#endif

#define __CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS, NUM_THREADS)                        \
    else if (thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS                                  \
             && thread_k_blocks == THREAD_K_BLOCKS && group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS)      \
    {                                                                                                                  \
        cudaFuncSetAttribute(                                                                                          \
            Marlin<NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages, GROUP_BLOCKS>,         \
            cudaFuncAttributeMaxDynamicSharedMemorySize,                                                               \
            max_shared_mem);                                                                                           \
        Marlin<NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages, GROUP_BLOCKS>              \
            <<<blocks, NUM_THREADS, max_shared_mem, stream>>>(A, B, C, D, s1, s2, s3, prob_m, prob_n, prob_k, locks);  \
    }

#define CALL_IF(N_BLOCKS, K_BLOCKS, NUM_THREADS)                                                                       \
    __CALL_IF(1, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                                                                  \
    __CALL_IF(1, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)                                                                   \
    __CALL_IF(1, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                                                                  \
    __CALL_IF(1, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)                                                                   \
    __CALL_IF(2, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                                                                  \
    __CALL_IF(2, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)                                                                   \
    __CALL_IF(3, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                                                                  \
    __CALL_IF(3, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)                                                                   \
    __CALL_IF(4, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                                                                  \
    __CALL_IF(4, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)

void invokeMarlinQQQGemm(const int4*  A,
                         const int4*  B,
                         int4*        C,
                         int4*        D,
                         const float* s1,
                         const int4*  s2,
                         const int4*  s3,
                         int          prob_m,
                         int          prob_n,
                         int          prob_k,
                         int*         locks,
                         int          thread_n_blocks,
                         int          thread_k_blocks,
                         int          group_blocks,
                         int          num_threads,
                         cudaStream_t stream)
{
    int tot_m        = prob_m;
    int tot_m_blocks = ceildiv(tot_m, 16);
    int pad          = 16 * tot_m_blocks - tot_m;

    int dev = 0;
    cudaGetDevice(&dev);

    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int blocks = sms;

    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    for (int i = 0; i < tot_m_blocks; i += 4) {
        int thread_m_blocks = tot_m_blocks - i;
        prob_m              = tot_m - 16 * i;
        int par             = 1;
        if (thread_m_blocks > 4) {
            // Note that parallel > 1 currently only works for inputs without any
            // padding
            par = (16 * thread_m_blocks - pad) / 64;
            if (par > max_par)
                par = max_par;
            prob_m = 64 * par;
            i += 4 * (par - 1);
            thread_m_blocks = 4;
        }

        // For compilation speed, we only define the kernel configurations that have
        // seemed useful (in terms of performance) in our testing, however many more
        // are, in principle, possible.
        if (false) {}
        CALL_IF(8, 8, 256)
        CALL_IF(16, 4, 256)
        CALL_IF(8, 4, 128)
        CALL_IF(4, 8, 128)
        else
        {
            throw std::runtime_error("Unsupported shapes: MKN = [" + str(prob_m) + ", " + str(prob_k) + ", "
                                     + str(prob_n) + "]" + ", thread_m_blocks = " + str(thread_m_blocks)
                                     + ", thread_n_blocks = " + str(thread_n_blocks)
                                     + ", thread_k_blocks = " + str(thread_k_blocks));
        }

        A += 16 * thread_m_blocks * (prob_k / 16) * par;
        D += 16 * thread_m_blocks * (prob_n / 8) * par;
        s1 += 16 * thread_m_blocks * par;
    }
}

}  // namespace marlin_qqq
}  // namespace turbomind
