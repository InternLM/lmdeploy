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

#include "marlin_qqq_gemm.h"
#include "marlin_qqq_gemm_kernel.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/string_utils.h"

namespace turbomind {
namespace marlin_qqq {

void MarlinQQQGemm::allocateBuffer(size_t workspace_size, size_t reduce_buf_size)
{
    workspace_buf_      = (int*)allocator_->reMalloc(workspace_buf_, sizeof(int) * workspace_size, true);
    reduce_buf_         = (int*)allocator_->reMalloc(reduce_buf_, sizeof(int) * reduce_buf_size, false);
    is_allocate_buffer_ = true;
}

void MarlinQQQGemm::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)&workspace_buf_);
        allocator_->free((void**)&reduce_buf_);
        is_allocate_buffer_ = false;
    }
}

bool MarlinQQQGemm::is_valid_config(thread_config_t const& th_config, int prob_m, int prob_n, int prob_k)
{
    // Sanity
    if (th_config.thread_k == -1 || th_config.thread_n == -1 || th_config.num_threads == -1) {
        return false;
    }

    // Verify K/N are divisible by thread K/N
    if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
        return false;
    }

    // thread_k can be only 128 or 64 (because it must be less than groupsize
    // which is 128)
    if (th_config.thread_k != 128 && th_config.thread_k != 64) {
        return false;
    }

    // Verify min for thread K/N
    if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
        return false;
    }

    // num_threads must be at least 128 (= 4 warps)
    if (th_config.num_threads < 128) {
        return false;
    }

    return true;
}

MarlinQQQGemm::thread_config_t MarlinQQQGemm::determine_thread_config(int prob_m, int prob_n, int prob_k)
{
    if (prob_m <= 16) {
        for (auto th_config : small_batch_thread_configs) {
            if (is_valid_config(th_config, prob_m, prob_n, prob_k)) {
                return th_config;
            }
        }
    }
    else {
        for (auto th_config : large_batch_thread_configs) {
            if (is_valid_config(th_config, prob_m, prob_n, prob_k)) {
                return th_config;
            }
        }
    }

    return thread_config_t{-1, -1, -1};
}

#define __CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS, NUM_THREADS)                        \
    else if (thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS                                  \
             && thread_k_blocks == THREAD_K_BLOCKS && group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS)      \
    {                                                                                                                  \
        cudaFuncSetAttribute(                                                                                          \
            Marlin<NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages, GROUP_BLOCKS>,         \
            cudaFuncAttributeMaxDynamicSharedMemorySize,                                                               \
            max_shared_mem);                                                                                           \
        Marlin<NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages, GROUP_BLOCKS>              \
            <<<blocks, NUM_THREADS, max_shared_mem, stream>>>(                                                         \
                A_ptr, B_ptr, C_ptr, D_ptr, s1_ptr, s2_ptr, s3_ptr, prob_m, prob_n, prob_k, locks);                    \
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

void MarlinQQQGemm::Run(half*         D,
                        const int8_t* A,
                        const uint*   B,
                        const float*  s1,
                        const float*  s2,
                        const half*   s3,
                        int           prob_m,
                        int           prob_n,
                        int           prob_k,
                        int           groupsize,
                        cudaStream_t  stream)
{

    // Set thread config
    thread_config_t th_config = determine_thread_config(prob_m, prob_n, prob_k);
    if (!is_valid_config(th_config, prob_m, prob_n, prob_k)) {
        throw std::runtime_error(
            fmtstr("Invalid thread config: thread_k = %d, thread_n = %d, num_threads = %d for MKN = [%d, %d, %d]",
                   th_config.thread_k,
                   th_config.thread_n,
                   th_config.num_threads,
                   prob_m,
                   prob_k,
                   prob_n));
    }

    int num_threads = th_config.num_threads;
    int thread_k    = th_config.thread_k;
    int thread_n    = th_config.thread_n;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;
    // QQQ only supports groupsize = -1 or 128
    int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;

    if (group_blocks != -1) {
        FT_CHECK_WITH_INFO(prob_k % group_blocks == 0,
                           fmtstr("prob_k = %d is not divisible by group_blocks = %d", prob_k, group_blocks));
    }
    FT_CHECK_WITH_INFO(prob_k % tile_size == 0,
                       fmtstr("prob_k = %d is not divisible by tile_size = %d", prob_k, tile_size));
    FT_CHECK_WITH_INFO(prob_n % min_thread_n == 0,
                       fmtstr("prob_n = %d is not divisible by min_thread_n = %d", prob_n, min_thread_n));

    if (reduce_buf_ == nullptr || workspace_buf_ == nullptr) {
        size_t workspace_size  = (prob_n / min_thread_n) * max_par;
        size_t reduce_buf_size = max_par * 64 * prob_n;
        allocateBuffer(workspace_size, reduce_buf_size);
    }

    const int4*  A_ptr  = (const int4*)A;
    const int4*  B_ptr  = (const int4*)B;
    int4*        C_ptr  = (int4*)reduce_buf_;
    int4*        D_ptr  = (int4*)D;
    const float* s1_ptr = (const float*)s1;
    const int4*  s2_ptr = (const int4*)s2;
    const int4*  s3_ptr = (const int4*)s3;
    int*         locks  = workspace_buf_;
    invokeMarlinQQQGemm(A_ptr,
                        B_ptr,
                        C_ptr,
                        D_ptr,
                        s1_ptr,
                        s2_ptr,
                        s3_ptr,
                        prob_m,
                        prob_n,
                        prob_k,
                        locks,
                        thread_n_blocks,
                        thread_k_blocks,
                        group_blocks,
                        num_threads,
                        stream);
    sync_check_cuda_error();
}
}  // namespace marlin_qqq
}  // namespace turbomind
