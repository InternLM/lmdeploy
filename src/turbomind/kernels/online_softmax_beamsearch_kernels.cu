/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "src/fastertransformer/kernels/online_softmax_beamsearch_kernels.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

#define DO_SPLIT_SMALL_TOP_K_SOFTMAX
static const int SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE = 256;

#define TOPK_FP16_STORAGE 0

template<typename T>
__device__ __forceinline__ T apply_length_penalty(T log_prob, int length, float length_penalty)
{
    // score = log(prob) / (length)^length_penalty.
    if (length_penalty == 0.0f || length == 1) {
        return log_prob;
    }
    return log_prob / static_cast<T>(powf(length, length_penalty));
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void batch_topK_kernel(int* topk_tmp_id_buf, T* topk_tmp_val_buf, int* id_buf)
{
    int            thread_id = threadIdx.x;
    int            block_id  = blockIdx.x;
    TopK<T, MAX_K> partial;
    if (thread_id == 0) {
        for (int i = 0; i < MAX_K; ++i) {
            partial.p[i] = -1;
            partial.u[i] = -FLT_MAX;
        }

        int index = block_id * MAX_K * MAX_K;
        for (int i = 0; i < MAX_K * MAX_K; i++) {
            partial.insert((T)topk_tmp_val_buf[index + i], topk_tmp_id_buf[index + i]);
        }

        index = block_id * MAX_K;
        for (int i = 0; i < MAX_K; i++) {
            id_buf[index + i] = partial.p[i];
        }
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void batch_topK_kernel(const int* __restrict topk_tmp_id_buf,
                                                                      const T* __restrict topk_tmp_val_buf,
                                                                      int* __restrict id_buf,
                                                                      T* __restrict val_buf)
{
    int            thread_id = threadIdx.x;
    int            block_id  = blockIdx.x;
    TopK<T, MAX_K> partial;
    if (thread_id == 0) {
        for (int i = 0; i < MAX_K; ++i) {
            partial.p[i] = -1;
            partial.u[i] = -FLT_MAX;
        }

        int index = block_id * MAX_K * MAX_K;
        for (int i = 0; i < MAX_K * MAX_K; i++) {
            partial.insert((T)topk_tmp_val_buf[index + i], topk_tmp_id_buf[index + i]);
        }

        index = block_id * MAX_K;
        for (int i = 0; i < MAX_K; i++) {
            id_buf[index + i]  = partial.p[i];
            val_buf[index + i] = partial.u[i];
        }
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void batch_topk_kernel(const int* __restrict x,
                                                                      const T* __restrict y,
                                                                      int* __restrict z,
                                                                      float* __restrict v,
                                                                      float*         output_log_probs,
                                                                      const bool*    finished,
                                                                      const int*     sequence_lengths,
                                                                      BeamHypotheses beam_hyps,
                                                                      const int      V,
                                                                      const int      K,
                                                                      const int      vocab_size,
                                                                      const float    length_penalty,
                                                                      const T        diversity_rate)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition x, y to data for the current vector
    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ int                               selected_beams;
    __shared__ float                             old_cum_log_probs[MAX_K];

    if (thread_id == 0) {
        selected_beams = 0;
    }
    if (thread_id < K) {
        old_cum_log_probs[thread_id] = v[vector_id * K + thread_id];
    }
    __syncthreads();
    if (beam_hyps.num_beams != nullptr) {
        const int global_batch_idx = beam_hyps.ite * beam_hyps.local_batch_size + vector_id;
        if (beam_hyps.num_beams[global_batch_idx] == 0 && thread_id == 0) {
            beam_hyps.min_normed_scores[global_batch_idx] = FLT_MAX;
        }
        else if (beam_hyps.num_beams[global_batch_idx] == K) {
            return;
        }
    }

    TopK<T, MAX_K> partial;
    for (int i = 0; i < MAX_K; ++i) {
        partial.p[i] = -1;
        partial.u[i] = -FLT_MAX;
    }

    for (int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE) {
        int i    = elem_id % K;
        T   elem = length_penalty == 0.0f ? y[elem_id] :
                                            apply_length_penalty(y[elem_id],
                                                               finished[vector_id] ? sequence_lengths[vector_id] :
                                                                                       sequence_lengths[vector_id] + 1,
                                                               length_penalty);
        elem += diversity_rate * (T)i;
        int elem_idx = elem_id;  // x[elem_id];
        partial.insert(elem, elem_idx);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0) {
        z += vector_id * K;
        v += vector_id * K;

        for (int i = 0; i < MAX_K; ++i) {
            if (beam_hyps.num_beams != nullptr && x[total.p[i]] % vocab_size == beam_hyps.end_ids[vector_id]) {
                // if beam_token does not belong to top num_beams tokens, it should not be added. Refer from
                // https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/generation_beam_search.py#L257
                if (i >= K) {
                    // do nothing
                }
                else {
                    const int   global_batch_idx = beam_hyps.ite * beam_hyps.local_batch_size + vector_id;
                    const float normed_score     = (float)total.u[i];
                    const int   num_beam         = beam_hyps.num_beams[global_batch_idx];
                    int         beam_idx         = num_beam;
                    // If there are beam_width finished sentences, check that the score of selected candidatet
                    // is higher than min_normed_score or not. If current score is better, replace worst one
                    // and update the min_normed_score.
                    if (num_beam == K) {
                        if (normed_score < beam_hyps.min_normed_scores[global_batch_idx]) {
                            // end the tracing and exist this for loop
                            selected_beams = K;
                            break;
                        }
                        else {
                            // find the beam index which's score = min_normed_score, erase it.
                            for (int j = 0; j < K; j++) {
                                if (beam_hyps.normed_scores[global_batch_idx * (K * 2) + j]
                                    == beam_hyps.min_normed_scores[global_batch_idx]) {
                                    beam_idx = j;
                                    beam_hyps.num_beams[global_batch_idx]--;

                                    beam_hyps.min_normed_scores[global_batch_idx]           = FLT_MAX;
                                    beam_hyps.normed_scores[global_batch_idx * (K * 2) + j] = normed_score;
                                    for (int l = 0; l < K; l++) {
                                        beam_hyps.min_normed_scores[global_batch_idx] =
                                            min(beam_hyps.min_normed_scores[global_batch_idx],
                                                beam_hyps.normed_scores[global_batch_idx * (K * 2) + l]);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    const int tgt_id_offset =
                        ((vector_id + beam_hyps.ite * beam_hyps.local_batch_size) * (K * 2) + beam_idx)
                        * (beam_hyps.max_seq_len);
                    beam_hyps.output_ids_tgt[tgt_id_offset + beam_hyps.step] = beam_hyps.end_ids[vector_id];
                    if (beam_hyps.log_probs != nullptr) {
                        beam_hyps.log_probs[tgt_id_offset + beam_hyps.step] =
                            (float)y[total.p[i]] - old_cum_log_probs[(x[total.p[i]] / vocab_size) % K];
                    }

                    int prev_id = (x[total.p[i]] / vocab_size) % K;
                    for (int j = beam_hyps.step - 1; j >= 0; j--) {
                        const int src_idx = j * beam_hyps.batch_size * K
                                            + beam_hyps.ite * beam_hyps.local_batch_size * K + vector_id * K + prev_id;

                        beam_hyps.output_ids_tgt[tgt_id_offset + j] = beam_hyps.output_ids_src[src_idx];
                        if (beam_hyps.log_probs != nullptr && beam_hyps.log_probs_src != nullptr) {
                            beam_hyps.log_probs[tgt_id_offset + j] = beam_hyps.log_probs_src[src_idx];
                        }
                        prev_id = beam_hyps.parent_ids_src[src_idx];
                    }
                    const int tgt_beam_idx                       = global_batch_idx * (K * 2) + beam_idx;
                    beam_hyps.sequence_lengths_tgt[tgt_beam_idx] = beam_hyps.step;
                    beam_hyps.normed_scores[tgt_beam_idx]        = normed_score;
                    beam_hyps.min_normed_scores[global_batch_idx] =
                        min(beam_hyps.min_normed_scores[global_batch_idx], beam_hyps.normed_scores[tgt_beam_idx]);

                    beam_hyps.num_beams[global_batch_idx]++;
                    beam_hyps.cum_log_probs[tgt_beam_idx] = (float)y[total.p[i]];
                }
            }
            else if ((beam_hyps.num_beams != nullptr && i < 2 * K) || (beam_hyps.num_beams == nullptr && i < K)) {
                z[selected_beams] = x[total.p[i]];
                if (output_log_probs != nullptr) {
                    output_log_probs[vector_id * K + selected_beams] =
                        (float)y[total.p[i]] - old_cum_log_probs[(z[selected_beams] / vocab_size) % K];
                }
                v[selected_beams] = (float)y[total.p[i]];
                selected_beams++;
            }
            __syncthreads();
            if (selected_beams >= K) {
                break;
            }
        }
    }
    if (threadIdx.x == 0 && beam_hyps.num_beams != nullptr) {
        if (beam_hyps.num_beams[blockIdx.x] < K) {
            beam_hyps.is_done[blockIdx.x] = false;
        }
        else if (beam_hyps.early_stopping) {
            beam_hyps.is_done[blockIdx.x] = true;
        }
    }
}

struct __align__(8) MD
{
    float m;
    float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger  = (a.m > b.m);
    MD   bigger_m  = a_bigger ? a : b;
    MD   smaller_m = a_bigger ? b : a;
    MD   res;
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

template<typename T, int MAX_K>
struct TopKMD {
    MD             md;
    TopK<T, MAX_K> topk;
};

template<typename T, int MAX_K>
__device__ __forceinline__ TopKMD<T, MAX_K> reduce_topk_md_op(const TopKMD<T, MAX_K>& a, const TopKMD<T, MAX_K>& b)
{
    TopKMD<T, MAX_K> res;
    res.md   = reduce_md_op(a.md, b.md);
    res.topk = reduce_topk_op(a.topk, b.topk);
    return res;
}

template<typename T, int ITEMS_PER_THREAD, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void beam_online_softmax_topk_kernel(const T* __restrict x,
                                                                                    const T* __restrict b,
                                                                                    const float* __restrict c,
                                                                                    const bool* __restrict finished,
                                                                                    int* __restrict z,
                                                                                    T* __restrict v,
                                                                                    int V,
                                                                                    int K,
                                                                                    const int* __restrict end_ids)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    const bool IS_FP16   = std::is_same<T, half>::value;
    const T    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    // reposition y to data for the current vector
    x += vector_id * V;

    typedef cub::BlockReduce<TopKMD<float, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage                     temp_storage;

    TopKMD<float, MAX_K> partial;
    bool                 finish = finished[vector_id];
    for (int i = 0; i < MAX_K; ++i) {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -MAX_T_VAL;
    }
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;

    if (finish) {
        for (int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE) {
            float elem = (elem_id == end_ids[vector_id / K]) ? MAX_T_VAL : -MAX_T_VAL;
            MD    new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
            // if (elem_id > THREADBLOCK_SIZE * MAX_K && (elem_id == E)) break;
        }
    }
    else {
        for (int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE) {
            float elem = x[elem_id] + b[elem_id];
            MD    new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }

    TopKMD<float, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<float, MAX_K>);

    if (thread_id == 0) {
        z += vector_id * K;
        v += vector_id * K;
        c += vector_id;

        // float d_total_inverse = __fdividef(1.0F, total.md.d);
        float d_total_log = logf(total.md.d);
        for (int i = 0; i < MAX_K; ++i) {
            // float val = __expf(total.topk.u[i] - total.md.m) * d_total_inverse;
            float val = total.topk.u[i] - total.md.m - d_total_log;
            if (i < K) {
                z[i] = total.topk.p[i] + vector_id * V;  // faster transformer needs absolute id
                v[i] = val + c[0];
            }
        }
    }
}

template<typename T, int ITEMS_PER_THREAD, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE, 1) __global__
    void beam_online_softmax_topk_stage1_kernel(const T* __restrict x,
                                                const T* __restrict b,
                                                const bool* __restrict finished,
                                                float* __restrict t,
                                                int V,
                                                int K,
                                                const int* __restrict end_ids)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;  // batch beam index.

    const int PACKED_TOP_KMD_SIZE = 2 * MAX_K + 2;

    const bool IS_FP16   = std::is_same<T, half>::value;
    const T    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    // one will have multiple sections per V
    const int v_local       = (V + gridDim.y - 1) / gridDim.y;
    const int section_start = v_local * blockIdx.y;
    int       section_end   = section_start + v_local;
    section_end             = (section_end > V) ? V : section_end;

    // reposition x to data for the current vector
    x += vector_id * V;
#if TOPK_FP16_STORAGE == 1
    typedef cub::BlockReduce<TopKMD<__half, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
#else
    typedef cub::BlockReduce<TopKMD<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
#endif
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float                             buf_s[PACKED_TOP_KMD_SIZE];  // save intermediate result

#if TOPK_FP16_STORAGE == 1
    TopKMD<__half, MAX_K> partial;
#else
    TopKMD<T, MAX_K>                                             partial;
#endif
    bool finish = finished[vector_id];
    for (int i = 0; i < MAX_K; ++i) {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -MAX_T_VAL;
    }
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;

    if (finish) {
#pragma unroll 1
        for (int elem_id = section_start + thread_id; elem_id < section_end; elem_id += THREADBLOCK_SIZE) {
            float elem = (elem_id == end_ids[vector_id / K]) ? MAX_T_VAL : -MAX_T_VAL;
            MD    new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }
    else {
#pragma unroll 1
        for (int elem_id = section_start + thread_id; elem_id < section_end; elem_id += THREADBLOCK_SIZE) {
            T  bias = b == nullptr ? (T)0.0f : b[elem_id];  // gpt-2 does not use bias
            T  elem = x[elem_id] + bias;
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }

#if TOPK_FP16_STORAGE == 1
    TopKMD<__half, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<__half, MAX_K>);
#else
    TopKMD<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<T, MAX_K>);
#endif

    if (thread_id == 0) {
        for (int i = 0; i < 2 * K; i++) {
            reinterpret_cast<int*>(buf_s)[i] = total.topk.p[i] + vector_id * V;  // faster transformer needs absolute id
            buf_s[MAX_K + i]                 = total.topk.u[i];
        }
        buf_s[2 * MAX_K]     = total.md.d;
        buf_s[2 * MAX_K + 1] = total.md.m;
    }
    __syncthreads();
    for (int elem_id = thread_id; elem_id < PACKED_TOP_KMD_SIZE; elem_id += THREADBLOCK_SIZE) {
        t[blockIdx.x * PACKED_TOP_KMD_SIZE * gridDim.y + blockIdx.y * PACKED_TOP_KMD_SIZE + elem_id] = buf_s[elem_id];
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void beam_online_softmax_topk_stage2_kernel(
    const float* __restrict x, const float* __restrict c, int* __restrict z, T* __restrict v, int K, int parts_per_beam)
{
    const int vector_id           = blockIdx.x;
    const int thread_id           = threadIdx.x;
    const int PACKED_TOP_KMD_SIZE = 2 * MAX_K + 2;

    const bool IS_FP16   = std::is_same<T, half>::value;
    const T    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    extern __shared__ char buf_s_[];  // intermediate result
    float*                 buf_s = reinterpret_cast<float*>(buf_s_);
    //__shared__ float buf_s[PACKED_TOP_KMD_SIZE * THREADBLOCK_SIZE]; // intermediate result

    typedef cub::BlockReduce<TopKMD<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage                 temp_storage;

    x += vector_id * PACKED_TOP_KMD_SIZE * parts_per_beam;

    TopKMD<T, MAX_K> partial;
    for (int i = 0; i < MAX_K; ++i) {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -MAX_T_VAL;
    }
    partial.md.m = -MAX_T_VAL;
    partial.md.d = 0.0F;

    // load and unpack into registers through smem
    for (int idx = thread_id; idx < PACKED_TOP_KMD_SIZE * parts_per_beam; idx += THREADBLOCK_SIZE) {
        buf_s[idx] = x[idx];
    }
    __syncthreads();

    if (threadIdx.x < parts_per_beam) {
        float* b_s = buf_s + thread_id * PACKED_TOP_KMD_SIZE;
        for (int i = 0; i < 2 * K; i++) {
            partial.topk.p[i] = reinterpret_cast<int*>(b_s)[i];
            partial.topk.u[i] = b_s[MAX_K + i];
        }
        partial.md.d = b_s[2 * MAX_K];
        partial.md.m = b_s[2 * MAX_K + 1];
    }
    __syncthreads();

    TopKMD<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<T, MAX_K>);

    if (thread_id == 0) {
        z += vector_id * 2 * K;
        v += vector_id * 2 * K;
        c += vector_id;

        float d_total_log = logf(total.md.d);
        for (int i = 0; i < MAX_K; ++i) {
            float val = (float)total.topk.u[i] - total.md.m - d_total_log;
            if (i < 2 * K) {
                z[i] = total.topk.p[i];
                v[i] = (float)val + (float)c[0];
            }
        }
    }
}

template<typename T, int MAX_K>
void beam_online_softmax_topk_stage2_kernelLauncher(const float* temp_storage,
                                                    const float* cum_log_probs,
                                                    int*         ids,
                                                    T*           vals,
                                                    int          batch_size,
                                                    int          beam_width,
                                                    int          parts_per_beam,
                                                    cudaStream_t stream)
{
    // might rewrite beam_online_softmax_topk_stage2_kernel no to depend on constant block size
    // in oreder to reduce compilation time
    int smem_stage2_size = parts_per_beam * (2 * MAX_K + 2) * sizeof(float);

    if (parts_per_beam <= 32) {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K, 32><<<batch_size * beam_width, 32, smem_stage2_size, stream>>>(
            temp_storage, cum_log_probs, ids, vals, beam_width, parts_per_beam);
        return;
    }
    if (parts_per_beam <= 64) {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K, 64><<<batch_size * beam_width, 64, smem_stage2_size, stream>>>(
            temp_storage, cum_log_probs, ids, vals, beam_width, parts_per_beam);
        return;
    }
    if (parts_per_beam <= 128) {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K, 128>
            <<<batch_size * beam_width, 128, smem_stage2_size, stream>>>(
                temp_storage, cum_log_probs, ids, vals, beam_width, parts_per_beam);
        return;
    }
    assert(0);
}

template<typename T, int MAX_K>
void topK_softMax_kernelLauncher(const T*        log_probs,
                                 const T*        bias,
                                 const bool*     finished,
                                 const int*      sequence_lengths,
                                 float*          cum_log_probs,
                                 float*          output_log_probs,
                                 int*            ids,
                                 void*           temp_storage,
                                 const int       temp_storage_size,
                                 BeamHypotheses* beam_hyps,
                                 const int       batch_size,
                                 const int       beam_width,
                                 const int       vocab_size,
                                 const int*      end_ids,
                                 T               diversity_rate,
                                 const float     length_penalty,
                                 cudaStream_t    stream)
{
    const int items_per_thread = 1;
    const int block_sz         = (MAX_K < 16) ? (MAX_K < 8) ? SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE : 128 : 64;
    // const int block_sz = SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE;

    assert(temp_storage_size % 2 == 0);
    assert(temp_storage_size >= 2 * batch_size * beam_width * beam_width * 2);
    // Beam search needs the sequence lengths of beams to apply length penalty.
    assert(length_penalty == 0.0f || sequence_lengths != nullptr);

    const int topk_buf_offset  = ceil(batch_size * beam_width * beam_width * 2 / 4.) * 4;
    int*      topk_tmp_id_buf  = reinterpret_cast<int*>(temp_storage);
    T*        topk_tmp_val_buf = reinterpret_cast<T*>(topk_tmp_id_buf + topk_buf_offset);
    float*    tmp_buffer       = reinterpret_cast<float*>(topk_tmp_val_buf + topk_buf_offset);

#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
    int voc_parts = 4;
    if (batch_size * beam_width < 256) {
        // Volta has 80 SMs, so we aim for three waves
        voc_parts = (240 + batch_size * beam_width - 1) / (batch_size * beam_width);
        voc_parts = std::min(128, voc_parts);  // we implement up to 128
    }
    dim3 grid(batch_size * beam_width, voc_parts);
    cudaFuncSetAttribute(beam_online_softmax_topk_stage1_kernel<T, items_per_thread, 2 * MAX_K, block_sz>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxL1);
    beam_online_softmax_topk_stage1_kernel<T, items_per_thread, 2 * MAX_K, block_sz>
        <<<grid, block_sz, 0, stream>>>(log_probs, bias, finished, tmp_buffer, vocab_size, beam_width, end_ids);
    sync_check_cuda_error();
#endif
    if (beam_width > 1) {
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
        beam_online_softmax_topk_stage2_kernelLauncher<T, 2 * MAX_K>(
            tmp_buffer, cum_log_probs, topk_tmp_id_buf, topk_tmp_val_buf, batch_size, beam_width, voc_parts, stream);
        sync_check_cuda_error();
#else
        beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz>
            <<<batch_size * beam_width, block_sz, 0, stream>>>(log_probs,
                                                               bias,
                                                               cum_log_probs,
                                                               finished,
                                                               topk_tmp_id_buf,
                                                               topk_tmp_val_buf,
                                                               vocab_size,
                                                               beam_width,
                                                               end_ids);
#endif
#if 0
            // wrong result with diversity_rate != 0.f
            batch_topK_kernel<T, MAX_K, 32><<<batch_size, 32, 0, stream>>>
                                (topk_tmp_id_buf, topk_tmp_val_buf, ids, cum_log_probs);
#else
        // We need 2*MAX_K candidates because at most k candidates are finished, and we
        // will not put them into next iteration
        batch_topk_kernel<T, MAX_K * 2, 32><<<batch_size, 32, 0, stream>>>(topk_tmp_id_buf,
                                                                           topk_tmp_val_buf,
                                                                           ids,
                                                                           cum_log_probs,
                                                                           output_log_probs,
                                                                           finished,
                                                                           sequence_lengths,
                                                                           *beam_hyps,
                                                                           beam_width * beam_width * 2,
                                                                           beam_width,
                                                                           vocab_size,
                                                                           length_penalty,
                                                                           diversity_rate);
        sync_check_cuda_error();
#endif
    }
    else {
        FT_CHECK(false);
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
        beam_online_softmax_topk_stage2_kernelLauncher<float, MAX_K>(
            tmp_buffer, cum_log_probs, ids, cum_log_probs, batch_size, beam_width, voc_parts, stream);
#else
        beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz>
            <<<batch_size * beam_width, block_sz, 0, stream>>>(
                log_probs, bias, cum_log_probs, finished, ids, cum_log_probs, vocab_size, beam_width, end_ids);
#endif
    }
}

#define CASE_K(K, MAX_K)                                                                                               \
    case K ... MAX_K:                                                                                                  \
        topK_softMax_kernelLauncher<T, MAX_K>(log_probs,                                                               \
                                              bias,                                                                    \
                                              finished,                                                                \
                                              sequence_lengths,                                                        \
                                              cum_log_probs,                                                           \
                                              output_log_probs,                                                        \
                                              ids,                                                                     \
                                              temp_storage,                                                            \
                                              temp_storage_size,                                                       \
                                              beam_hyps,                                                               \
                                              batch_size,                                                              \
                                              beam_width,                                                              \
                                              vocab_size,                                                              \
                                              end_ids,                                                                 \
                                              diversity_rate,                                                          \
                                              length_penalty,                                                          \
                                              stream);                                                                 \
        break;

template<typename T>
void invokeTopkSoftMax(const T*        log_probs,
                       const T*        bias,
                       const bool*     finished,
                       const int*      sequence_lengths,
                       float*          cum_log_probs,
                       float*          output_log_probs,
                       int*            ids,
                       void*           temp_storage,
                       const int       temp_storage_size,
                       BeamHypotheses* beam_hyps,
                       const int       batch_size,
                       const int       beam_width,
                       const int       vocab_size,
                       const int*      end_ids,
                       const float     diversity_rate,
                       const float     length_penalty,
                       cudaStream_t    stream)
{
    switch (beam_width) {
        CASE_K(1, 4);
        CASE_K(5, 8);
        CASE_K(9, 16);
        CASE_K(17, 32);
        CASE_K(33, 64);
        default:
            throw std::runtime_error(fmtstr("Topk kernel of beam search does not support beam_width=%d", beam_width));
    }
}

#undef CASE_K

template void invokeTopkSoftMax<float>(const float*    log_probs,
                                       const float*    bias,
                                       const bool*     finished,
                                       const int*      sequence_lengths,
                                       float*          cum_log_probs,
                                       float*          output_log_probs,
                                       int*            ids,
                                       void*           tmp_storage,
                                       const int       temp_storage_size,
                                       BeamHypotheses* beam_hyps,
                                       const int       batch_size,
                                       const int       beam_width,
                                       const int       vocab_size,
                                       const int*      end_ids,
                                       const float     diversity_rate,
                                       const float     length_penalty,
                                       cudaStream_t    stream);

template void invokeTopkSoftMax<half>(const half*     log_probs,
                                      const half*     bias,
                                      const bool*     finished,
                                      const int*      sequence_lengths,
                                      float*          cum_log_probs,
                                      float*          output_log_probs,
                                      int*            ids,
                                      void*           tmp_storage,
                                      const int       temp_storage_size,
                                      BeamHypotheses* beam_hyps,
                                      const int       batch_size,
                                      const int       beam_width,
                                      const int       vocab_size,
                                      const int*      end_ids,
                                      const float     diversity_rate,
                                      const float     length_penalty,
                                      cudaStream_t    stream);

}  // end of namespace fastertransformer
