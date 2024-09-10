/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <stdexcept>
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11000)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/utils/constant.h"

namespace turbomind {

__global__ void curandInitialize(curandState_t* state, const int size, const unsigned long long random_seed)
{
    if (threadIdx.x + blockIdx.x * blockDim.x < size) {
        curand_init(random_seed, 0, 0, &state[blockIdx.x * blockDim.x + threadIdx.x]);
    }
}

void invokeCurandInitialize(curandState_t*           state,
                            const size_t             batch_size,
                            const unsigned long long random_seed,
                            cudaStream_t             stream)
{
    dim3 block(256);
    dim3 grid((int)(ceil(batch_size * 1.0 / 256)));
    curandInitialize<<<grid, block, 0, stream>>>(state, batch_size, random_seed);
}

__global__ void curandBatchInitialize(curandState_t* states, const int size, const unsigned long long* random_seeds)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        curand_init(random_seeds[idx], 0, 0, &states[idx]);
    }
}

void invokeCurandBatchInitialize(curandState_t*            states,
                                 const size_t              batch_size,
                                 const unsigned long long* random_seeds,
                                 cudaStream_t              stream)
{
    dim3 block(256);
    dim3 grid((int)(ceil(batch_size * 1.0 / 256)));
    curandBatchInitialize<<<grid, block, 0, stream>>>(states, batch_size, random_seeds);
}

template<typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topKSortStage1(T*         logits,
                               int*       topk_tmp_id_buf,
                               T*         topk_tmp_val_buf,
                               const int  max_top_k,
                               const int* top_ks,
                               const int  vocab_size,
                               const int  vocab_size_padded)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage    temp_storage;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int block_lane = bid % BLOCKS_PER_BEAM;  // block id for a beam
    const int batch_id   = bid / BLOCKS_PER_BEAM;  // row id for log_probs
    const int k          = top_ks[batch_id];
    if (k == 0) {
        return;
    }

    logits += batch_id * vocab_size_padded;
    topk_tmp_id_buf += batch_id * BLOCKS_PER_BEAM * max_top_k + block_lane * k;
    topk_tmp_val_buf += batch_id * BLOCKS_PER_BEAM * max_top_k + block_lane * k;

    TopK_2<T>  partial;
    const bool IS_FP16   = std::is_same<T, half>::value;
    const T    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    for (int ite = 0; ite < k; ite++) {
        partial.init();
#pragma unroll
        for (int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size;
             elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM) {
            partial.insert(logits[elem_id], elem_id);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0) {
            topk_tmp_id_buf[ite]  = total.p;
            topk_tmp_val_buf[ite] = total.u;
            logits[total.p]       = -MAX_T_VAL;
        }
        __syncthreads();
    }
}

template<typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topKSortStage2(const int* top_ks,
                               const int  max_top_k,
                               const int* topk_tmp_id_buf,
                               T*         topk_tmp_val_buf,
                               const int  vocab_size_padded,
                               T*         sorted_logits,
                               int*       sorted_indices,
                               int*       kept)
{
    const bool IS_FP16   = std::is_same<T, half>::value;
    const T    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    const int tid      = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int k        = top_ks[batch_id];

    if (k == 0) {
        return;
    }

    sorted_indices += batch_id * vocab_size_padded;
    sorted_logits += batch_id * vocab_size_padded;
    const int size   = k * BLOCKS_PER_BEAM;
    const int stride = max_top_k * BLOCKS_PER_BEAM;

    typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage        temp_storage;
    extern __shared__ char                              array[];
    __shared__ float                                    s_sum;
    __shared__ float                                    s_max;
    T*                                                  s_val  = topk_tmp_val_buf + batch_id * stride;
    int*                                                s_id   = reinterpret_cast<int*>(array);
    float*                                              s_val2 = reinterpret_cast<float*>(s_id + k);

    if (tid == 0) {
        kept[batch_id] = min(kept[batch_id], k);
        s_sum          = 0.0f;
    }

    TopK_2<float> partial;
    for (int ite = 0; ite < k; ite++) {
        partial.init();
#pragma unroll
        for (int i = tid; i < size; i += BLOCK_SIZE) {
            partial.insert((float)s_val[i], i);
        }

        TopK_2<float> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<float>);

        if (tid == 0) {
            if (ite == 0) {
                s_max = total.u;
            }
            s_id[ite]      = total.p;
            s_val[total.p] = -MAX_T_VAL;
            total.u        = __expf(total.u - s_max);
            s_val2[ite]    = total.u;
            s_sum += total.u;
        }
        __syncthreads();
    }

    // norm selected
    float thread_sum = s_sum;
    topk_tmp_id_buf += batch_id * stride;
    for (int i = tid; i < k; i += BLOCK_SIZE) {
        sorted_logits[i]  = s_val2[i] / thread_sum;
        sorted_indices[i] = topk_tmp_id_buf[s_id[i]];
    }
}

#define CASE_K(K_MAX, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCKS_PER_BEAM)                                                     \
    topKSortStage1<T, BLOCK_SIZE_1, BLOCKS_PER_BEAM>                                                                   \
        <<<batch_size * BLOCKS_PER_BEAM, BLOCK_SIZE_1, 0, stream>>>((T*)params.logits,                                 \
                                                                    topk_tmp_id_buf,                                   \
                                                                    topk_tmp_val_buf,                                  \
                                                                    max_top_k,                                         \
                                                                    params.top_ks,                                     \
                                                                    params.vocab_size,                                 \
                                                                    params.vocab_size_padded);                         \
    topKSortStage2<T, BLOCK_SIZE_2, BLOCKS_PER_BEAM>                                                                   \
        <<<batch_size, BLOCK_SIZE_2, K_MAX * sizeof(int) + K_MAX * sizeof(float), stream>>>(params.top_ks,             \
                                                                                            params.max_top_k,          \
                                                                                            topk_tmp_id_buf,           \
                                                                                            topk_tmp_val_buf,          \
                                                                                            params.vocab_size_padded,  \
                                                                                            (T*)params.sorted_logits,  \
                                                                                            params.sorted_indices,     \
                                                                                            params.kept);

template<typename T>
void invokeTopKSortFilter(TopKSortFilterParams& params, cudaStream_t stream)
{
    const int max_top_k             = params.max_top_k;
    const int batch_size            = params.batch_size;
    const int max_block_per_beam    = 8;
    int       topk_tmp_ids_buf_size = batch_size * max_top_k * max_block_per_beam;  // type int
    int       topk_tmp_val_buf_size = batch_size * max_top_k * max_block_per_beam;  // type T

    // prevent memory misaligned address
    topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
    topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

    if (params.workspace == nullptr) {
        params.workspace_size = sizeof(int) * topk_tmp_ids_buf_size + sizeof(T) * topk_tmp_val_buf_size;
        return;
    }

    int* topk_tmp_id_buf  = (int*)params.workspace;
    T*   topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);

    if (max_top_k <= 16) {
        CASE_K(16, 128, 128, 8);
    }
    else if (max_top_k <= 32) {
        CASE_K(32, 256, 128, 8);
    }
    else if (max_top_k <= 64) {
        CASE_K(64, 256, 256, 8);
    }
    else if (max_top_k <= 1024) {
        CASE_K(1024, 256, 256, 8);
    }
    else {
        throw std::domain_error(fmtstr("top-k kernel supports 1<=k<=1024 but got k=%d", max_top_k));
    }
}

template void invokeTopKSortFilter<float>(TopKSortFilterParams& params, cudaStream_t stream);

}  // namespace turbomind
