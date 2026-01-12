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

#include <assert.h>
#include <float.h>

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
#include "src/turbomind/kernels/sampling_penalty_kernels.h"

namespace turbomind {

template<typename T, int vec_size>
__global__ void batchApplyTemperaturePenalty_v2(T*           logits,
                                                const T*     bias,
                                                const float* temperatures,
                                                const int    batch_size,
                                                const int    vocab_size,
                                                const int    vocab_size_padded)
{
    const int vi = blockIdx.x * blockDim.x + threadIdx.x;
    const int bi = blockIdx.y;

    __shared__ float shared_scale;

    if (threadIdx.x == 0) {
        shared_scale = fdividef(1.f, temperatures[bi] + 1e-6f);
    }

    __syncthreads();

    const float scale = shared_scale;

    logits += (size_t)bi * vocab_size_padded;

    const int step = gridDim.x * blockDim.x * vec_size;

    for (int i = vi * vec_size; i < vocab_size_padded; i += step) {
        Array<T, vec_size> vec;
        // load
        if constexpr (sizeof(vec) >= sizeof(uint)) {
            Load(vec, logits + i);
        }
        else {
            PRAGMA_UNROLL
            for (int j = 0; j < vec_size; ++j) {
                vec[j] = logits[i + j];
            }
        }

        // process
        PRAGMA_UNROLL
        for (int c = 0; c < vec_size; ++c) {
            if (i + c < vocab_size) {
                vec[c] = (float)vec[c] * scale;
            }
            else {
                vec[c] = -getInfValue<T>();
            }
        }

        // store
        if constexpr (sizeof(vec) >= sizeof(uint)) {
            Store(logits + i, vec);
        }
        else {
            PRAGMA_UNROLL
            for (int j = 0; j < vec_size; ++j) {
                logits[i + j] = vec[j];
            }
        }
    }
}

template<typename T>
void invokeBatchApplyTemperaturePenalty_v2(T*           logits,
                                           const T*     bias,
                                           const float* temperatures,
                                           const int    batch_size,
                                           const int    vocab_size,
                                           const int    vocab_size_padded,
                                           cudaStream_t stream)
{

    auto invoke = [&](auto vec_size) {
        constexpr int threads        = 256;
        const int     blocks_per_tok = (vocab_size_padded + threads * vec_size - 1) / (threads * vec_size);
        const dim3    blocks(blocks_per_tok, batch_size);
        batchApplyTemperaturePenalty_v2<T, vec_size.value><<<blocks, threads, 0, stream>>>(  //
            logits,
            bias,
            temperatures,
            batch_size,
            vocab_size,
            vocab_size_padded);
    };

    if (vocab_size_padded % 4 == 0) {
        invoke(std::integral_constant<int, 4>{});
    }
    else if (vocab_size_padded % 2 == 0) {
        invoke(std::integral_constant<int, 2>{});
    }
    else {
        invoke(std::integral_constant<int, 1>{});
    }
}

#define INSTANTIATE_INVOKE_BATCH_APPLY_TEMPERATURE_PENALTY_V2(T)                                                       \
    template void invokeBatchApplyTemperaturePenalty_v2(T*           logits,                                           \
                                                        const T*     bias,                                             \
                                                        const float* temperatures,                                     \
                                                        const int    batch_size,                                       \
                                                        const int    vocab_size,                                       \
                                                        const int    vocab_size_padded,                                \
                                                        cudaStream_t stream);

INSTANTIATE_INVOKE_BATCH_APPLY_TEMPERATURE_PENALTY_V2(float);

template<class T>
__global__ void RepetitionPenaltyKernel(T*                logits,
                                        const float*      penalties,
                                        const int* const* token_ids_ptrs,
                                        const int*        sequence_length,
                                        int               vocab_size,
                                        int               mask_size)
{
    const int bi = blockIdx.x;

    const int  seq_len   = sequence_length[bi];
    const int* token_ids = token_ids_ptrs[bi];

    extern __shared__ uint32_t masks[];  // up to 512k vocab size on 64k smem devices

    for (int i = threadIdx.x; i < mask_size; i += blockDim.x) {
        masks[i] = 0;
    }

    __syncthreads();

    for (int ti = threadIdx.x; ti < seq_len; ti += blockDim.x) {
        const int token_id = token_ids[ti];
        atomicOr(&masks[token_id / 32], 1U << (token_id % 32));
    }

    __syncthreads();

    logits += bi * (int64_t)vocab_size;

    const float penalty = penalties[bi];

    for (int di = threadIdx.x; di < vocab_size; di += blockDim.x) {
        if (masks[di / 32] & (1U << (di % 32))) {
            const float logit = logits[di];
            logits[di]        = logit < 0.f ? logit * penalty : logit / penalty;
        }
    }
}

void ApplyRepetitionPenalty(Tensor&               logits,
                            const Buffer_<float>& penalties,
                            const Buffer_<int*>&  token_ids_ptrs,
                            const Buffer_<int>&   sequence_length,
                            cudaStream_t          stream)
{
    TM_CHECK_EQ(logits.ndim(), 2);
    auto invoke = [&](auto dtype) {
        using T                      = decltype(dtype);
        const auto [bsz, vocab_size] = logits.shapes(0, 1);
        const int mask_size          = cdiv((int)vocab_size, 32);
        const int smem_size          = sizeof(uint32_t) * mask_size;
        auto      func               = RepetitionPenaltyKernel<T>;
        if (smem_size > (48 << 10)) {
            TM_CHECK_EQ(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size), 0);
        }
        TM_LOG_ERROR("smem_size = %d", smem_size);
        func<<<bsz, 1024, smem_size, stream>>>(
            logits.data<T>(), penalties.data(), token_ids_ptrs.data(), sequence_length.data(), vocab_size, mask_size);
    };
    invoke(float{});
}

template<typename T>
__global__ void batchApplyMinLengthPenalty(T* __restrict__ logits,
                                           const int* __restrict__ min_lengths,
                                           const int* __restrict__ sequence_lengths,
                                           const int vocab_size_padded,
                                           const int batch_size,
                                           const int* __restrict__ end_ids,
                                           const int end_ids_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int bid = tid / end_ids_size;
    int eid = tid % end_ids_size;
    if (bid < batch_size) {
        int end_id = end_ids[bid * end_ids_size + eid];
        if (end_id > 0 && sequence_lengths[bid] + 1 < min_lengths[bid]) {
            T mask_val                               = -getMaxValue<T>();
            logits[bid * vocab_size_padded + end_id] = mask_val;
        }
    }
}

template<typename T>
void invokeMinLengthPenalty(T*           logits,
                            const int*   min_lengths,
                            const int*   sequnece_lengths,
                            const int    vocab_size_padded,
                            const int    batch_size,
                            const int*   end_ids,
                            const int    end_ids_size,
                            cudaStream_t stream)
{
    const dim3 block(std::min(batch_size * end_ids_size, 1024));
    const dim3 grid((batch_size * end_ids_size + block.x - 1) / block.x);
    batchApplyMinLengthPenalty<<<block, grid, 0, stream>>>(
        logits, min_lengths, sequnece_lengths, vocab_size_padded, batch_size, end_ids, end_ids_size);
}

#define INSTANTIATE_INVOKE_MIN_LENGTH_PENALTY(T)                                                                       \
    template void invokeMinLengthPenalty(T*           logits,                                                          \
                                         const int*   min_lengths,                                                     \
                                         const int*   sequnece_lengths,                                                \
                                         const int    vocab_size_padded,                                               \
                                         const int    batch_size,                                                      \
                                         const int*   end_ids,                                                         \
                                         const int    end_ids_size,                                                    \
                                         cudaStream_t stream);

INSTANTIATE_INVOKE_MIN_LENGTH_PENALTY(float);

}  // namespace turbomind
