// Modified from xgrammar python/xgrammar/kernels/apply_token_bitmask_inplace_cuda.cu

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// clang-format off
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/turbomind/core/context.h"
#include "src/turbomind/kernels/apply_token_bitmask_inplace_cuda.h"
// clang-format on

using namespace std;

#ifndef CUDART_INF_FP16
#define CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif

#if __CUDA_ARCH__ >= 800
#ifndef CUDART_INF_BF16
#define CUDART_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif
#endif

constexpr int32_t BITS_PER_BLOCK           = 32;
constexpr int32_t THREADS_PER_THREAD_BLOCK = 256;

template<typename T>
__device__ T NegativeInfinity()
{
    return -INFINITY;
}

template<>
__device__ __half NegativeInfinity<__half>()
{
    return -CUDART_INF_FP16;
}

#if __CUDA_ARCH__ >= 800
template<>
__device__ __nv_bfloat16 NegativeInfinity<__nv_bfloat16>()
{
    return -CUDART_INF_BF16;
}
#endif

template<typename T, typename PackedT>
__device__ PackedT PackedNegativeInfinity()
{
    constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
    T             packed[kAlignment];
#pragma unroll
    for (int i = 0; i < kAlignment; i++) {
        packed[i] = NegativeInfinity<T>();
    }
    return *reinterpret_cast<PackedT*>(packed);
}

template<typename T, typename PackedT, int32_t kBitsPerThread>
__global__ void __launch_bounds__(THREADS_PER_THREAD_BLOCK) LogitsBitmaskKernel(T* __restrict__ logits,
                                                                                const int32_t* __restrict__ bitmask,
                                                                                const int32_t* __restrict__ indices,
                                                                                int32_t vocab_size,
                                                                                int32_t logits_stride,
                                                                                int32_t bitmask_stride)
{
    constexpr int      kAlignment  = sizeof(PackedT) / sizeof(T);
    constexpr uint32_t kPackedMask = (1 << kAlignment) - 1;

    const int batch_idx = (indices == nullptr) ? blockIdx.y : indices[blockIdx.y];

    const int      block_offset      = blockIdx.x * THREADS_PER_THREAD_BLOCK * kBitsPerThread;
    T*             logits_gmem_ptr   = logits + batch_idx * logits_stride + block_offset;
    const int32_t* bitmask_gmem_ptr  = bitmask + batch_idx * bitmask_stride + block_offset / BITS_PER_BLOCK;
    const int      bitmask_inner_idx = threadIdx.x % (BITS_PER_BLOCK / kAlignment);
    T              logits_reg[kAlignment];

#pragma unroll
    for (int offset = threadIdx.x * kAlignment; offset < THREADS_PER_THREAD_BLOCK * kBitsPerThread;
         offset += THREADS_PER_THREAD_BLOCK * kAlignment) {
        if (block_offset + offset >= vocab_size) {
            break;
        }

        const uint32_t bitmask_val =
            (~bitmask_gmem_ptr[offset / BITS_PER_BLOCK] >> (bitmask_inner_idx * kAlignment)) & kPackedMask;

        if (bitmask_val == 0) {
            continue;
        }

        if (bitmask_val == kPackedMask) {
            *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset) = PackedNegativeInfinity<T, PackedT>();
            continue;
        }

        *reinterpret_cast<PackedT*>(logits_reg) = *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset);
#pragma unroll
        for (int i = 0; i < kAlignment; i++) {
            if (((bitmask_val >> i) & 1)) {
                logits_reg[i] = NegativeInfinity<T>();
            }
        }
        *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset) = *reinterpret_cast<PackedT*>(logits_reg);
    }
}

template<typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
constexpr auto CeilDiv(T numerator, T denominator)
{
    return (numerator + denominator - 1) / denominator;
}

template<typename T, typename PackedT>
void ApplyTokenBitmaskInplaceDispatchToBitsPerThread(T* __restrict__ logits,
                                                     const int32_t* __restrict__ bitmask,
                                                     const int32_t* __restrict__ indices,
                                                     int32_t vocab_size,
                                                     int32_t logits_stride,
                                                     int32_t bitmask_stride,
                                                     int32_t num_rows)
{
    constexpr int kAlignment          = sizeof(PackedT) / sizeof(T);
    const int32_t num_blocks_per_row  = CeilDiv(2048 / THREADS_PER_THREAD_BLOCK * 128, num_rows);
    const int32_t num_bits_per_thread = CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * num_blocks_per_row);

    const dim3  block(THREADS_PER_THREAD_BLOCK);
    const auto& stream = turbomind::core::Context::stream();

    if (num_bits_per_thread <= 4 && kAlignment <= 4) {
        const dim3 grid(CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * 4), num_rows);
        LogitsBitmaskKernel<T, PackedT, 4>
            <<<grid, block, 0, stream.handle()>>>(logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride);
    }
    else if (num_bits_per_thread <= 8 && kAlignment <= 8) {
        const dim3 grid(CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * 8), num_rows);
        LogitsBitmaskKernel<T, PackedT, 8>
            <<<grid, block, 0, stream.handle()>>>(logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride);
    }
    else if (num_bits_per_thread <= 16 && kAlignment <= 16) {
        const dim3 grid(CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * 16), num_rows);
        LogitsBitmaskKernel<T, PackedT, 16>
            <<<grid, block, 0, stream.handle()>>>(logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride);
    }
    else {
        const dim3 grid(CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * 32), num_rows);
        LogitsBitmaskKernel<T, PackedT, 32>
            <<<grid, block, 0, stream.handle()>>>(logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride);
    }
}

template<typename T>
void ApplyTokenBitmaskInplaceDispatchToPackedT(T* __restrict__ logits,
                                               const int32_t* __restrict__ bitmask,
                                               const int32_t* __restrict__ indices,
                                               int32_t vocab_size,
                                               int32_t logits_stride,
                                               int32_t bitmask_stride,
                                               int32_t num_rows)
{
    if (logits_stride % (sizeof(float4) / sizeof(T)) == 0) {
        ApplyTokenBitmaskInplaceDispatchToBitsPerThread<T, float4>(
            logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride, num_rows);
    }
    else {
        ApplyTokenBitmaskInplaceDispatchToBitsPerThread<T, T>(
            logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride, num_rows);
    }
}

namespace turbomind {
using namespace turbomind::core;

void ApplyTokenBitmaskInplace(Tensor logits, Tensor bitmask, std::optional<Tensor> indices)
{
    std::pair<int32_t, int32_t> logits_shape =
        logits.ndim() == 2 ?
            std::make_pair(static_cast<int32_t>(logits.shape(0)), static_cast<int32_t>(logits.shape(1))) :
            std::make_pair(1, static_cast<int32_t>(logits.shape(0)));

    std::pair<int32_t, int32_t> bitmask_shape =
        bitmask.ndim() == 2 ?
            std::make_pair(static_cast<int32_t>(bitmask.shape(0)), static_cast<int32_t>(bitmask.shape(1))) :
            std::make_pair(1, static_cast<int32_t>(bitmask.shape(0)));

    int vocab_size = std::min(logits_shape.second, bitmask_shape.second * BITS_PER_BLOCK);

    int32_t  num_rows    = logits_shape.first;
    int32_t* indices_ptr = nullptr;
    if (indices) {
        num_rows    = indices->shape(0);
        indices_ptr = indices->data<int32_t>();
    }
    else {
        TM_CHECK(logits_shape.first == bitmask_shape.first) << "logits and bitmask must have the same batch size.";
    }

    // Currently we use only float logits.
    TM_CHECK(logits.dtype() == kFloat32);
    ApplyTokenBitmaskInplaceDispatchToPackedT(logits.data<float>(),
                                              bitmask.data<int32_t>(),
                                              indices_ptr,
                                              vocab_size,
                                              logits.stride(0),
                                              bitmask.stride(0),
                                              num_rows);
}
}  // namespace turbomind
