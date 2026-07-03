/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cfloat>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/cross_entropy_kernels.h"
#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T>
__global__ void CrossEntropyLossKernel(float*     ce_loss,
                                       const T*   logits,
                                       int64_t    logits_stride,
                                       const int* target_ids,
                                       int        target_offset,
                                       int        logit_offset,
                                       int        token_num,
                                       int        vocab_size)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= token_num) {
        return;
    }

    const T* row_logits = logits + (logit_offset + row) * logits_stride;

    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        local_max = fmaxf(local_max, static_cast<float>(row_logits[i]));
    }

    const float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);

    __shared__ float s_max;
    if (tid == 0) {
        s_max = max_val;
    }
    __syncthreads();

    float local_sum = 0.f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        local_sum += __expf(static_cast<float>(row_logits[i]) - s_max);
    }

    const float sum_exp = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
    if (tid == 0) {
        const int   target       = target_ids[target_offset + row];
        const float target_logit = static_cast<float>(row_logits[target]);
        const float loss         = __logf(sum_exp + 1e-9f) + s_max - target_logit;
        atomicAdd(ce_loss, loss);
    }
}

void invokeCrossEntropyLoss(float*        ce_loss,
                            const Tensor& logits,
                            const int*    target_ids,
                            int           target_offset,
                            int           logit_offset,
                            int           token_num,
                            int           vocab_size,
                            cudaStream_t  stream)
{
    if (token_num == 0) {
        return;
    }

    const int block_size = vocab_size < 1024 ? (vocab_size + 31) / 32 * 32 : 1024;
    TM_CHECK_EQ(block_size % 32, 0);

    auto dispatch = [&](auto t) {
        using T = decltype(t);
        CrossEntropyLossKernel<T><<<token_num, block_size, 0, stream>>>(ce_loss,
                                                                        logits.data<T>(),
                                                                        logits.stride(0),
                                                                        target_ids,
                                                                        target_offset,
                                                                        logit_offset,
                                                                        token_num,
                                                                        vocab_size);
    };

    TM_DISPATCH_PRIMARY_DTYPES(logits.dtype(), dispatch);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // end of namespace turbomind
