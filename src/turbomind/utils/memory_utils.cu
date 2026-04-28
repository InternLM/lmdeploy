/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_format.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<typename T_OUT, typename T_IN>
__global__ void transpose102(T_OUT* dst, T_IN* src, const int dim0, const int dim1, const int dim2)
{
    // src permutation: [0, 1, 2]
    // dst permutation: [1, 0, 2]
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < dim0 * dim1 * dim2; tid += blockDim.x * gridDim.x) {
        int       tmp_idx                                           = tid;
        const int dim_2_idx                                         = tmp_idx % dim2;
        tmp_idx                                                     = (tmp_idx - dim_2_idx) / dim2;
        const int dim_1_idx                                         = tmp_idx % dim1;
        tmp_idx                                                     = (tmp_idx - dim_1_idx) / dim1;
        const int dim_0_idx                                         = tmp_idx % dim0;
        dst[dim_1_idx * dim0 * dim2 + dim_0_idx * dim2 + dim_2_idx] = src[tid];
    }
}

template<typename T>
void invokeInPlaceTranspose102(
    T* data, T* workspace, const int dim0, const int dim1, const int dim2, bool copy, cudaStream_t stream)
{
    // copy data to workspace, and then transpose from workspace to data
    // Note that this kernel is used for pre-processing and not very efficient.
    const size_t count = dim0 * dim1 * dim2;
    if (copy) {
        check_cuda_error(cudaMemcpyAsync(workspace, data, sizeof(T) * count, cudaMemcpyDefault, stream));
    }
    const int block = 512;
    const int grid  = std::min((count + block - 1) / block, (size_t)8192);
    transpose102<<<grid, block, 0, stream>>>(data, workspace, dim0, dim1, dim2);
}

template void invokeInPlaceTranspose102(uint16_t*    data,
                                        uint16_t*    workspace,
                                        const int    dim0,
                                        const int    dim1,
                                        const int    dim2,
                                        bool         copy,
                                        cudaStream_t stream);

// -----------------------------------------------------------------------
// Element-wise dtype cast kernel (fp32 <-> fp16 <-> bf16)
// -----------------------------------------------------------------------

template<typename To, typename Ti>
__global__ void dtype_cast_kernel(To* dst, const Ti* src, size_t n)
{
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = static_cast<To>(src[i]);
    }
}

void invokeDtypeCast(
    void* dst, const void* src, size_t count, DataType dst_dtype, DataType src_dtype, cudaStream_t stream)
{
    const int block = 512;
    const int grid  = std::min((count + block - 1) / block, (size_t)8192);

    using half_t = turbomind::half_t;
    using bf16_t = turbomind::bfloat16_t;

    // fp32 -> fp16
    if (src_dtype == turbomind::kFloat32 && dst_dtype == turbomind::kFloat16) {
        dtype_cast_kernel<<<grid, block, 0, stream>>>((half_t*)dst, (const float*)src, count);
    }
    // fp32 -> bf16
    else if (src_dtype == turbomind::kFloat32 && dst_dtype == turbomind::kBfloat16) {
        dtype_cast_kernel<<<grid, block, 0, stream>>>((bf16_t*)dst, (const float*)src, count);
    }
    // fp16 -> fp32
    else if (src_dtype == turbomind::kFloat16 && dst_dtype == turbomind::kFloat32) {
        dtype_cast_kernel<<<grid, block, 0, stream>>>((float*)dst, (const half_t*)src, count);
    }
    // bf16 -> fp32
    else if (src_dtype == turbomind::kBfloat16 && dst_dtype == turbomind::kFloat32) {
        dtype_cast_kernel<<<grid, block, 0, stream>>>((float*)dst, (const bf16_t*)src, count);
    }
    // fp16 -> bf16
    else if (src_dtype == turbomind::kFloat16 && dst_dtype == turbomind::kBfloat16) {
        dtype_cast_kernel<<<grid, block, 0, stream>>>((bf16_t*)dst, (const half_t*)src, count);
    }
    // bf16 -> fp16
    else if (src_dtype == turbomind::kBfloat16 && dst_dtype == turbomind::kFloat16) {
        dtype_cast_kernel<<<grid, block, 0, stream>>>((half_t*)dst, (const bf16_t*)src, count);
    }
}

// -----------------------------------------------------------------------
// EnsureFloatDtype — cast tensor to target dtype if both are trivial float
// -----------------------------------------------------------------------

void EnsureFloatDtype(core::Tensor& tensor, DataType target_dtype)
{
    if (!tensor || tensor.dtype() == target_dtype) {
        return;
    }
    if (!IsTrivialFloatType(tensor.dtype()) || !IsTrivialFloatType(target_dtype)) {
        return;
    }
    auto         stream = core::Context::stream().handle();
    core::Tensor casted{tensor.shape(), target_dtype, tensor.device()};
    invokeDtypeCast(casted.raw_data(), tensor.raw_data(), tensor.size(), target_dtype, tensor.dtype(), stream);
    tensor = std::move(casted);
}

}  // namespace turbomind
