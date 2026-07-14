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

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/kernels/activation_ops.h"
#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/cuda_type_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace turbomind {

// `output` may be an alias of `inter_buf`
template<int VecSize, template<typename T> class Activation, typename T>
__global__ void activation_kernel(T* inter_buf, const T* __restrict__ gate_buf, int64_t stride, int token_num, int dims)
{
    const int di = threadIdx.x + blockIdx.y * blockDim.x;
    const int ti = blockIdx.x;

    dims /= VecSize;

    if (di >= dims) {
        return;
    }

    using Vec = Array<T, VecSize>;

    auto p_inter = reinterpret_cast<Vec*>(inter_buf + ti * stride);
    auto p_gate  = reinterpret_cast<const Vec*>(gate_buf + ti * stride);

    Vec inter;
    Load(inter, (T*)&p_inter[di]);

    Vec gate;
    Ldg(gate, (const T*)&p_gate[di]);

    PRAGMA_UNROLL
    for (int i = 0; i < VecSize; ++i) {
        inter[i] = Activation<T>::apply(inter[i]) * gate[i];
    }

    Store((T*)&p_inter[di], inter);
}

template<template<typename T> class Activation, typename T>
void invokeGenericActivation_v2(
    T* inter_buf, const T* __restrict__ gate_buf, int64_t stride, int token_num, int dims, cudaStream_t stream)
{
    constexpr int kVecSize = 4;

    constexpr int block = 512;
    const dim3    grid(token_num, ceil_div(dims, block * kVecSize));

    activation_kernel<kVecSize, Activation, T>
        <<<grid, block, 0, stream>>>(inter_buf, gate_buf, stride, token_num, dims);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<template<typename T> class Activation>
void invokeGenericActivation_v3(Ref<Tensor> inter_, const Tensor& gate, cudaStream_t stream)
{
    auto& inter = inter_.get();
    TM_CHECK_EQ(inter.ndim(), 2);
    TM_CHECK_EQ(gate.ndim(), 2);
    TM_CHECK_EQ(inter.stride(0), gate.stride(0));

    TM_CHECK(inter.shape() == gate.shape());

    auto invoke = [&](auto t) {
        using T = decltype(t);

        const auto [num, dim] = inter.shapes(0, 1);

        constexpr int kVecSize = 4;
        constexpr int block    = 512;

        const dim3 grid(num, cdiv((int)dim, block * kVecSize));

        activation_kernel<kVecSize, Activation, T>
            <<<grid, block, 0, stream>>>(inter.data<T>(), gate.data<T>(), inter.stride(0), num, dim);
    };

    TM_DISPATCH_PRIMARY_DTYPES(inter.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

template void invokeGenericActivation_v3<SiluActivation>(Ref<Tensor> inter_, const Tensor& gate, cudaStream_t stream);

}  // namespace turbomind
