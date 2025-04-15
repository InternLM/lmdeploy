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

/* Gelu Activation */

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__inline__ __device__ float tanh_opt(float x)
{
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
#else
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

template<typename T>
struct GeluActivation {
    using return_type = T;
    static __device__ __forceinline__ T apply(const T& val)
    {
        const float cdf = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (val + 0.044715f * val * val * val))));
        return val * cdf;
    }
};

template<>
struct GeluActivation<half2> {
    using return_type = half2;
    static __device__ __forceinline__ half2 apply(const half2& val)
    {
        half2  val_pow3 = __hmul2(val, __hmul2(val, val));
        float2 tmp_pow  = __half22float2(val_pow3);
        float2 tmp      = __half22float2(val);

        tmp.x = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
        tmp.y = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
        return __hmul2(val, __float22half2_rn(tmp));
    }
};

#ifdef ENABLE_BF16
template<>
struct GeluActivation<__nv_bfloat162> {
    using return_type = __nv_bfloat162;
    static __device__ __forceinline__ __nv_bfloat162 apply(const __nv_bfloat162& val)
    {
        __nv_bfloat162 val_pow3 = bf16hmul2(val, bf16hmul2(val, val));
        float2         tmp_pow  = bf1622float2(val_pow3);
        float2         tmp      = bf1622float2(val);

        tmp.x = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
        tmp.y = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
        return bf16hmul2(val, __floats2bfloat162_rn(tmp.x, tmp.y));
    }
};
#endif

/* Relu Activation */

template<typename T>
struct ReluActivation {
    using return_type = T;
    static __device__ __forceinline__ T apply(const T& val)
    {
        return val > static_cast<T>(0.0f) ? val : static_cast<T>(0.0f);
    }
};

template<>
struct ReluActivation<half2> {
    using return_type = half2;
    static __device__ __forceinline__ half2 apply(const half2& val)
    {
        const half zero_half = static_cast<half>(0.0f);
        return make_half2(val.x > zero_half ? val.x : zero_half, val.y > zero_half ? val.y : zero_half);
    }
};

#ifdef ENABLE_BF16
template<>
struct ReluActivation<__nv_bfloat162> {
    using return_type = __nv_bfloat162;
    static __device__ __forceinline__ __nv_bfloat162 apply(const __nv_bfloat162& val)
    {
        const __nv_bfloat16 zero_bf16 = static_cast<__nv_bfloat16>(0.0f);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
        return turbomind::make_bfloat162(val.x > zero_bf16 ? val.x : zero_bf16, val.y > zero_bf16 ? val.y : zero_bf16);
#else
        return make_bfloat162(val.x > zero_bf16 ? val.x : zero_bf16, val.y > zero_bf16 ? val.y : zero_bf16);
#endif
    }
};
#endif

/* Silu Activation */

template<typename T>
struct SiluActivation {
    using return_type = T;
    static __device__ __forceinline__ T apply(const T& val)
    {
        return (T)((float)val / (1.0f + __expf((float)-val)));
    }
};

template<>
struct SiluActivation<half2> {
    using return_type = float2;
    static __device__ __forceinline__ float2 apply(const half2& val)
    {
        return make_float2(SiluActivation<float>::apply(val.x), SiluActivation<float>::apply(val.y));
    }
};

#ifdef ENABLE_BF16
template<>
struct SiluActivation<__nv_bfloat162> {
    using return_type = float2;
    static __device__ __forceinline__ float2 apply(const __nv_bfloat162& val)
    {
        return make_float2(SiluActivation<float>::apply(val.x), SiluActivation<float>::apply(val.y));
    }
};
#endif  // ENABLE_BF16

/* Identity Activation (= no activation) */

template<typename T>
struct IdentityActivation {
    using return_type = T;
    static __device__ __forceinline__ T apply(const T& val)
    {
        return val;
    }
};

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
}

template void invokeGenericActivation_v3<SiluActivation>(Ref<Tensor> inter_, const Tensor& gate, cudaStream_t stream);

}  // namespace turbomind
