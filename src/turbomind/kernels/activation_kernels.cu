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

#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
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

// clang-format off
template<template<typename T> class Activation, typename T, typename BT, typename QT, bool enable_quant>
__global__ void generic_activation(T*                      out,
                                   const BT*  __restrict   bias,
                                   const T*   __restrict   gated_weights,
                                   const BT*  __restrict   gated_bias,
                                   const int* __restrict   ia3_tasks,
                                   const T*   __restrict   ia3_weights,
                                   QT*       __restrict   quant_out,
                                   float*     __restrict   quant_scale,
                                   const int               int8_mode,
                                   const float* __restrict activation_in,
                                   const float* __restrict activation_out,
                                   const int* __restrict padding_offset,
                                   const int seq_len,
                                   int m,
                                   int n)
{
    constexpr size_t packed_elems = num_elems<T>::value;

    const bool with_bias = bias != nullptr;
    const bool with_gate = gated_weights != nullptr;

    using Act_T         = typename Activation<T>::return_type;
    using Single_T      = typename packed_as<T, 1>::type;
    using Packed_Float  = typename packed_as<float, packed_elems>::type;

    if constexpr (enable_quant) {
        __shared__ float s_amax;
        float amax_val = 0.0f;
        for (int64_t id = threadIdx.x; id < n; id += blockDim.x) {
            T val = out[blockIdx.x * n + id];
            T gated_val;
            if (with_gate) {
                gated_val = gated_weights[blockIdx.x * n + id];
                val = cuda_cast<T>(Activation<T>::apply(val) * cuda_cast<Act_T>(gated_val));
            }
            if (int8_mode != 2) {
                out[blockIdx.x * n + id] = val;
            }
            amax_val = cuda_max(amax_val, cuda_cast<float>(cuda_max<Single_T>(cuda_abs(val))));
        }
        amax_val = blockReduceMax<float>(amax_val);
        if (threadIdx.x == 0) {
            s_amax = amax_val;
            quant_scale[blockIdx.x] = amax_val / 127.0f;
        }
        __syncthreads();
        const float tmp_scale = 127.0f / s_amax;
        for (int64_t id = threadIdx.x; id < n; id += blockDim.x) {
            T val = out[blockIdx.x * n + id];
            Packed_Float tmp = cuda_cast<Packed_Float>(val) * tmp_scale;
            quant_out[blockIdx.x * n + id] = cuda_cast<QT>(tmp);
        }
    } else {
        for (int64_t id = threadIdx.x; id < n; id += blockDim.x) {
            T val = out[blockIdx.x * n + id];
            T gated_val;
            if (with_gate) {
                gated_val = gated_weights[blockIdx.x * n + id];
                val = cuda_cast<T>(Activation<T>::apply(val) * cuda_cast<Act_T>(gated_val));
            }
            if (int8_mode != 2) {
                out[blockIdx.x * n + id] = val;
            }
        }
    }
}
// clang-format on

template<template<typename T> class Activation, typename T, typename BT>
void invokeGenericActivation(T*           out,
                             const BT*    bias,
                             const T*     gated_weights,
                             const BT*    gated_bias,
                             const int*   ia3_tasks,
                             const T*     ia3_weights,
                             int8_t*      quant_out,
                             float*       quant_scale,
                             const int    m,
                             const int    n,
                             const int    int8_mode,
                             const float* activation_in,
                             const float* activation_out,
                             const int*   padding_offset,
                             const int    seq_len,
                             cudaStream_t stream)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TM_LOG_DEBUG("invokeGenericActivation %d %d %d", m, n, seq_len);
    using PT                   = typename packed_type<T>::type;
    constexpr int packed_elems = num_elems<PT>::value;
    using PBT                  = typename packed_as<BT, packed_elems>::type;
    using PQT                  = typename packed_as<int8_t, packed_elems>::type;

    if (packed_elems > 1) {
        FT_CHECK(n % packed_elems == 0);
    }

    dim3 grid(m);
    dim3 block(std::min(n, 1024));

    TM_LOG_DEBUG("%d %d", grid.x, block.x);
    sync_check_cuda_error();
    if (quant_out == nullptr) {
        generic_activation<Activation, PT, PBT, PQT, false>
            <<<grid, block, 0, stream>>>(reinterpret_cast<PT*>(out),
                                         reinterpret_cast<const PBT*>(bias),
                                         reinterpret_cast<const PT*>(gated_weights),
                                         reinterpret_cast<const PBT*>(gated_bias),
                                         ia3_tasks,
                                         reinterpret_cast<const PT*>(ia3_weights),
                                         reinterpret_cast<PQT*>(quant_out),
                                         quant_scale,
                                         int8_mode,
                                         activation_in,
                                         activation_out,
                                         padding_offset,
                                         seq_len,
                                         m,
                                         n / packed_elems);
    }
    else {
        generic_activation<Activation, PT, PBT, PQT, true>
            <<<grid, block, 0, stream>>>(reinterpret_cast<PT*>(out),
                                         reinterpret_cast<const PBT*>(bias),
                                         reinterpret_cast<const PT*>(gated_weights),
                                         reinterpret_cast<const PBT*>(gated_bias),
                                         ia3_tasks,
                                         reinterpret_cast<const PT*>(ia3_weights),
                                         reinterpret_cast<PQT*>(quant_out),
                                         quant_scale,
                                         int8_mode,
                                         activation_in,
                                         activation_out,
                                         padding_offset,
                                         seq_len,
                                         m,
                                         n / packed_elems);
    }
    sync_check_cuda_error();
}

#define INSTANTIATE_GENERIC_ACTIVATION(Activation, T, BT)                                                              \
    template void invokeGenericActivation<Activation, T, BT>(T * out,                                                  \
                                                             const BT*    bias,                                        \
                                                             const T*     gated_weights,                               \
                                                             const BT*    gated_bias,                                  \
                                                             const int*   ia3_tasks,                                   \
                                                             const T*     ia3_weights,                                 \
                                                             int8_t*      quant_out,                                   \
                                                             float*       quant_scale,                                 \
                                                             const int    m,                                           \
                                                             const int    n,                                           \
                                                             const int    int8_mode,                                   \
                                                             const float* activation_in,                               \
                                                             const float* activation_out,                              \
                                                             const int*   padding_offset,                              \
                                                             const int    seq_len,                                     \
                                                             cudaStream_t stream);

INSTANTIATE_GENERIC_ACTIVATION(SiluActivation, half, half);
#ifdef ENABLE_FP32
INSTANTIATE_GENERIC_ACTIVATION(SiluActivation, float, float);
#endif
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(SiluActivation, __nv_bfloat16, __nv_bfloat16);
#endif

}  // namespace turbomind
