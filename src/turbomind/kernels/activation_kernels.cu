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

// clang-format off
template<template<typename T> class Activation, typename T, typename BT>
__global__ void generic_activation(T*                      out,
                                   const BT*  __restrict   bias,
                                   const T*   __restrict   gated_weights,
                                   const BT*  __restrict   gated_bias,
                                   const int* __restrict   ia3_tasks,
                                   const T*   __restrict   ia3_weights,
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
    // const bool with_ia3  = ia3_tasks != nullptr;

    using Act_T         = typename Activation<T>::return_type;
    using Float_T       = typename packed_as<float, packed_elems>::type;
    using Packed_Int8_t = typename packed_as<int8_t, packed_elems>::type;

    for (int64_t id = blockIdx.x * blockDim.x + threadIdx.x; id < 1LL * m * n; id += blockDim.x * gridDim.x) {
        T val;
        if (int8_mode == 2) {
            // val = cuda_cast<T>(cuda_cast<Float_T>(reinterpret_cast<Packed_Int8_t*>(out)[id]) * activation_in[0]);
        }
        else {
            val = out[id];
        }

        T gated_val;
        if (with_gate) {
            gated_val = gated_weights[id];
        }

        // if (with_bias) {
        //     const T reg_bias = static_cast<T>(bias[id % n]);
        //     val              = val + reg_bias;

        //     if (with_gate) {
        //         const T reg_gated_bias = static_cast<T>(gated_bias[id % n]);
        //         gated_val              = gated_val + reg_gated_bias;
        //     }
        // }

        if (with_gate) {
            val = cuda_cast<T>(Activation<T>::apply(val) * cuda_cast<Act_T>(gated_val));
        }
        else {
            // val = cuda_cast<T>(Activation<T>::apply(val));
        }

        // if (with_ia3) {
        //     const int word_id = id / n;
        //     const int offset = padding_offset == nullptr ? 0 : padding_offset[word_id];
        //     const int batch_id = (word_id + offset) / seq_len;
        //     const int task = ia3_tasks[batch_id];
        //     val            = val * ia3_weights[task * n + (id % n)];
        // }

        if (int8_mode != 2) {
            out[id] = val;
        }
        else {
            // reinterpret_cast<Packed_Int8_t*>(out)[id] =
            //     cuda_cast<Packed_Int8_t>(cuda_cast<Float_T>(val) * activation_out[0]);
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

    const int n_threads = 512;

    dim3 block, grid;
    if (n / 4 / packed_elems <= n_threads) {
        block.x = n / 4 / packed_elems;
        grid.x  = m;
    }
    else {
        block.x = n_threads;
        grid.x  = ceil(1LL * m * n / double(n_threads));
    }
    TM_LOG_DEBUG("%d %d", grid.x, block.x);
    sync_check_cuda_error();
    generic_activation<Activation><<<grid, block, 0, stream>>>(reinterpret_cast<PT*>(out),
                                                               reinterpret_cast<const PBT*>(bias),
                                                               reinterpret_cast<const PT*>(gated_weights),
                                                               reinterpret_cast<const PBT*>(gated_bias),
                                                               ia3_tasks,
                                                               reinterpret_cast<const PT*>(ia3_weights),
                                                               int8_mode,
                                                               activation_in,
                                                               activation_out,
                                                               padding_offset,
                                                               seq_len,
                                                               m,
                                                               n / packed_elems);
    sync_check_cuda_error();
}

#define INSTANTIATE_GENERIC_ACTIVATION(Activation, T, BT)                                                              \
    template void invokeGenericActivation<Activation, T, BT>(T * out,                                                  \
                                                             const BT*    bias,                                        \
                                                             const T*     gated_weights,                               \
                                                             const BT*    gated_bias,                                  \
                                                             const int*   ia3_tasks,                                   \
                                                             const T*     ia3_weights,                                 \
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

// `output` may be an alias of `inter_buf`
template<int VecSize, template<typename T> class Activation, typename T>
__global__ void activation_kernel(T* inter_buf, const T* __restrict__ gate_buf, int64_t stride, int token_num, int dims)
{
    const int di = threadIdx.x + blockIdx.x * blockDim.x;
    const int ti = blockIdx.y;

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

    constexpr int block = 256;
    const dim3    grid(ceil_div(dims, block * kVecSize), token_num);

    activation_kernel<kVecSize, Activation, T>
        <<<grid, block, 0, stream>>>(inter_buf, gate_buf, stride, token_num, dims);
}

#define INSTANTIATE_ACTIVATION(Activation, T)                                                                          \
    template void invokeGenericActivation_v2<SiluActivation>(                                                          \
        T * inter_buf, const T* __restrict__ gate_buf, int64_t stride, int token_num, int dims, cudaStream_t stream)

INSTANTIATE_ACTIVATION(SiluActivation, half);
#ifdef ENABLE_FP32
INSTANTIATE_ACTIVATION(SiluActivation, float);
#endif
#ifdef ENABLE_BF16
INSTANTIATE_ACTIVATION(SiluActivation, __nv_bfloat16);
#endif

}  // namespace turbomind
