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

#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/utils/cuda_type_utils.cuh"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace fastertransformer {

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
        return make_bfloat162(val.x > zero_bf16 ? val.x : zero_bf16, val.y > zero_bf16 ? val.y : zero_bf16);
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

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
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
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_LOG_DEBUG("invokeGenericActivation %d %d %d", m, n, seq_len);
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
        grid.x  = ceil(m * n / double(n_threads));
    }
    FT_LOG_DEBUG("%d %d", grid.x, block.x);
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

INSTANTIATE_GENERIC_ACTIVATION(GeluActivation, float, float);
INSTANTIATE_GENERIC_ACTIVATION(GeluActivation, half, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(GeluActivation, __nv_bfloat16, __nv_bfloat16);
#endif

INSTANTIATE_GENERIC_ACTIVATION(ReluActivation, float, float);
INSTANTIATE_GENERIC_ACTIVATION(ReluActivation, half, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(ReluActivation, __nv_bfloat16, __nv_bfloat16);
#endif

INSTANTIATE_GENERIC_ACTIVATION(SiluActivation, float, float);
INSTANTIATE_GENERIC_ACTIVATION(SiluActivation, half, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(SiluActivation, __nv_bfloat16, __nv_bfloat16);
#endif

INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, float, float);
INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, half, half);
INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, float, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, __nv_bfloat16, __nv_bfloat16);
INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, float, __nv_bfloat16);
#endif
#undef INSTANCIATE_GENERIC_ACTIVATION

template<typename T>
__global__ void add_bias_tanh(T* out, const T* __restrict bias, int m, int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        T val = out[id];
        if (bias != nullptr) {
            val = val + ldg(&bias[id % n]);
        }
        out[id] = tanhf(val);
    }
}

template<>
__global__ void add_bias_tanh(half* out, const half* __restrict bias, int m, int n)
{
    half2*       out_ptr  = (half2*)out;
    const half2* bias_ptr = (half2*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 val = out_ptr[id];
        if (bias != nullptr) {
            val = val + __ldg(&bias_ptr[id % n]);
        }
        val.x       = tanhf(val.x);
        val.y       = tanhf(val.y);
        out_ptr[id] = val;
    }
}

#ifdef ENABLE_BF16
template<>
__global__ void add_bias_tanh(__nv_bfloat16* out, const __nv_bfloat16* __restrict bias, int m, int n)
{
    __nv_bfloat162*       out_ptr  = (__nv_bfloat162*)out;
    const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        __nv_bfloat162 val = out_ptr[id];
        if (bias != nullptr) {
            val = bf16hadd2(val, ldg(&bias_ptr[id % n]));
        }
        val.x       = tanhf(val.x);
        val.y       = tanhf(val.y);
        out_ptr[id] = val;
    }
}
#endif

template<typename T>
void invokeAddBiasTanh(T* out, const T* bias, const int m, const int n, cudaStream_t stream)
{
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32, 2 for fp16 and bf16
    dim3      block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x  = m;
    }
    else {
        block.x = 1024;
        grid.x  = ceil(m * n / 1024.);
    }
    add_bias_tanh<T><<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
}

template void invokeAddBiasTanh(float* out, const float* bias, const int m, const int n, cudaStream_t stream);
template void invokeAddBiasTanh(half* out, const half* bias, const int m, const int n, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
invokeAddBiasTanh(__nv_bfloat16* out, const __nv_bfloat16* bias, const int m, const int n, cudaStream_t stream);
#endif

template<typename T2, int N>
__global__ void addBiasGeluV2(T2* out,
                              const T2* __restrict bias,
                              const int* ia3_tasks,
                              const T2*  ia3_weights,
                              const int  size,
                              const int* padding_offset,
                              const int  seq_len)
{
    const bool with_ia3 = ia3_tasks != nullptr;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < size; id += blockDim.x * gridDim.x) {
        T2 val = out[id];
        if (bias != nullptr) {
            T2 reg_bias = ldg(&bias[id % N]);
            val         = hadd2(val, reg_bias);
        }
        val = GeluActivation<T2>::apply(val);
        if (with_ia3) {
            const int word_id  = id / N;
            const int offset   = padding_offset == nullptr ? 0 : padding_offset[word_id];
            const int batch_id = (word_id + offset) / seq_len;
            const int task     = ia3_tasks[batch_id];
            val                = val * ia3_weights[task * N + (id % N)];
        }
        out[id] = val;
    }
}

template<typename T2, int N, int ELEMENT_PER_ROUND>
__global__ void addBiasGeluV3(T2* out,
                              const T2* __restrict bias,
                              const int* ia3_tasks,
                              const T2*  ia3_weights,
                              const int  size,
                              const int* padding_offset,
                              const int  seq_len)
{
    const bool with_ia3 = ia3_tasks != nullptr;
    T2         buffer[ELEMENT_PER_ROUND];
    T2         tmp_bias[ELEMENT_PER_ROUND];
    for (int id = blockIdx.x * blockDim.x * ELEMENT_PER_ROUND + threadIdx.x * ELEMENT_PER_ROUND; id < size;
         id += blockDim.x * gridDim.x * ELEMENT_PER_ROUND) {
#pragma unroll
        for (int i = 0; i < ELEMENT_PER_ROUND; i++) {
            buffer[i] = out[id + i];
            if (bias != nullptr) {
                tmp_bias[i] = ldg(&bias[(id + i) % N]);
            }
        }
#pragma unroll
        for (int i = 0; i < ELEMENT_PER_ROUND; i++) {
            if (bias != nullptr) {
                buffer[i] = hadd2(buffer[i], tmp_bias[i]);
            }
            buffer[i] = GeluActivation<T2>::apply(buffer[i]);
            if (with_ia3) {
                const int word_id  = (id + i) / N;
                const int offset   = padding_offset == nullptr ? 0 : padding_offset[word_id];
                const int batch_id = (word_id + offset) / seq_len;
                const int task     = ia3_tasks[batch_id];
                buffer[i]          = buffer[i] * ia3_weights[task * N + ((id + i) % N)];
            }
            out[id + i] = buffer[i];
        }
    }
}

#define ADD_BIAS_GELU(HALF_N, ELEMENT_PER_ROUND)                                                                       \
    case HALF_N:                                                                                                       \
        if (ELEMENT_PER_ROUND > 1) {                                                                                   \
            grid.x = grid.x / ELEMENT_PER_ROUND;                                                                       \
            addBiasGeluV3<T2, HALF_N, ELEMENT_PER_ROUND><<<grid, block, 0, stream>>>(                                  \
                (T2*)out, (const T2*)bias, ia3_tasks, (T2*)ia3_weights, m * half_n, padding_offset, seq_len);          \
        }                                                                                                              \
        else {                                                                                                         \
            addBiasGeluV2<T2, HALF_N><<<grid, block, 0, stream>>>(                                                     \
                (T2*)out, (const T2*)bias, ia3_tasks, (T2*)ia3_weights, m * half_n, padding_offset, seq_len);          \
        }                                                                                                              \
        break;

template<typename T>
void invokeAddBiasGeluV2(T*           out,
                         const T*     bias,
                         const int*   ia3_tasks,
                         const T*     ia3_weights,
                         const int*   padding_offset,
                         const int    seq_len,
                         const int    m,
                         const int    n,
                         cudaStream_t stream)
{
    if (n % 2 == 0 && sizeof(T) == 2) {
        const int half_n = n / 2;
        dim3      block, grid;
        block.x  = std::min(half_n, 512);
        grid.x   = (m * half_n + (block.x - 1)) / block.x;
        using T2 = typename TypeConverter<T>::Type;

        if (grid.x >= 512) {
            switch (half_n) {
                ADD_BIAS_GELU(256, 1)
                ADD_BIAS_GELU(512, 1)
                ADD_BIAS_GELU(1024, 1)
                ADD_BIAS_GELU(1536, 1)
                ADD_BIAS_GELU(2048, 1)
                ADD_BIAS_GELU(4096, 2)
                ADD_BIAS_GELU(8192, 2)
                ADD_BIAS_GELU(16384, 2)
                ADD_BIAS_GELU(24576, 2)
                ADD_BIAS_GELU(40960, 4)
                default:
                    invokeGenericActivation<GeluActivation>(out,
                                                            bias,
                                                            (T*)nullptr,
                                                            (T*)nullptr,
                                                            ia3_tasks,
                                                            ia3_weights,
                                                            m,
                                                            n,
                                                            0,
                                                            (float*)nullptr,
                                                            (float*)nullptr,
                                                            padding_offset,
                                                            seq_len,
                                                            stream);
                    break;
            }
        }
        else {
            switch (half_n) {
                ADD_BIAS_GELU(256, 1)
                ADD_BIAS_GELU(512, 1)
                ADD_BIAS_GELU(1024, 1)
                ADD_BIAS_GELU(1536, 1)
                ADD_BIAS_GELU(2048, 1)
                ADD_BIAS_GELU(4096, 1)
                ADD_BIAS_GELU(8192, 2)
                ADD_BIAS_GELU(16384, 2)
                ADD_BIAS_GELU(24576, 2)
                ADD_BIAS_GELU(40960, 2)
                default:
                    invokeGenericActivation<GeluActivation>(out,
                                                            bias,
                                                            (T*)nullptr,
                                                            (T*)nullptr,
                                                            ia3_tasks,
                                                            ia3_weights,
                                                            m,
                                                            n,
                                                            0,
                                                            (float*)nullptr,
                                                            (float*)nullptr,
                                                            padding_offset,
                                                            seq_len,
                                                            stream);
                    break;
            }
        }
    }
    else {
        invokeGenericActivation<GeluActivation>(out,
                                                bias,
                                                (T*)nullptr,
                                                (T*)nullptr,
                                                ia3_tasks,
                                                ia3_weights,
                                                m,
                                                n,
                                                0,
                                                (float*)nullptr,
                                                (float*)nullptr,
                                                padding_offset,
                                                seq_len,
                                                stream);
    }
}

#undef ADD_BIAS_GELU

template void invokeAddBiasGeluV2(float*       out,
                                  const float* bias,
                                  const int*   ia3_tasks,
                                  const float* ia3_weights,
                                  const int*   padding_offset,
                                  const int    seq_len,
                                  const int    m,
                                  const int    n,
                                  cudaStream_t stream);
template void invokeAddBiasGeluV2(half*        out,
                                  const half*  bias,
                                  const int*   ia3_tasks,
                                  const half*  ia3_weights,
                                  const int*   padding_offset,
                                  const int    seq_len,
                                  const int    m,
                                  const int    n,
                                  cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeAddBiasGeluV2(__nv_bfloat16*       out,
                                  const __nv_bfloat16* bias,
                                  const int*           ia3_tasks,
                                  const __nv_bfloat16* ia3_weights,
                                  const int*           padding_offset,
                                  const int            seq_len,
                                  const int            m,
                                  const int            n,
                                  cudaStream_t         stream);
#endif  // ENABLE_BF16

template<typename T>
__global__ void sigmoid_kernel(T* data, const int size, const float scale)
{
    const int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        float val   = cuda_cast<float>(data[index]);
        val         = 1.0f / (1.0f + exp(-val)) * scale;
        data[index] = T(val);
    }
}

template<>
__global__ void sigmoid_kernel(half2* data, const int size, const float scale)
{
    const int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size / 2) {
        half2  val        = data[index];
        float2 val_float2 = cuda_cast<float2>(val);
        val_float2.x      = 1.0f / (1.0f + exp(-val_float2.x)) * scale;
        val_float2.y      = 1.0f / (1.0f + exp(-val_float2.y)) * scale;
        data[index]       = cuda_cast<half2>(val_float2);
    }
}

template<typename T>
void invokeSigmoid(T* data, const int size, const float scale, cudaStream_t stream)
{
    if (std::is_same<T, float>::value || (size % 2 != 0)) {
        dim3 block(128);
        dim3 grid((size + 127) / 128);
        sigmoid_kernel<<<grid, block, 0, stream>>>(data, size, scale);
    }
    else {
        dim3 block(128);
        dim3 grid((size + 255) / 256);
        sigmoid_kernel<<<grid, block, 0, stream>>>((half2*)data, size, scale);
    }
}

template void invokeSigmoid(float* data, const int size, const float scale, cudaStream_t stream);
template void invokeSigmoid(half* data, const int size, const float scale, cudaStream_t stream);

}  // namespace fastertransformer
