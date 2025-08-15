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

#include <limits>
#include <type_traits>

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
#include "src/turbomind/kernels/unfused_attention_kernels.h"

#include "src/turbomind/utils/cuda_type_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T, int ITEMS_PER_THREAD>
__global__ void __launch_bounds__(1024) softmax_kernel(T*           attn_score,
                                                       const float* qk,
                                                       const T*     attn_mask,
                                                       const T*     sinks,
                                                       const int    batch_size,
                                                       const int    head_num,
                                                       const int    q_length,
                                                       const int    k_length)
{
    // attn_score [batch_size, num_heads, q_length, k_length]
    // qk         [batch_size, num_heads, q_length, k_length]
    // attn_mask  [batch_size,            q_length, k_length]

    const long bi = blockIdx.y;  // Batch index.
    const int  hi = blockIdx.z;  // Head index.

    __shared__ float s_mean, s_max;

    float sink = -std::numeric_limits<float>::infinity();
    if (sinks) {
        sink = sinks[hi];
    }

    // Loop along with Q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x) {

        float data[ITEMS_PER_THREAD];
        long  qk_offset;
        float local_max = -std::numeric_limits<float>::infinity();

        // Loop along with K dimension.
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            if (int ki = blockDim.x * i + threadIdx.x; ki < k_length) {  // Index of K dimension.

                qk_offset = ((bi * head_num + hi) * q_length + qi) * k_length + ki;

                float qk_val  = static_cast<float>(qk[qk_offset]);
                float qk_bias = 0.0f;

                long  mask_offset = (bi * q_length + qi) * k_length + ki;
                float mask_val    = static_cast<float>(ldg(&attn_mask[mask_offset]));

                if (!mask_val) {
                    qk_bias -= std::numeric_limits<float>::infinity();
                }

                data[i]   = qk_val + qk_bias;
                local_max = fmaxf(local_max, data[i]);
            }
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);

        if (threadIdx.x == 0) {
            s_max = fmaxf(max_val, sink);
        }

        __syncthreads();

        float local_sum = 0;

        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            if (blockDim.x * i + threadIdx.x < k_length) {
                data[i] = expf(data[i] - s_max);
                local_sum += data[i];
            }
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);

        if (threadIdx.x == 0) {
            sum_val += expf(sink - s_max);
            s_mean = sum_val;
            s_mean = fdividef(1.f, s_mean);
        }
        __syncthreads();

        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            if (blockDim.x * i + threadIdx.x < k_length) {
                qk_offset = ((bi * head_num + hi) * q_length + qi) * k_length + blockDim.x * i + threadIdx.x;
                attn_score[qk_offset] = (T)(data[i] * s_mean);
            }
        }
    }
}

template<typename T>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T>& param, cudaStream_t stream)
{
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.

    dim3 grid(param.q_length, param.batch_size, param.num_heads);

    auto invoke = [&](auto items_per_thread) {
        const int block = round_up(cdiv(param.k_length, items_per_thread.value), WARP_SIZE);
        FT_CHECK(block <= 1024);
        softmax_kernel<T, items_per_thread.value><<<grid, block, 0, stream>>>(param.attention_score,
                                                                              param.qk,
                                                                              param.attention_mask,
                                                                              param.sinks,
                                                                              param.batch_size,
                                                                              param.num_heads,
                                                                              param.q_length,
                                                                              param.k_length);
    };

    const auto k = param.k_length;

    if (k <= 1024) {
        invoke(std::integral_constant<int, 1>{});
    }
    else if (k <= 2048) {
        invoke(std::integral_constant<int, 2>{});
    }
    else if (k <= 4096) {
        invoke(std::integral_constant<int, 4>{});
    }
    else if (k <= 8192) {
        invoke(std::integral_constant<int, 8>{});
    }
    else if (k <= 16384) {
        invoke(std::integral_constant<int, 16>{});
    }
    else if (k <= 32768) {
        invoke(std::integral_constant<int, 32>{});
    }
    else if (k <= 65536) {
        invoke(std::integral_constant<int, 64>{});
    }
    else if (k <= 131072) {
        invoke(std::integral_constant<int, 128>{});
    }
    else {
        throw std::runtime_error("not impelmented");
    }
}

template void invokeMaskedSoftmax(MaskedSoftmaxParam<half>& param, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeMaskedSoftmax(MaskedSoftmaxParam<nv_bfloat16>& param, cudaStream_t stream);
#endif
#if ENABLE_FP32
template void invokeMaskedSoftmax(MaskedSoftmaxParam<float>& param, cudaStream_t stream);
#endif

// clang-format off
template<typename T> struct packed_type;
template <>          struct packed_type<float>         { using type = float; }; // we don't need to pack float by default
template <>          struct packed_type<half>          { using type = half2; };

#ifdef ENABLE_BF16
template<>
struct packed_type<__nv_bfloat16> {
    using type = __nv_bfloat162;
};
#endif

template<typename T> struct num_elems;
template <>          struct num_elems<float>           { static constexpr int value = 1; };
template <>          struct num_elems<float2>          { static constexpr int value = 2; };
template <>          struct num_elems<float4>          { static constexpr int value = 4; };
template <>          struct num_elems<half>            { static constexpr int value = 1; };
template <>          struct num_elems<half2>           { static constexpr int value = 2; };
#ifdef ENABLE_BF16
template <>          struct num_elems<__nv_bfloat16>   { static constexpr int value = 1; };
template <>          struct num_elems<__nv_bfloat162>  { static constexpr int value = 2; };
#endif

template<typename T, int num> struct packed_as;
template<typename T>          struct packed_as<T, 1>              { using type = T; };
template<>                    struct packed_as<half,  2>          { using type = half2; };
template<>                    struct packed_as<float,  2>         { using type = float2; };
template<>                    struct packed_as<int8_t, 2>         { using type = int16_t; };
template<>                    struct packed_as<int32_t, 2>        { using type = int2; };
template<>                    struct packed_as<half2, 1>          { using type = half; };
#ifdef ENABLE_BF16
template<> struct packed_as<__nv_bfloat16,  2> { using type = __nv_bfloat162; };
template<> struct packed_as<__nv_bfloat162, 1> { using type = __nv_bfloat16;  };
#endif

inline __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __device__ float2 operator*(float2 a, float  b) { return make_float2(a.x * b, a.y * b); }
// clang-format on

template<typename T>
__global__ void transpose_remove_padding(const T*     src,
                                         T*           dst,
                                         const int    batch_size,
                                         const int    seq_len,
                                         const int    head_num,
                                         const int    size_per_head,
                                         const int*   mask_offset,
                                         const float* scale,
                                         const int    int8_mode)
{
    // TODO: optimize this kernel?
    // do remove_sequence_length_padding
    const int bid = blockIdx.x;  // batch * seq_len or valid_word_num

    const int token_offset = mask_offset ? mask_offset[bid] : 0;

    const int src_batch_id = (bid + token_offset) / seq_len;
    const int src_seq_id   = (bid + token_offset) % seq_len;

    const int dst_seq_id = bid;

    const int src_offset_base = src_batch_id * seq_len * head_num * size_per_head + src_seq_id * size_per_head;
    const int dst_offset_base = dst_seq_id * head_num * size_per_head;

    using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
    using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
    const Float_Packed_T scale_val =
        int8_mode == 2 ? cuda_cast<Float_Packed_T>(*scale) : cuda_cast<Float_Packed_T>(0.0f);

    for (int idx = threadIdx.x; idx < head_num * size_per_head; idx += blockDim.x) {
        const int head_id   = idx / size_per_head;
        const int hidden_id = idx % size_per_head;
        const T   src_elem  = ldg(&src[src_offset_base + head_id * seq_len * size_per_head + hidden_id]);
        if (int8_mode == 2) {
            reinterpret_cast<Int8_Packed_T*>(dst)[dst_offset_base + idx] =
                cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(src_elem) * scale_val);
        }
        else {
            dst[dst_offset_base + idx] = src_elem;
        }
    }
}

// clang-format off
template<typename T>
void invokeTransposeAttentionOutRemovePadding(T*           src,
                                              T*           dst,
                                              const int    valid_word_num,
                                              const int    batch_size,
                                              const int    seq_len,
                                              const int    head_num,
                                              const int    size_per_head,
                                              const int*   mask_offset,
                                              const float* scale,
                                              const int    int8_mode,
                                              cudaStream_t stream)
{
#ifdef ENABLE_BF16
    bool is_half2 = (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) && (size_per_head % 2 == 0);
#else
    bool is_half2 = (std::is_same<T, half>::value) && (size_per_head % 2 == 0);
#endif
    using T2       = typename TypeConverter<T>::Type;  // fp16 to half2, bf16 to bf162
    int block_size = head_num * size_per_head;
    if (is_half2) {
        while (block_size > 512) {
            if (block_size % 2 == 0) {
                block_size /= 2;
            }
            else {
                is_half2   = false;
                block_size = std::min(block_size, 1024);
                break;
            }
        }
    }
    else {
        block_size = std::min(block_size, 1024);
    }

    if (is_half2) {
        transpose_remove_padding<T2><<<valid_word_num, block_size, 0, stream>>>(
            (T2*)src, (T2*)dst, batch_size, seq_len, head_num, size_per_head / 2, mask_offset, scale, int8_mode);
    }
    else {
        transpose_remove_padding<<<valid_word_num, block_size, 0, stream>>>(
            src, dst, batch_size, seq_len, head_num, size_per_head, mask_offset, scale, int8_mode);
    }
}
// clang-format on

#define INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(T)                                                               \
    template void invokeTransposeAttentionOutRemovePadding(T*           src,                                           \
                                                           T*           dst,                                           \
                                                           const int    valid_word_num,                                \
                                                           const int    batch_size,                                    \
                                                           const int    seq_len,                                       \
                                                           const int    head_num,                                      \
                                                           const int    size_per_head,                                 \
                                                           const int*   mask_offset,                                   \
                                                           const float* scale,                                         \
                                                           const int    int8_mode,                                     \
                                                           cudaStream_t stream)
#ifdef ENABLE_FP32
INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(float);
#endif
INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(__nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING

template<typename T>
__global__ void addRelativeAttentionBias(
    T* qk_buf, const T* relative_attention_bias, const int batch_size, const int head_num, const int seq_len)
{
    for (int i = threadIdx.x; i < batch_size * seq_len; i += blockDim.x) {
        int batch_id = i / seq_len;
        int seq_id   = i % seq_len;

        const int bias_index = blockIdx.x * seq_len + seq_id;
        const int qk_index   = batch_id * gridDim.x * seq_len + bias_index;
        qk_buf[qk_index]     = add(qk_buf[qk_index], relative_attention_bias[bias_index]);
    }
}

template<typename T>
void invokeAddRelativeAttentionBias(T*           qk_buf,
                                    const T*     relative_attention_bias,
                                    const int    batch_size,
                                    const int    head_num,
                                    const int    seq_len,
                                    cudaStream_t stream)
{
    // qk_buf: [batch_size, head_num, seq_len, seq_len]
    // relative_attention_bias: [1, head_num, seq_len, seq_len]
    dim3 grid(head_num * seq_len);
    dim3 block(512);
    using T2 = typename TypeConverter<T>::Type;
#ifdef ENABLE_BF16
    const bool is_half2 = (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) && (seq_len % 2 == 0);
#else
    const bool is_half2 = (std::is_same<T, half>::value) && (seq_len % 2 == 0);
#endif
    if (is_half2) {
        addRelativeAttentionBias<T2><<<grid, block, 0, stream>>>(
            (T2*)qk_buf, (const T2*)relative_attention_bias, batch_size, head_num, seq_len / 2);
    }
    else {
        addRelativeAttentionBias<<<grid, block, 0, stream>>>(
            qk_buf, relative_attention_bias, batch_size, head_num, seq_len);
    }
}

#define INSTANTIATEADDRELATIVEATTENTIONBIAS(T)                                                                         \
    template void invokeAddRelativeAttentionBias(T*           qk_buf,                                                  \
                                                 const T*     relative_attention_bias,                                 \
                                                 const int    batch_size,                                              \
                                                 const int    head_num,                                                \
                                                 const int    seq_len,                                                 \
                                                 cudaStream_t stream)
#if 0
#ifdef ENABLE_FP32
INSTANTIATEADDRELATIVEATTENTIONBIAS(float);
#endif
INSTANTIATEADDRELATIVEATTENTIONBIAS(half);
#ifdef ENABLE_BF16
INSTANTIATEADDRELATIVEATTENTIONBIAS(__nv_bfloat16);
#endif
#undef INSTANTIATEADDRELATIVEATTENTIONBIAS
#endif

}  // namespace turbomind
