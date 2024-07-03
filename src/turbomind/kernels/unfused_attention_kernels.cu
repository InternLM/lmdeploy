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

#include "src/turbomind/kernels/attention/rotary_embedding.h"
#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
#include "src/turbomind/kernels/unfused_attention_kernels.h"
#include "src/turbomind/utils/cuda_type_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

__inline__ __device__ int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
    return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

template<typename T, typename T_IN, int ITEMS_PER_THREAD>
__global__ void softmax_kernel(T*          attn_score,
                               const T_IN* qk,
                               const T*    attn_mask,
                               const T*    linear_bias_slopes,
                               const int   batch_size,
                               const int   head_num,
                               const int   q_length,
                               const int   k_length,
                               const float qk_scale)
{
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    const long bi = blockIdx.y;  // Batch index.
    const int  hi = blockIdx.z;  // Head index.

    __shared__ float s_mean, s_max;

    const float linear_bias_slope = linear_bias_slopes != nullptr ? (float)linear_bias_slopes[hi] : 0.0f;

    // Loop along with Q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x) {

        float data[ITEMS_PER_THREAD];
        long  qk_offset;
        float local_max = -1e20f;

        // Loop along with K dimension.
        for (int i = 0; blockDim.x * i + threadIdx.x < k_length; i++) {
            int ki    = blockDim.x * i + threadIdx.x;  // Index of K dimension.
            qk_offset = ((bi * head_num + hi) * q_length + qi) * k_length + ki;

            float qk_val  = static_cast<float>(qk[qk_offset]);
            float qk_bias = 0.0f;

            if (linear_bias_slopes != nullptr) {
                // We don't handle the upper diagonal (ki > qi) separately, whose values
                // are negligible due to the negative infinity mask. And it matches with
                // the HF's implementation.
                qk_bias += static_cast<float>(linear_bias_slope * (ki - qi));
            }

            long  mask_offset = (bi * q_length + qi) * k_length + ki;
            float mask_val    = static_cast<float>(ldg(&attn_mask[mask_offset]));
            qk_bias += (1.0f - mask_val) * -10000.0f;

            data[i]   = qk_scale * qk_val + qk_bias;
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < k_length; i++) {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < k_length; i++) {
            qk_offset             = ((bi * head_num + hi) * q_length + qi) * k_length + blockDim.x * i + threadIdx.x;
            attn_score[qk_offset] = (T)(data[i] * s_mean);
        }
    }
}

template<typename T, int ITEMS_PER_THREAD>
__global__ void softmax_kernel_h2(T*        attn_score,
                                  const T*  qk_buf,
                                  const T*  attn_mask,
                                  const T*  linear_bias_slopes,
                                  const int batch_size,
                                  const int head_num,
                                  const int q_length,
                                  const int k_length,
                                  const T   qk_scale)
{
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    using T2 = typename TypeConverter<T>::Type;

    T2*       attn_score_h2 = reinterpret_cast<T2*>(attn_score);
    const T2* qk_buf_h2     = reinterpret_cast<const T2*>(qk_buf);
    const T2* attn_mask_h2  = reinterpret_cast<const T2*>(attn_mask);

    const long bi = blockIdx.y;  // Batch index
    const int  hi = blockIdx.z;  // Head index.

    __shared__ float s_mean, s_max;

    // Constant values that will be used repeately in the q/k loop.
    const T2 ONE       = cuda_cast<T2>(1.0f);
    const T2 ZERO      = cuda_cast<T2>(0.0f);
    const T2 NEG_INFTY = cuda_cast<T2>(-10000.0f);

    // The normalization factor of QK.
    const T2 qk_scale_h2 = cuda_cast<T2>(qk_scale);
    // The slope of a linear position bias of the current attention head.
    const T2 linear_bias_slope = linear_bias_slopes != nullptr ? cuda_cast<T2>(linear_bias_slopes[hi]) : ZERO;

    // Loop over q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x) {
        T2    data[ITEMS_PER_THREAD];
        long  qk_offset;
        float local_max = -1e20f;

        // Loop over k dimension.
        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            // The half of the index of k dimension. We will use the elements at {2 * ki, 2 * ki + 1}.
            int ki           = blockDim.x * i + threadIdx.x;
            qk_offset        = ((bi * head_num + hi) * q_length + qi) * (k_length / 2) + ki;
            long mask_offset = (bi * q_length + qi) * (k_length / 2) + ki;

            // The value of QK^T matrix at (qi, ki).
            T2 qk = qk_buf_h2[qk_offset];
            // The bias value to the position (qi, ki) including both mask and positional bias.
            T2 qk_bias = ZERO;

            if (linear_bias_slopes != nullptr) {
                // The position bias depends on the distance between qi/ki and is zero if qi >= 2*ki
                // or qi >= 2*ki+1. For T2 vectorization, we should handle every two elements along
                // with k-dim simultaneously. To do this, we check qi / 2 > ki at ones instead of
                // qi >= 2*ki or 2*ki+1. It works because an diagonal element for an odd qi will be
                // zero due to slope * (qi - 2*ki+1) = 0. Thus, we don't handle the upper diagonal
                // separately, whose values are negligible due to the negative infinity mask.
                T2 dist(2.0f * ki - qi, 2.0f * ki + 1 - qi);
                qk_bias = hadd2<T2>(qk_bias, hmul2<T2>(linear_bias_slope, dist));
            }

            T2 mask_val = ldg(&attn_mask_h2[mask_offset]);
            qk_bias     = hadd2<T2>(qk_bias, hmul2<T2>(hsub2<T2>(ONE, mask_val), NEG_INFTY));

            data[i]   = hadd2<T2>(hmul2<T2>(qk, qk_scale_h2), qk_bias);
            local_max = fmax(local_max, fmax((float)data[i].x, (float)data[i].y));
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            data[i] = hexp2<T2>(hsub2<T2>(data[i], cuda_cast<T2>(s_max)));
            local_sum += (float)(data[i].x + data[i].y);
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);

        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((bi * head_num + hi) * q_length + qi) * (k_length / 2) + blockDim.x * i + threadIdx.x;
            attn_score_h2[qk_offset] = hmul2<T2>(data[i], cuda_cast<T2>(s_mean));
        }
    }
}

template<typename T, int K_ITEMS_PER_THREAD, int Q_ITEMS_PER_THREAD>
__global__ void softmax_kernel_h2_v2(T*        attn_score,
                                     const T*  qk_buf,
                                     const T*  attn_mask,
                                     const T*  linear_bias_slopes,
                                     const int batch_size,
                                     const int head_num,
                                     const int q_length,
                                     const int k_length,
                                     const T   scalar)
{
    // attn_score, [batch_size, num_heads, q_length, k_length]
    // qk, [batch_size, num_heads, q_length, k_length]
    // attn_mask, [batch_size, q_length, k_length]
    // linear_bias_slopes, [num_heads]

    using T2 = typename TypeConverter<T>::Type;

    // QK^T matrix of shape (batch_size, head_num, q_length, k_length / 2)
    T2*       attn_score_h2 = reinterpret_cast<T2*>(attn_score);
    const T2* qk_buf_h2     = reinterpret_cast<const T2*>(qk_buf);
    const T2* attn_mask_h2  = reinterpret_cast<const T2*>(attn_mask);

    const long bi = blockIdx.y;  // Batch index
    const int  hi = blockIdx.z;  // Head index.

    // Constant values that will be used repeately in the q/k loop.
    const T2 ONE       = cuda_cast<T2>(1.0f);
    const T2 ZERO      = cuda_cast<T2>(0.0f);
    const T2 NEG_INFTY = cuda_cast<T2>(-10000.0f);

    // The normalization factor of QK.
    const T2 qk_scale = cuda_cast<T2>(scalar);
    // The slope of a linear position bias of the current attention head.
    const T2 linear_bias_slope = linear_bias_slopes != nullptr ? cuda_cast<T2>(linear_bias_slopes[hi]) : ZERO;

    __shared__ float s_sum[Q_ITEMS_PER_THREAD], s_max[Q_ITEMS_PER_THREAD];

    // Loop over q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x * Q_ITEMS_PER_THREAD) {
        T2 data[Q_ITEMS_PER_THREAD][K_ITEMS_PER_THREAD];

        long qk_offset[Q_ITEMS_PER_THREAD];

        float local_max[Q_ITEMS_PER_THREAD];
#pragma unroll
        for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
            local_max[j] = -1e20f;
        }

        // Loop over k dimension.
        const int Q_ITEMS = min((q_length - qi + gridDim.x - 1) / gridDim.x, Q_ITEMS_PER_THREAD);
        for (int i = 0; blockDim.x * i + threadIdx.x < k_length / 2 && i < K_ITEMS_PER_THREAD; ++i) {
            // The half of the index of k dimension. We will use the elements at {2 * ki, 2 * ki + 1}.
            int ki = blockDim.x * i + threadIdx.x;

            long mask_offset[Q_ITEMS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                qk_offset[j]   = ((bi * head_num + hi) * q_length + qi + j * gridDim.x) * (k_length / 2) + ki;
                mask_offset[j] = (bi * q_length + qi + j * gridDim.x) * (k_length / 2) + ki;
            }

            T2 mask_val[Q_ITEMS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                mask_val[j] = ldg(&attn_mask_h2[mask_offset[j]]);
            }

            T2 qk[Q_ITEMS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                qk[j] = qk_buf_h2[qk_offset[j]];
            }

            T2 pos_bias[Q_ITEMS_PER_THREAD];
            if (linear_bias_slopes != nullptr) {
#pragma unroll
                for (int j = 0; j < Q_ITEMS; j++) {
                    // The position bias depends on the distance between qi/ki and is zero if qi >= 2*ki
                    // or qi >= 2*ki+1. For T2 vectorization, we should handle every two elements along
                    // with k-dim simultaneously. To do this, we check qi / 2 > ki at ones instead of
                    // qi >= 2*ki or 2*ki+1. It works because an diagonal element for an odd qi will be
                    // zero due to slope * (qi - 2*ki+1) = 0. Thus, we don't handle the upper diagonal
                    // separately, whose values are negligible due to the negative infinity mask.
                    int qidx = qi + j * gridDim.x;
                    T2  dist(2.0f * ki - qidx, 2.0f * ki + 1 - qidx);
                    pos_bias[j] = hmul2<T2>(linear_bias_slope, dist);
                }
            }
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                mask_val[j] = hmul2<T2>(hsub2<T2>(ONE, mask_val[j]), NEG_INFTY);
            }

#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                T2 val = hadd2<T2>(hmul2<T2>(qk_scale, qk[j]), mask_val[j]);
                if (linear_bias_slopes != nullptr) {
                    val = hadd2<T2>(val, pos_bias[j]);
                }
                data[j][i]   = val;
                local_max[j] = fmax(local_max[j], fmax((float)data[j][i].x, (float)data[j][i].y));
            }
        }

        if (blockDim.x <= 32) {
            warpReduceMaxV2<float, Q_ITEMS_PER_THREAD>(local_max);
        }
        else {
            blockReduceMaxV2<float, Q_ITEMS_PER_THREAD>(local_max);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
                s_max[j] = local_max[j];
            }
        }
        __syncthreads();

        float local_sum[Q_ITEMS_PER_THREAD];
#pragma unroll
        for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
            local_sum[j] = {0.f};
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < k_length / 2 && i < K_ITEMS_PER_THREAD; ++i) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS; ++j) {
                data[j][i] = hexp2<T2>(hsub2<T2>(data[j][i], cuda_cast<T2>(s_max[j])));
            }

#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                local_sum[j] += (float)(data[j][i].x + data[j][i].y);
            }
        }

        if (blockDim.x <= 32) {
            warpReduceSumV2<float, Q_ITEMS_PER_THREAD>(local_sum);
        }
        else {
            blockReduceSumV2<float, Q_ITEMS_PER_THREAD>(local_sum);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS_PER_THREAD; j++) {
                s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
            }
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < k_length / 2 && i < K_ITEMS_PER_THREAD; ++i) {
#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                qk_offset[j] = ((bi * head_num + hi) * q_length + qi + j * gridDim.x) * (k_length / 2) + blockDim.x * i
                               + threadIdx.x;
            }

#pragma unroll
            for (int j = 0; j < Q_ITEMS; j++) {
                attn_score_h2[qk_offset[j]] = hmul2<T2>(data[j][i], cuda_cast<T2>(s_sum[j]));
            }
        }
    }
}

#define LAUNCH_MAKSED_SOFTMAX_(T_, ITEMS_PER_THREAD)                                                                   \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    block.x = (block.x + 31) / 32 * 32;                                                                                \
    assert(block.x <= 1024);                                                                                           \
    if (is_half2) {                                                                                                    \
        if (grid.x % 4 == 0) {                                                                                         \
            grid.x /= 4;                                                                                               \
            softmax_kernel_h2_v2<T_, ITEMS_PER_THREAD, 4>                                                              \
                <<<grid, block, 0, stream>>>((T_*)param.attention_score,                                               \
                                             (const T_*)param.qk,                                                      \
                                             (const T_*)param.attention_mask,                                          \
                                             (const T_*)param.linear_bias_slopes,                                      \
                                             param.batch_size,                                                         \
                                             param.num_heads,                                                          \
                                             param.q_length,                                                           \
                                             param.k_length,                                                           \
                                             (const T_)param.qk_scale);                                                \
        }                                                                                                              \
        else {                                                                                                         \
            softmax_kernel_h2<T_, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>((T_*)param.attention_score,            \
                                                                                (const T_*)param.qk,                   \
                                                                                (const T_*)param.attention_mask,       \
                                                                                (const T_*)param.linear_bias_slopes,   \
                                                                                param.batch_size,                      \
                                                                                param.num_heads,                       \
                                                                                param.q_length,                        \
                                                                                param.k_length,                        \
                                                                                (const T_)param.qk_scale);             \
        }                                                                                                              \
    }                                                                                                                  \
    else {                                                                                                             \
        softmax_kernel<T, T_IN, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>(param.attention_score,                   \
                                                                              param.qk,                                \
                                                                              param.attention_mask,                    \
                                                                              param.linear_bias_slopes,                \
                                                                              param.batch_size,                        \
                                                                              param.num_heads,                         \
                                                                              param.q_length,                          \
                                                                              param.k_length,                          \
                                                                              param.qk_scale);                         \
    }

#define LAUNCH_MAKSED_SOFTMAX(ITEMS_PER_THREAD) LAUNCH_MAKSED_SOFTMAX_(half, ITEMS_PER_THREAD)

template<typename T, typename T_IN>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T, T_IN>& param, cudaStream_t stream)
{
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    // linear_bias_slopes, (head_num,) the slopes of the linear position bias.

    dim3 grid(param.q_length, param.batch_size, param.num_heads);
    if (param.batch_size * param.num_heads > 360) {
        grid.x = ceil(float(param.q_length) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && param.k_length % 2 == 0;
    dim3 block((param.k_length / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 4096 && block.x <= 8192) {
        LAUNCH_MAKSED_SOFTMAX(8);
    }
    else if (block.x > 2048 && block.x <= 4096) {
        LAUNCH_MAKSED_SOFTMAX(4)
    }
    else if (block.x > 1024) {
        LAUNCH_MAKSED_SOFTMAX(2)
    }
    else if (block.x > 0) {
        LAUNCH_MAKSED_SOFTMAX(1)
    }
    else {
        FT_CHECK(param.k_length <= 8192);
    }
}

#if ENABLE_FP32
template void invokeMaskedSoftmax(MaskedSoftmaxParam<float, float>& param, cudaStream_t stream);
#endif
template void invokeMaskedSoftmax(MaskedSoftmaxParam<half, float>& param, cudaStream_t stream);
template void invokeMaskedSoftmax(MaskedSoftmaxParam<half, half>& param, cudaStream_t stream);

#ifdef ENABLE_BF16
template<>
void invokeMaskedSoftmax(MaskedSoftmaxParam<__nv_bfloat16, float>& param, cudaStream_t stream)
{
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    // linear_bias_slopes, (head_num,) the slopes of the linear position bias.

    using T    = __nv_bfloat16;
    using T_IN = float;

    dim3 grid(param.q_length, param.batch_size, param.num_heads);
    if (param.batch_size * param.num_heads > 360) {
        grid.x = ceil(float(param.q_length) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && param.k_length % 2 == 0;
    dim3 block((param.k_length / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 2048 && block.x <= 4096) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 4);
    }
    else if (block.x > 1024) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 2);
    }
    else if (block.x > 0) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 1);
    }
    else {
        FT_CHECK(param.k_length <= 4096);
    }
}
template<>
void invokeMaskedSoftmax(MaskedSoftmaxParam<__nv_bfloat16, __nv_bfloat16>& param, cudaStream_t stream)
{
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    // linear_bias_slopes, (head_num,) the slopes of the linear position bias.

    using T    = __nv_bfloat16;
    using T_IN = __nv_bfloat16;

    dim3 grid(param.q_length, param.batch_size, param.num_heads);
    if (param.batch_size * param.num_heads > 360) {
        grid.x = ceil(float(param.q_length) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && param.k_length % 2 == 0;
    dim3 block((param.k_length / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 2048 && block.x <= 4096) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 4);
    }
    else if (block.x > 1024) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 2);
    }
    else if (block.x > 0) {
        LAUNCH_MAKSED_SOFTMAX_(__nv_bfloat16, 1);
    }
    else {
        FT_CHECK(param.k_length <= 4096);
    }
}

#endif

#undef LAUNCH_MAKSED_SOFTMAX
#undef LAUNCH_MAKSED_SOFTMAX_

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
