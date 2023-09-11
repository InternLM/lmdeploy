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

#include "src/turbomind/kernels/decoder_masked_multihead_attention_utils.h"
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

    const int bi = blockIdx.y;  // Batch index.
    const int hi = blockIdx.z;  // Head index.

    __shared__ float s_mean, s_max;

    const float linear_bias_slope = linear_bias_slopes != nullptr ? (float)linear_bias_slopes[hi] : 0.0f;

    // Loop along with Q dimension.
    for (int qi = blockIdx.x; qi < q_length; qi += gridDim.x) {

        float data[ITEMS_PER_THREAD];
        int   qk_offset;
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

            int   mask_offset = (bi * q_length + qi) * k_length + ki;
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

    const int bi = blockIdx.y;  // Batch index
    const int hi = blockIdx.z;  // Head index.

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
        int   qk_offset;
        float local_max = -1e20f;

        // Loop over k dimension.
        for (int i = 0; blockDim.x * i + threadIdx.x < (k_length / 2) && i < ITEMS_PER_THREAD; i++) {
            // The half of the index of k dimension. We will use the elements at {2 * ki, 2 * ki + 1}.
            int ki          = blockDim.x * i + threadIdx.x;
            qk_offset       = ((bi * head_num + hi) * q_length + qi) * (k_length / 2) + ki;
            int mask_offset = (bi * q_length + qi) * (k_length / 2) + ki;

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

    const int bi = blockIdx.y;  // Batch index
    const int hi = blockIdx.z;  // Head index.

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

        int qk_offset[Q_ITEMS_PER_THREAD];

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

            int mask_offset[Q_ITEMS_PER_THREAD];
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

    if (block.x > 2048 && block.x <= 4096) {
        LAUNCH_MAKSED_SOFTMAX(4)
    }
    else if (block.x > 1024) {
        LAUNCH_MAKSED_SOFTMAX(2)
    }
    else if (block.x > 0) {
        LAUNCH_MAKSED_SOFTMAX(1)
    }
    else {
        FT_CHECK(param.k_length <= 4096);
    }
}

template void invokeMaskedSoftmax(MaskedSoftmaxParam<float, float>& param, cudaStream_t stream);
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
__global__ void transpose(const T*     src,
                          T*           dst,
                          const int    batch_size,
                          const int    seq_len,
                          const int    head_num,
                          const int    size_per_head,
                          const float* scale,
                          int          int8_mode)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_id = tid / (head_num * seq_len * size_per_head);
    int head_id  = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
    int seq_id   = (tid % (seq_len * size_per_head)) / size_per_head;
    int id       = tid % size_per_head;

    int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);

    if (int8_mode == 2) {
        using Int8_Packed_T  = typename packed_as<int8_t, num_elems<T>::value>::type;
        using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;

        const Float_Packed_T scale_val = cuda_cast<Float_Packed_T>(*scale);
        reinterpret_cast<Int8_Packed_T*>(dst)[target_id] =
            cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(src[tid]) * scale_val);
    }
    else {
        dst[target_id] = src[tid];
    }
}

template<>
__global__ void transpose(const float* src,
                          float*       dst,
                          const int    batch_size,
                          const int    seq_len,
                          const int    head_num,
                          const int    size_per_head,
                          const float* scale,
                          int          int8_mode)
{
    int batch_id = blockIdx.x / (head_num * seq_len);
    int seq_id   = blockIdx.x % seq_len;
    int head_id  = (blockIdx.x % (head_num * seq_len)) / seq_len;

    const int target_id = batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
                          + head_id * size_per_head + threadIdx.x;
    const int src_id = blockIdx.x * size_per_head + threadIdx.x;

    if (int8_mode == 2) {
        const float scale_val                     = *scale;
        reinterpret_cast<int8_t*>(dst)[target_id] = cuda_cast<int8_t>(src[src_id] * scale_val);
    }
    else {
        dst[target_id] = src[src_id];
    }
}

template<typename T>
void invokeTransposeQKV(T*           dst,
                        T*           src,
                        const int    batch_size,
                        const int    seq_len,
                        const int    head_num,
                        const int    size_per_head,
                        const float* scale,
                        const int    int8_mode,
                        cudaStream_t stream)
{
    dim3 grid, block;
    if (sizeof(T) == 2) {
        int seq_per_block = 1;
        grid.x            = batch_size * head_num * seq_len / seq_per_block;
        while (seq_per_block < 4 && grid.x % 2 == 0) {
            grid.x /= 2;
            seq_per_block *= 2;
        }

        FT_CHECK(grid.x * seq_per_block == (size_t)batch_size * head_num * seq_len);

        if (seq_per_block * size_per_head % 2 == 0) {
            block.x = seq_per_block * size_per_head / 2;
            if (std::is_same<T, half>::value) {
                transpose<half2><<<grid, block, 0, stream>>>(
                    (half2*)src, (half2*)dst, batch_size, seq_len, head_num, size_per_head / 2, scale, int8_mode);
            }
#ifdef ENABLE_BF16
            else {
                transpose<__nv_bfloat162><<<grid, block, 0, stream>>>((__nv_bfloat162*)src,
                                                                      (__nv_bfloat162*)dst,
                                                                      batch_size,
                                                                      seq_len,
                                                                      head_num,
                                                                      size_per_head / 2,
                                                                      scale,
                                                                      int8_mode);
            }
#endif
        }
        else {
            block.x = seq_per_block * size_per_head;
            transpose<T>
                <<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head, scale, int8_mode);
        }
    }
    else {
        const int seq_per_block = 1;
        grid.x                  = batch_size * head_num * seq_len / seq_per_block;
        block.x                 = seq_per_block * size_per_head;
        transpose<T>
            <<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head, scale, int8_mode);
    }
}

#define INSTANTIATETRANSPOSEQKV(T)                                                                                     \
    template void invokeTransposeQKV(T*           src,                                                                 \
                                     T*           dst,                                                                 \
                                     const int    batch_size,                                                          \
                                     const int    seq_len,                                                             \
                                     const int    head_num,                                                            \
                                     const int    size_per_head,                                                       \
                                     const float* scale,                                                               \
                                     const int    int8_mode,                                                           \
                                     cudaStream_t stream)
INSTANTIATETRANSPOSEQKV(float);
INSTANTIATETRANSPOSEQKV(half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSEQKV(__nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSEQKV

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

    const int src_batch_id = (bid + mask_offset[bid]) / seq_len;
    const int src_seq_id   = (bid + mask_offset[bid]) % seq_len;

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
INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(float);
INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING(__nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSEATTENTIONOUTREMOVEPADDING

template<typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T* q_buf,
                                                   T* k_buf,
                                                   T* v_buf,
                                                   T* QKV,
                                                   const T* __restrict qkv_bias,
                                                   const int*   padding_offset,
                                                   const int    batch_size,
                                                   const int    seq_len,
                                                   const int    token_num,
                                                   const int    head_num,
                                                   const int    size_per_head,
                                                   const float* scale,
                                                   const int    int8_mode)
{
    // QKV: [token_num, 3, n]
    // qkv_bias: [3, n]
    // q_buf, k_buf, v_buf: [batch, head_num, seq_len, size_per_head]

    T*        qkv_ptr[3] = {q_buf, k_buf, v_buf};
    const int n          = head_num * size_per_head;
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < token_num * 3 * n;
         index += gridDim.x * blockDim.x) {
        const int bias_id = index % (3 * n);

        const int token_idx        = index / (3 * n);
        const int token_padded_idx = token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
        const int target_batch_id  = token_padded_idx / seq_len;
        const int seq_id           = token_padded_idx % seq_len;

        const int qkv_id  = (index % (3 * n)) / n;
        const int head_id = (index % n) / size_per_head;
        const int size_id = index % size_per_head;

        T val;
        if (int8_mode == 2) {
            val = cuda_cast<T>(cuda_cast<float>(reinterpret_cast<const int8_t*>(QKV)[index]) * scale[qkv_id]);
        }
        else {
            val = ldg(&QKV[index]);
        }
        val = val + ldg(&qkv_bias[bias_id]);

        if (int8_mode == 2) {
            // TODO(mseznec): add support for int8 BMM with FusedAtt
        }
        else {
            QKV[index] = val;
        }

        qkv_ptr[qkv_id][target_batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head
                        + seq_id * size_per_head + size_id] = val;
    }
}

template<typename T>
struct Vec_t {
    static constexpr int size = 0;
};

template<>
struct Vec_t<float> {
    using Type                = float2;
    static constexpr int size = 2;
};

template<>
struct Vec_t<half> {
    using Type                = uint32_t;
    static constexpr int size = 2;
};

#ifdef ENABLE_BF16
template<>
struct Vec_t<__nv_bfloat16> {
    using Type                = __nv_bfloat162;
    static constexpr int size = 2;
};
#endif

/// TODO: support batch step offset
template<typename T, bool PREFIX_PROMPT>
__global__ void add_fusedQKV_bias_transpose_kernel(T* q_buf,
                                                   T* k_buf,
                                                   T* v_buf,
                                                   T* QKV,
                                                   const T* __restrict qkv_bias,
                                                   const int* padding_offset,
                                                   const int* history_length,
                                                   const int* input_length,
                                                   int        batch_size,
                                                   int        seq_len,
                                                   int        head_num,
                                                   int        kv_head_num,
                                                   int        size_per_head,
                                                   int        rotary_embedding_dim,
                                                   float      rotary_embedding_base,
                                                   int        max_position_embeddings,
                                                   bool       use_dynamic_ntk,
                                                   bool       use_logn_attn)
{
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
    // QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].
    // For q and k, also apply the rotary embedding.

    // NOTE: QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //  QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    extern __shared__ __align__(sizeof(float2)) char smem_[];  // align on largest vector type

    constexpr int vec_size         = Vec_t<T>::size;
    using Vec_t                    = typename Vec_t<T>::Type;
    const int token_idx            = blockIdx.x;
    const int token_padding_offset = (padding_offset == nullptr || token_idx < 0) ? 0 : padding_offset[token_idx];
    const int tgt_token_idx        = token_idx + token_padding_offset;

    const int batch_idx = tgt_token_idx / seq_len;
    const int seq_idx   = tgt_token_idx % seq_len;

    const int head_idx = blockIdx.y;
    const int tidx     = threadIdx.x;

    const int total_seq_len = seq_len;

    const bool is_masked = tidx * vec_size >= size_per_head;

    const int hidden_idx = head_idx * size_per_head + tidx * vec_size;

    const int q_kv_head_num = head_num + 2 * kv_head_num;

    const int k_offset = head_num * size_per_head;
    const int v_offset = k_offset + kv_head_num * size_per_head;

    // src QKV: [batch, time, q_kv_head_num, hidden]
    const int src_q_idx = token_idx * q_kv_head_num * size_per_head + hidden_idx;
    const int src_k_idx = token_idx * q_kv_head_num * size_per_head + hidden_idx + k_offset;
    const int src_v_idx = token_idx * q_kv_head_num * size_per_head + hidden_idx + v_offset;

    Vec_t q, k, v;
    Vec_t q_bias, k_bias, v_bias;

    // load Q and apply bias
    if (!is_masked) {
        q = *reinterpret_cast<const Vec_t*>(&QKV[src_q_idx]);
        if (qkv_bias) {
            q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
            q      = mmha::add(q, q_bias);
        }
    }

    // load KV and apply bias
    if (!is_masked && head_idx < kv_head_num) {
        k = *reinterpret_cast<const Vec_t*>(&QKV[src_k_idx]);
        v = *reinterpret_cast<const Vec_t*>(&QKV[src_v_idx]);
        if (qkv_bias) {
            k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + k_offset]);
            v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + v_offset]);
            k      = mmha::add(k, k_bias);
            v      = mmha::add(v, v_bias);
        }
    }

    const int history_len = history_length[batch_idx];
    const int context_len = history_len + input_length[batch_idx];
    const int timestep    = history_len + seq_idx;

    if (use_dynamic_ntk) {
        rotary_embedding_base = mmha::rotary_embedding_get_base(
            context_len, max_position_embeddings, rotary_embedding_dim, rotary_embedding_base);
    }

    // TODO: unused computation on k if GQA is used
    mmha::apply_rotary_embedding(q, k, tidx, rotary_embedding_dim, rotary_embedding_base, timestep);

    if (use_logn_attn) {
        // +1 to convert to context length at the timestep
        float logn_scaling = mmha::logn_attn_get_scaling(timestep + 1, max_position_embeddings);
        if constexpr (std::is_same_v<T, float>) {
            q = mmha::mul<Vec_t, float, Vec_t>(logn_scaling, q);
        }
        else if constexpr (std::is_same_v<T, half>) {
            half tmp = __float2half(logn_scaling);
            q        = mmha::mul<Vec_t, uint16_t, Vec_t>((uint16_t&)tmp, q);
        }
    }

    if (!is_masked && !q_buf) {  // also skip modifying QKV if q/k/v_buf are present
        *reinterpret_cast<Vec_t*>(&QKV[src_q_idx]) = q;
        if (head_idx < kv_head_num) {
            *reinterpret_cast<Vec_t*>(&QKV[src_k_idx]) = k;
            *reinterpret_cast<Vec_t*>(&QKV[src_v_idx]) = v;
        }
    }

    const int dest_q_idx = batch_idx * size_per_head * seq_len * head_num + head_idx * size_per_head * seq_len
                           + seq_idx * size_per_head + tidx * vec_size;

    const int dest_kv_idx = batch_idx * size_per_head * total_seq_len * kv_head_num
                            + head_idx * size_per_head * total_seq_len + seq_idx * size_per_head + tidx * vec_size;

    if (!is_masked) {
        *reinterpret_cast<Vec_t*>(&q_buf[dest_q_idx]) = q;
        if (head_idx < kv_head_num) {
            *reinterpret_cast<Vec_t*>(&k_buf[dest_kv_idx]) = k;
            *reinterpret_cast<Vec_t*>(&v_buf[dest_kv_idx]) = v;
        }
    }
}

#define FUSED_QKV_BIAS_TRANSPOSE_LAUNCH(T, PREFIX_PROMPT)                                                              \
    add_fusedQKV_bias_transpose_kernel<T, PREFIX_PROMPT><<<grid, block, smem_size, stream>>>(q_buf,                    \
                                                                                             k_buf,                    \
                                                                                             v_buf,                    \
                                                                                             QKV,                      \
                                                                                             qkv_bias,                 \
                                                                                             padding_offset,           \
                                                                                             history_length,           \
                                                                                             input_length,             \
                                                                                             batch_size,               \
                                                                                             seq_len,                  \
                                                                                             head_num,                 \
                                                                                             kv_head_num,              \
                                                                                             size_per_head,            \
                                                                                             rotary_embedding_dim,     \
                                                                                             rotary_embedding_base,    \
                                                                                             max_position_embeddings,  \
                                                                                             use_dynamic_ntk,          \
                                                                                             use_logn_attn);

template<typename T>
void invokeAddFusedQKVBiasTranspose(T*           q_buf,
                                    T*           k_buf,
                                    T*           v_buf,
                                    T*           QKV,
                                    const T*     qkv_bias,
                                    const int*   padding_offset,
                                    const int*   history_length,
                                    const int*   input_length,
                                    const int    batch_size,
                                    const int    seq_len,
                                    const int    token_num,
                                    const int    head_num,
                                    const int    kv_head_num,
                                    const int    size_per_head,
                                    const int    rotary_embedding_dim,
                                    float        rotary_embedding_base,
                                    int          max_position_embeddings,
                                    bool         use_dynamic_ntk,
                                    bool         use_logn_attn,
                                    cudaStream_t stream)
{
    FT_CHECK(rotary_embedding_dim);
    // To implement rotary embeddings, each thread processes two QKV elems:
    dim3   block((size_per_head / Vec_t<T>::size + 31) / 32 * 32);
    dim3   grid(token_num, head_num);
    size_t smem_size = 0;
    FUSED_QKV_BIAS_TRANSPOSE_LAUNCH(T, false);
}

#define INSTANTIATEADDFUSEDQKVBIASTRANSPOSE(T)                                                                         \
    template void invokeAddFusedQKVBiasTranspose(T*           q_buf,                                                   \
                                                 T*           k_buf,                                                   \
                                                 T*           v_buf,                                                   \
                                                 T*           QKV,                                                     \
                                                 const T*     qkv_bias,                                                \
                                                 const int*   padding_offset,                                          \
                                                 const int*   history_length,                                          \
                                                 const int*   input_length,                                            \
                                                 const int    batch_size,                                              \
                                                 const int    seq_len,                                                 \
                                                 const int    token_num,                                               \
                                                 const int    head_num,                                                \
                                                 const int    kv_head_num,                                             \
                                                 const int    size_per_head,                                           \
                                                 const int    rotary_embedding_dim,                                    \
                                                 float        rotary_embedding_base,                                   \
                                                 int          max_position_embeddings,                                 \
                                                 bool         use_dynamic_ntk,                                         \
                                                 bool         use_logn_attn,                                           \
                                                 cudaStream_t stream)
INSTANTIATEADDFUSEDQKVBIASTRANSPOSE(float);
INSTANTIATEADDFUSEDQKVBIASTRANSPOSE(half);
#ifdef ENABLE_BF16
INSTANTIATEADDFUSEDQKVBIASTRANSPOSE(__nv_bfloat16);
#endif
#undef INSTANTIATEADDFUSEDQKVBIASTRANSPOSE

template<typename T>
__global__ void transpose_4d(T*        dst,
                             T*        src,
                             const int dim0,
                             const int dim1,
                             const int dim2,
                             const int dim3,
                             const int dim0_leading_dim,
                             const int ite)
{
    // transpose from [dim0, dim1, dim2, dim3] to [dim2, X, dim1, dim3]
    // where the dimension of X is dim0_leading_dim, and offset is ite * dim0
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dim0 * dim1 * dim2 * dim3; i += blockDim.x * gridDim.x) {
        int       index = i;
        const int d3    = index % dim3;
        index           = (index - d3) / dim3;
        const int d2    = index % dim2;
        index           = (index - d2) / dim2;
        const int d1    = index % dim1;
        index           = (index - d1) / dim1;
        const int d0    = index % dim0;
        index           = (index - d0) / dim0;
        dst[d2 * dim0_leading_dim * dim1 * dim3 + (d0 + dim0 * ite) * dim1 * dim3 + d1 * dim3 + d3] = src[i];
    }
}

template<>
__global__ void transpose_4d(half*     dst,
                             half*     src,
                             const int dim0,
                             const int dim1,
                             const int dim2,
                             const int dim3,
                             const int dim0_leading_dim,
                             const int ite)
{
    half2*    dst_ptr   = (half2*)dst;
    half2*    src_ptr   = (half2*)src;
    const int half_dim3 = dim3 / 2;
    // transpose from [dim0, dim1, dim2, half_dim3] to [dim2, dim0, dim1, half_dim3]
    // where the dimension of X is dim0_leading_dim, and offset is ite * dim0
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dim0 * dim1 * dim2 * half_dim3;
         i += blockDim.x * gridDim.x) {
        int       index = i;
        const int d3    = index % half_dim3;
        index           = (index - d3) / half_dim3;
        const int d2    = index % dim2;
        index           = (index - d2) / dim2;
        const int d1    = index % dim1;
        index           = (index - d1) / dim1;
        const int d0    = index % dim0;
        index           = (index - d0) / dim0;
        dst_ptr[d2 * dim0_leading_dim * dim1 * half_dim3 + (d0 + dim0 * ite) * dim1 * half_dim3 + d1 * half_dim3 + d3] =
            src_ptr[i];
    }
}

template<typename T>
void invokeTranspose4d(T*           dst,
                       T*           src,
                       const int    local_batch_size,
                       const int    seq_len,
                       const int    size_per_head,
                       const int    local_hidden_units,
                       const int    local_head_num,
                       const int    batch_size,
                       const int    ite,
                       cudaStream_t stream)
{
    transpose_4d<<<local_batch_size * seq_len * local_hidden_units / 512, 512 / (4 / (sizeof(T))), 0, stream>>>(
        dst, src, local_batch_size, local_head_num, seq_len, size_per_head, batch_size, ite);
}

#define INSTANTIATETRANSPOSE4D(T)                                                                                      \
    template void invokeTranspose4d(T*           dst,                                                                  \
                                    T*           src,                                                                  \
                                    const int    local_batch_size,                                                     \
                                    const int    seq_len,                                                              \
                                    const int    size_per_head,                                                        \
                                    const int    local_hidden_units,                                                   \
                                    const int    local_head_num,                                                       \
                                    const int    batch_size,                                                           \
                                    const int    ite,                                                                  \
                                    cudaStream_t stream)
INSTANTIATETRANSPOSE4D(float);
INSTANTIATETRANSPOSE4D(half);
#undef INSTANTIATETRANSPOSE4D
template<typename T>
__global__ void transpose_4d_batch_major_k_cache(
    T* k_dst, const T* k_src, const int head_num, const int size_per_head, const int seq_len, const int max_seq_len)
{
    const int     batch_id = blockIdx.y;
    const int     head_id  = blockIdx.z;
    constexpr int X_ELEMS  = (sizeof(T) == 4) ? 4 : 8;

    auto key_src = reinterpret_cast<const uint4*>(k_src + batch_id * head_num * size_per_head * seq_len
                                                  + head_id * size_per_head * seq_len);
    auto key_dst = reinterpret_cast<uint4*>(k_dst + batch_id * head_num * size_per_head * max_seq_len
                                            + head_id * size_per_head * max_seq_len);

    const int out_idx             = blockIdx.x * blockDim.x + threadIdx.x;
    int       size_per_head_div_x = size_per_head / X_ELEMS;
    if (out_idx >= size_per_head_div_x * max_seq_len) {
        return;
    }

    int       idx            = out_idx;
    const int k_seq_len_id   = idx % max_seq_len;
    idx                      = (idx - k_seq_len_id) / max_seq_len;
    const int k_head_size_id = idx % size_per_head_div_x;

    if (k_seq_len_id < seq_len) {
        key_dst[out_idx] = key_src[k_seq_len_id * size_per_head_div_x + k_head_size_id];
    }
}

template<typename T>
__global__ void transpose_4d_batch_major_v_cache(
    T* v_dst, const T* v_src, const int head_num, const int size_per_head, const int seq_len, const int max_seq_len)
{
    const int batch_id = blockIdx.y;
    const int head_id  = blockIdx.z;

    // 16 byte loads will handle "x" dimension
    auto val_src = reinterpret_cast<const uint4*>(v_src + batch_id * head_num * size_per_head * seq_len
                                                  + head_id * size_per_head * seq_len);
    auto val_dst = reinterpret_cast<uint4*>(v_dst + batch_id * head_num * size_per_head * max_seq_len
                                            + head_id * size_per_head * max_seq_len);

    // idx is over output dimension L * size_per_head / x for values
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int X_ELEMS             = (sizeof(T) == 4) ? 4 : 8;
    const int     size_per_head_div_x = size_per_head / X_ELEMS;

    if (idx >= size_per_head_div_x * seq_len) {
        return;
    }

    val_dst[idx] = val_src[idx];
}

template<typename T>
void invokeTranspose4dBatchMajor(T*           k_dst,
                                 T*           v_dst,
                                 const T*     k_src,
                                 const T*     v_src,
                                 const int    local_batch_size,
                                 const int    seq_len,
                                 const int    max_seq_len,
                                 const int    size_per_head,
                                 const int    local_head_num,
                                 cudaStream_t stream)
{
    constexpr int block_sz = 128;
    constexpr int x        = (sizeof(T) == 4) ? 4 : 8;
    int           size     = max_seq_len * size_per_head / x;
    dim3          grid((size + block_sz - 1) / block_sz, local_batch_size, local_head_num);
    dim3          grid_v((seq_len * size_per_head / x + block_sz - 1) / block_sz, local_batch_size, local_head_num);

    transpose_4d_batch_major_k_cache<<<grid, block_sz, 0, stream>>>(
        k_dst, k_src, local_head_num, size_per_head, seq_len, max_seq_len);

    transpose_4d_batch_major_v_cache<<<grid_v, block_sz, 0, stream>>>(
        v_dst, v_src, local_head_num, size_per_head, seq_len, max_seq_len);
}

#define INSTANTIATETRANSPOSE4DBATCHMAJOR(T)                                                                            \
    template void invokeTranspose4dBatchMajor(T*           k_dst,                                                      \
                                              T*           v_dst,                                                      \
                                              const T*     k_src,                                                      \
                                              const T*     v_src,                                                      \
                                              const int    local_batch_size,                                           \
                                              const int    seq_len,                                                    \
                                              const int    max_seq_len,                                                \
                                              const int    size_per_head,                                              \
                                              const int    local_head_num,                                             \
                                              cudaStream_t stream)
INSTANTIATETRANSPOSE4DBATCHMAJOR(float);
INSTANTIATETRANSPOSE4DBATCHMAJOR(half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSE4DBATCHMAJOR(__nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSE4DBATCHMAJOR

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
INSTANTIATEADDRELATIVEATTENTIONBIAS(float);
INSTANTIATEADDRELATIVEATTENTIONBIAS(half);
#ifdef ENABLE_BF16
INSTANTIATEADDRELATIVEATTENTIONBIAS(__nv_bfloat16);
#endif
#undef INSTANTIATEADDRELATIVEATTENTIONBIAS

/*******************  invokeAddHead3SizeQKVBias  ***********************/
// m = batch*window_num*window_len
// mm_qkv is [m, head*3*size_per_head] row-major
// bias_qkv is [head*3*size_per_head]
// q_buf_, k_buf_, v_buf_ is [batch*window_num, num_head, window_len, size_per_head] row-major
// grid(window_len, window_num, 3*batch);
// block(num_head * size_per_head)
template<typename T>
__global__ void add_head3Size_QKV_bias(const T*  mm_qkv,
                                       const T*  bias_qkv,
                                       T*        q_buf_,
                                       T*        k_buf_,
                                       T*        v_buf_,
                                       const int batch,
                                       const int window_num,
                                       const int window_len,
                                       const int num_head,
                                       const int size_per_head)
{

    T*  buf_ptr;
    int qkv_id = blockIdx.z / batch;
    if (qkv_id == 0) {
        buf_ptr = q_buf_;
    }
    else if (qkv_id == 1) {
        buf_ptr = k_buf_;
    }
    else {
        buf_ptr = v_buf_;
    }

    const int batch_id   = blockIdx.z % batch;
    const int token_id   = blockIdx.x;
    const int window_id  = blockIdx.y;
    const int head_id    = threadIdx.x / size_per_head;
    const int id_in_head = threadIdx.x % size_per_head;

    const int bias_idx = (head_id * 3 + qkv_id) * size_per_head + id_in_head;
    const T   bias     = ldg(bias_qkv + bias_idx);

    const int input_idx =
        ((batch_id * window_num + window_id) * window_len + token_id) * num_head * 3 * size_per_head + bias_idx;
    T tmp = mm_qkv[input_idx] + bias;

    int target_id = (((batch_id * window_num + window_id) * num_head + head_id) * window_len + token_id) * size_per_head
                    + id_in_head;
    ;
    buf_ptr[target_id] = tmp;
}

// for float2, size_per_head /= 2
// m = batch*window_num*window_len
// mm_qkv is [m, head*3*size_per_head] row-major
// bias_qkv is [head*3*size_per_head]
// q_buf_, k_buf_, v_buf_ is [batch*window_num, num_head, window_len, size_per_head] row-major
// grid(window_len, window_num, 3*batch);
// block(num_head * size_per_head)
template<>
__global__ void add_head3Size_QKV_bias(const float2* mm_qkv,
                                       const float2* bias_qkv,
                                       float2*       q_buf_,
                                       float2*       k_buf_,
                                       float2*       v_buf_,
                                       const int     batch,
                                       const int     window_num,
                                       const int     window_len,
                                       const int     num_head,
                                       const int     size_per_head)
{

    float2* buf_ptr;
    int     qkv_id = blockIdx.z / batch;
    if (qkv_id == 0) {
        buf_ptr = q_buf_;
    }
    else if (qkv_id == 1) {
        buf_ptr = k_buf_;
    }
    else {
        buf_ptr = v_buf_;
    }

    const int batch_id   = blockIdx.z % batch;
    const int token_id   = blockIdx.x;
    const int window_id  = blockIdx.y;
    const int head_id    = threadIdx.x / size_per_head;
    const int id_in_head = threadIdx.x % size_per_head;

    const int    bias_idx = (head_id * 3 + qkv_id) * size_per_head + id_in_head;
    const float2 bias     = ldg(bias_qkv + bias_idx);

    const int input_idx =
        ((batch_id * window_num + window_id) * window_len + token_id) * num_head * 3 * size_per_head + bias_idx;
    float2 tmp = mm_qkv[input_idx];
    tmp.x += bias.x;
    tmp.y += bias.y;

    int target_id = (((batch_id * window_num + window_id) * num_head + head_id) * window_len + token_id) * size_per_head
                    + id_in_head;
    ;
    buf_ptr[target_id] = tmp;
}

// for half2, size_per_head /= 2
// m = batch*window_num*window_len
// mm_qkv is [m, head*3*size_per_head] row-major
// bias_qkv is [head*3*size_per_head]
// q_buf_, k_buf_, v_buf_ is [batch*window_num, num_head, window_len, size_per_head] row-major
// grid(window_len, window_num, batch);
// block(num_head * size_per_head)
template<>
__global__ void add_head3Size_QKV_bias(const half2* mm_qkv,
                                       const half2* bias_qkv,
                                       half2*       q_buf_,
                                       half2*       k_buf_,
                                       half2*       v_buf_,
                                       const int    batch,
                                       const int    window_num,
                                       const int    window_len,
                                       const int    num_head,
                                       const int    size_per_head)
{

    const int batch_id   = blockIdx.z;
    const int token_id   = blockIdx.x;
    const int window_id  = blockIdx.y;
    const int head_id    = threadIdx.x / size_per_head;
    const int id_in_head = threadIdx.x % size_per_head;

    const int input_offset =
        ((batch_id * window_num + window_id) * window_len + token_id) * num_head * 3 * size_per_head;
    const int target_id =
        (((batch_id * window_num + window_id) * num_head + head_id) * window_len + token_id) * size_per_head
        + id_in_head;

    int   qkv_id      = 0;
    int   bias_idx    = (head_id * 3 + qkv_id) * size_per_head + id_in_head;
    half2 bias        = __ldg(bias_qkv + bias_idx);
    int   input_idx   = input_offset + bias_idx;
    half2 tmp         = mm_qkv[input_idx];
    tmp               = __hadd2(tmp, bias);
    q_buf_[target_id] = tmp;

    qkv_id            = 1;
    bias_idx          = (head_id * 3 + qkv_id) * size_per_head + id_in_head;
    bias              = __ldg(bias_qkv + bias_idx);
    input_idx         = input_offset + bias_idx;
    tmp               = mm_qkv[input_idx];
    tmp               = __hadd2(tmp, bias);
    k_buf_[target_id] = tmp;

    qkv_id            = 2;
    bias_idx          = (head_id * 3 + qkv_id) * size_per_head + id_in_head;
    bias              = __ldg(bias_qkv + bias_idx);
    input_idx         = input_offset + bias_idx;
    tmp               = mm_qkv[input_idx];
    tmp               = __hadd2(tmp, bias);
    v_buf_[target_id] = tmp;
}

#ifdef ENABLE_BF16
template<>
__global__ void add_head3Size_QKV_bias(const __nv_bfloat162* mm_qkv,
                                       const __nv_bfloat162* bias_qkv,
                                       __nv_bfloat162*       q_buf_,
                                       __nv_bfloat162*       k_buf_,
                                       __nv_bfloat162*       v_buf_,
                                       const int             batch,
                                       const int             window_num,
                                       const int             window_len,
                                       const int             num_head,
                                       const int             size_per_head)
{

    const int batch_id   = blockIdx.z;
    const int token_id   = blockIdx.x;
    const int window_id  = blockIdx.y;
    const int head_id    = threadIdx.x / size_per_head;
    const int id_in_head = threadIdx.x % size_per_head;

    const int input_offset =
        ((batch_id * window_num + window_id) * window_len + token_id) * num_head * 3 * size_per_head;
    const int target_id =
        (((batch_id * window_num + window_id) * num_head + head_id) * window_len + token_id) * size_per_head
        + id_in_head;

    int            qkv_id    = 0;
    int            bias_idx  = (head_id * 3 + qkv_id) * size_per_head + id_in_head;
    __nv_bfloat162 bias      = ldg(bias_qkv + bias_idx);
    int            input_idx = input_offset + bias_idx;
    __nv_bfloat162 tmp       = mm_qkv[input_idx];
    tmp                      = bf16hadd2(tmp, bias);
    q_buf_[target_id]        = tmp;

    qkv_id            = 1;
    bias_idx          = (head_id * 3 + qkv_id) * size_per_head + id_in_head;
    bias              = ldg(bias_qkv + bias_idx);
    input_idx         = input_offset + bias_idx;
    tmp               = mm_qkv[input_idx];
    tmp               = bf16hadd2(tmp, bias);
    k_buf_[target_id] = tmp;

    qkv_id            = 2;
    bias_idx          = (head_id * 3 + qkv_id) * size_per_head + id_in_head;
    bias              = ldg(bias_qkv + bias_idx);
    input_idx         = input_offset + bias_idx;
    tmp               = mm_qkv[input_idx];
    tmp               = bf16hadd2(tmp, bias);
    v_buf_[target_id] = tmp;
}
#endif

template<typename T>
void invokeAddHead3SizeQKVBias(const T*     mm_qkv,
                               const T*     bias_qkv,
                               T*           q_buf_,
                               T*           k_buf_,
                               T*           v_buf_,
                               const int    batch,
                               const int    window_num,
                               const int    window_len,
                               const int    num_head,
                               const int    size_per_head,
                               cudaStream_t stream)
{
    if (std::is_same<T, float>::value) {
        dim3 grid(window_len, window_num, 3 * batch);
        dim3 block(num_head * size_per_head);

        if (block.x < 1024) {
            add_head3Size_QKV_bias<<<grid, block, 0, stream>>>(
                mm_qkv, bias_qkv, q_buf_, k_buf_, v_buf_, batch, window_num, window_len, num_head, size_per_head);
        }
        else if ((block.x % 2 == 0) && (block.x / 2 < 1024)) {
            block.x /= 2;
            add_head3Size_QKV_bias<<<grid, block, 0, stream>>>((const float2*)mm_qkv,
                                                               (const float2*)bias_qkv,
                                                               (float2*)q_buf_,
                                                               (float2*)k_buf_,
                                                               (float2*)v_buf_,
                                                               batch,
                                                               window_num,
                                                               window_len,
                                                               num_head,
                                                               size_per_head / 2);
        }
        else {
            printf("[ERROR][invokeAddHead3SizeQKVBias] unsupported block.x!\n");
            exit(-1);
        }
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) {
#else
    else if (std::is_same<T, half>::value) {
#endif
        dim3 grid(window_len, window_num, batch);
        dim3 block(num_head * size_per_head / 2);

        using T2 = typename TypeConverter<T>::Type;  // half2 or bfloat16

        if (block.x > 1024) {
            printf("[ERROR][invokeAddHead3SizeQKVBias] block.x > 1024!\n");
            exit(-1);
        }

        add_head3Size_QKV_bias<<<grid, block, 0, stream>>>((const T2*)mm_qkv,
                                                           (const T2*)bias_qkv,
                                                           (T2*)q_buf_,
                                                           (T2*)k_buf_,
                                                           (T2*)v_buf_,
                                                           batch,
                                                           window_num,
                                                           window_len,
                                                           num_head,
                                                           size_per_head / 2);
    }
}

#define INSTANTIATEADDHEAD3SIZEQKVBIAS(T)                                                                              \
    template void invokeAddHead3SizeQKVBias<T>(const T*     mm_qkv,                                                    \
                                               const T*     bias_qkv,                                                  \
                                               T*           q_buf_,                                                    \
                                               T*           k_buf_,                                                    \
                                               T*           v_buf_,                                                    \
                                               const int    batch,                                                     \
                                               const int    window_num,                                                \
                                               const int    window_len,                                                \
                                               const int    num_head,                                                  \
                                               const int    size_per_head,                                             \
                                               cudaStream_t stream)
INSTANTIATEADDHEAD3SIZEQKVBIAS(float);
INSTANTIATEADDHEAD3SIZEQKVBIAS(half);
#ifdef ENABLE_BF16
INSTANTIATEADDHEAD3SIZEQKVBIAS(__nv_bfloat16);
#endif
#undef INSTANTIATEADDHEAD3SIZEQKVBIAS

/*******************  invokeMaskedSoftMaxWithRelPosBias  ***********************/

// grid = (window_len/word_per_thread, window_num*num_head, batch_size)
// block.x = max(32, (window_len + 31)/32*32)
// qk_buf is [batch, window_num, num_head, window_len, window_len]
// attn_mask is [window_num, window_len, window_len] + row-major
// relative_pos_bias is [num_head, window_len, window_len] + row-majot
template<typename T>
__global__ void softmax_withRelPosBias_element1_kernel(T*          qk_buf,
                                                       const T*    attn_mask,
                                                       const T*    relative_pos_bias,
                                                       const int   batch_size,
                                                       const int   num_head,
                                                       const int   window_num,
                                                       const int   window_len,
                                                       const int   window_len_x_window_len,
                                                       const float qk_scale)
{

    bool qual = threadIdx.x < window_len;
    for (int window_id = blockIdx.x; window_id < window_len; window_id += gridDim.x) {
        float            tmp = -1e20f;
        __shared__ float s_mean, s_max;
        int              qk_offset;
        if (qual) {
            const int offset_in_window = window_id * window_len + threadIdx.x;
            qk_offset = (blockIdx.z * gridDim.y + blockIdx.y) * window_len_x_window_len + offset_in_window;
            const int relative_pos_bias_offset = (blockIdx.y % num_head) * window_len_x_window_len + offset_in_window;
            float     mask_val =
                (attn_mask == nullptr) ?
                        0.0f :
                        static_cast<float>(
                        ldg(attn_mask + ((blockIdx.y / num_head) * window_len_x_window_len + offset_in_window)));
            tmp = qk_scale * static_cast<float>(qk_buf[qk_offset]) + mask_val
                  + static_cast<float>(ldg(relative_pos_bias + relative_pos_bias_offset));
        }

        float max_val = blockReduceMax<float>(tmp);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float qk_tmp  = qual ? __expf(tmp - s_max) : 0.0f;
        float sum_val = blockReduceSum<float>(qk_tmp);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();
        if (qual) {
            qk_buf[qk_offset] = (T)(qk_tmp * s_mean);
        }
    }
}

// grid = (window_len/word_per_thread, window_num*num_head, batch_size)
// block.x = max(32, (window_len/2 + 31)/32*32)
// qk_buf is [batch, window_num, num_head, window_len, window_len]
// attn_mask is [window_num, window_len, window_len] + row-major
// relative_pos_bias is [num_head, window_len, window_len] + row-majot
template<typename T2, typename T>
__global__ void softmax_withRelPosBias_element2_kernel(T2*         qk_buf,
                                                       const T2*   attn_mask,
                                                       const T2*   relative_pos_bias,
                                                       const int   batch_size,
                                                       const int   num_head,
                                                       const int   window_num,
                                                       const int   window_len,
                                                       const int   window_len_x_window_len,
                                                       const float qk_scale)
{
    const int window_len_2 = window_len / 2;
    const int tidx         = threadIdx.x;
    bool      qual         = tidx < window_len_2;
    const T2  zero         = {T(0.0f), T(0.0f)};
    const int bdim         = blockDim.x;
    for (int window_id = blockIdx.x; window_id < window_len; window_id += gridDim.x) {
        float            tmp = -1e20f;
        __shared__ float s_mean, s_max;
        int              qk_offset;
        float2           local_qk_val;
        T2               qk_val;
        if (qual) {
            const int offset_in_window = window_id * window_len + 2 * tidx;
            qk_offset = ((blockIdx.z * gridDim.y + blockIdx.y) * window_len_x_window_len + offset_in_window) / 2;
            const int relative_pos_bias_offset =
                ((blockIdx.y % num_head) * window_len_x_window_len + offset_in_window) / 2;
            T2 mask_val =
                (attn_mask == nullptr) ?
                    zero :
                    ldg(attn_mask + ((blockIdx.y / num_head) * window_len_x_window_len + offset_in_window) / 2);
            qk_val            = qk_buf[qk_offset];
            local_qk_val.x    = static_cast<float>(qk_val.x);
            local_qk_val.y    = static_cast<float>(qk_val.y);
            const T2 bias_val = ldg(relative_pos_bias + relative_pos_bias_offset);
            local_qk_val.x =
                qk_scale * local_qk_val.x + static_cast<float>(mask_val.x) + static_cast<float>(bias_val.x);
            local_qk_val.y =
                qk_scale * local_qk_val.y + static_cast<float>(mask_val.y) + static_cast<float>(bias_val.y);
            tmp = local_qk_val.x > local_qk_val.y ? local_qk_val.x : local_qk_val.y;
        }

        float max_val = bdim <= 32 ? warpReduceMax<float>(tmp) : blockReduceMax<float>(tmp);
        if (tidx == 0) {
            s_max = max_val;
        }
        __syncthreads();

        local_qk_val.x = qual ? __expf(local_qk_val.x - s_max) : 0.0f;
        local_qk_val.y = qual ? __expf(local_qk_val.y - s_max) : 0.0f;

        float sum_val = bdim <= 32 ? warpReduceSum<float>(local_qk_val.x + local_qk_val.y) :
                                     blockReduceSum<float>(local_qk_val.x + local_qk_val.y);
        if (tidx == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();
        if (qual) {
            local_qk_val.x    = local_qk_val.x * s_mean;
            local_qk_val.y    = local_qk_val.y * s_mean;
            qk_val.x          = T(local_qk_val.x);
            qk_val.y          = T(local_qk_val.y);
            qk_buf[qk_offset] = qk_val;
        }
    }
}

// grid = (window_len/word_per_thread, window_num*num_head, batch_size)
// block.x = max(32, (window_len/4 + 31)/32*32)
// qk_buf is [batch, window_num, num_head, window_len, window_len]
// attn_mask is [window_num, window_len, window_len] + row-major
// relative_pos_bias is [num_head, window_len, window_len] + row-majot
template<typename T4, typename T>
__global__ void softmax_withRelPosBias_element4_kernel(T4*         qk_buf,
                                                       const T4*   attn_mask,
                                                       const T4*   relative_pos_bias,
                                                       const int   batch_size,
                                                       const int   num_head,
                                                       const int   window_num,
                                                       const int   window_len,
                                                       const int   window_len_x_window_len,
                                                       const float qk_scale)
{
    const int window_len_4 = window_len / 4;
    const int tidx         = threadIdx.x;
    bool      qual         = tidx < window_len_4;
    const T4  zero         = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
    const int bdim         = blockDim.x;
    for (int window_id = blockIdx.x; window_id < window_len; window_id += gridDim.x) {
        float            tmp = -1e20f;
        __shared__ float s_mean, s_max;
        int              qk_offset;
        float4           local_qk_val;
        T4               qk_val;
        if (qual) {
            const int offset_in_window = window_id * window_len + 4 * tidx;
            qk_offset = ((blockIdx.z * gridDim.y + blockIdx.y) * window_len_x_window_len + offset_in_window) / 4;
            const int relative_pos_bias_offset =
                ((blockIdx.y % num_head) * window_len_x_window_len + offset_in_window) / 4;
            T4 mask_val       = (attn_mask == nullptr) ?
                                    zero :
                                    attn_mask[((blockIdx.y / num_head) * window_len_x_window_len + offset_in_window) / 4];
            qk_val            = qk_buf[qk_offset];
            local_qk_val.x    = static_cast<float>(qk_val.x);
            local_qk_val.y    = static_cast<float>(qk_val.y);
            local_qk_val.z    = static_cast<float>(qk_val.z);
            local_qk_val.w    = static_cast<float>(qk_val.w);
            const T4 bias_val = relative_pos_bias[relative_pos_bias_offset];
            local_qk_val.x =
                qk_scale * local_qk_val.x + static_cast<float>(mask_val.x) + static_cast<float>(bias_val.x);
            local_qk_val.y =
                qk_scale * local_qk_val.y + static_cast<float>(mask_val.y) + static_cast<float>(bias_val.y);
            local_qk_val.z =
                qk_scale * local_qk_val.z + static_cast<float>(mask_val.z) + static_cast<float>(bias_val.z);
            local_qk_val.w =
                qk_scale * local_qk_val.w + static_cast<float>(mask_val.w) + static_cast<float>(bias_val.w);
            tmp = local_qk_val.x > local_qk_val.y ? local_qk_val.x : local_qk_val.y;
            tmp = tmp > local_qk_val.z ? tmp : local_qk_val.z;
            tmp = tmp > local_qk_val.w ? tmp : local_qk_val.w;
        }

        float max_val = bdim <= 32 ? warpReduceMax<float>(tmp) : blockReduceMax<float>(tmp);
        if (tidx == 0) {
            s_max = max_val;
        }
        __syncthreads();

        local_qk_val.x = qual ? __expf(local_qk_val.x - s_max) : 0.0f;
        local_qk_val.y = qual ? __expf(local_qk_val.y - s_max) : 0.0f;
        local_qk_val.z = qual ? __expf(local_qk_val.z - s_max) : 0.0f;
        local_qk_val.w = qual ? __expf(local_qk_val.w - s_max) : 0.0f;

        float sum_val = bdim <= 32 ?
                            warpReduceSum<float>(local_qk_val.x + local_qk_val.y + local_qk_val.z + local_qk_val.w) :
                            blockReduceSum<float>(local_qk_val.x + local_qk_val.y + local_qk_val.z + local_qk_val.w);
        if (tidx == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();
        if (qual) {
            local_qk_val.x    = local_qk_val.x * s_mean;
            local_qk_val.y    = local_qk_val.y * s_mean;
            local_qk_val.z    = local_qk_val.z * s_mean;
            local_qk_val.w    = local_qk_val.w * s_mean;
            qk_val.x          = T(local_qk_val.x);
            qk_val.y          = T(local_qk_val.y);
            qk_val.z          = T(local_qk_val.z);
            qk_val.w          = T(local_qk_val.w);
            qk_buf[qk_offset] = qk_val;
        }
    }
}

template<typename T>
void invokeMaskedSoftMaxWithRelPosBias(T*           qk_buf,
                                       const T*     attn_mask,
                                       const T*     relative_pos_bias,
                                       const int    batch_size,
                                       const int    num_head,
                                       const int    window_num,
                                       const int    window_len,
                                       float        qk_scale,
                                       cudaStream_t stream)
{
    const int word_per_thread = 1;
    dim3      grid((window_len + word_per_thread - 1) / word_per_thread, window_num * num_head, batch_size);
    if ((window_len % 4 == 0) && window_len / 4 >= 32) {
        dim3 block((window_len / 4 + 31) / 32 * 32);
        if (std::is_same<T, float>::value) {
            softmax_withRelPosBias_element4_kernel<float4, float>
                <<<grid, block, 0, stream>>>((float4*)qk_buf,
                                             (const float4*)attn_mask,
                                             (const float4*)relative_pos_bias,
                                             batch_size,
                                             num_head,
                                             window_num,
                                             window_len,
                                             window_len * window_len,
                                             qk_scale);
        }
        else if (std::is_same<T, half>::value) {
            softmax_withRelPosBias_element4_kernel<half4, half>
                <<<grid, block, 0, stream>>>((half4*)qk_buf,
                                             (const half4*)attn_mask,
                                             (const half4*)relative_pos_bias,
                                             batch_size,
                                             num_head,
                                             window_num,
                                             window_len,
                                             window_len * window_len,
                                             qk_scale);
        }
#ifdef ENABLE_BF16
        else {
            dim3 block((window_len + 31) / 32 * 32);
            softmax_withRelPosBias_element1_kernel<<<grid, block, 0, stream>>>(qk_buf,
                                                                               attn_mask,
                                                                               relative_pos_bias,
                                                                               batch_size,
                                                                               num_head,
                                                                               window_num,
                                                                               window_len,
                                                                               window_len * window_len,
                                                                               qk_scale);
        }
#endif
    }
    else if (window_len % 2 == 0) {
        dim3 block((window_len / 2 + 31) / 32 * 32);
        if (std::is_same<T, float>::value) {
            softmax_withRelPosBias_element2_kernel<float2, float>
                <<<grid, block, 0, stream>>>((float2*)qk_buf,
                                             (const float2*)attn_mask,
                                             (const float2*)relative_pos_bias,
                                             batch_size,
                                             num_head,
                                             window_num,
                                             window_len,
                                             window_len * window_len,
                                             qk_scale);
        }
        else if (std::is_same<T, half>::value) {
            softmax_withRelPosBias_element2_kernel<half2, half>
                <<<grid, block, 0, stream>>>((half2*)qk_buf,
                                             (const half2*)attn_mask,
                                             (const half2*)relative_pos_bias,
                                             batch_size,
                                             num_head,
                                             window_num,
                                             window_len,
                                             window_len * window_len,
                                             qk_scale);
        }
#ifdef ENABLE_BF16
        else {
            dim3 block((window_len + 31) / 32 * 32);
            softmax_withRelPosBias_element1_kernel<<<grid, block, 0, stream>>>(qk_buf,
                                                                               attn_mask,
                                                                               relative_pos_bias,
                                                                               batch_size,
                                                                               num_head,
                                                                               window_num,
                                                                               window_len,
                                                                               window_len * window_len,
                                                                               qk_scale);
        }
#endif
    }
    else {
        dim3 block((window_len + 31) / 32 * 32);
        softmax_withRelPosBias_element1_kernel<<<grid, block, 0, stream>>>(qk_buf,
                                                                           attn_mask,
                                                                           relative_pos_bias,
                                                                           batch_size,
                                                                           num_head,
                                                                           window_num,
                                                                           window_len,
                                                                           window_len * window_len,
                                                                           qk_scale);
    }
}

#define INSTANTIATEMASKEDSOFTMAXWITHRELPOSBIAS(T)                                                                      \
    template void invokeMaskedSoftMaxWithRelPosBias(T*           qk_buf,                                               \
                                                    const T*     attn_mask,                                            \
                                                    const T*     relative_pos_bias,                                    \
                                                    const int    batch_size,                                           \
                                                    const int    num_head,                                             \
                                                    const int    window_num,                                           \
                                                    const int    window_len,                                           \
                                                    const float  qk_scale,                                             \
                                                    cudaStream_t stream)
INSTANTIATEMASKEDSOFTMAXWITHRELPOSBIAS(float);
INSTANTIATEMASKEDSOFTMAXWITHRELPOSBIAS(half);
#ifdef ENABLE_BF16
INSTANTIATEMASKEDSOFTMAXWITHRELPOSBIAS(__nv_bfloat16);
#endif
#undef INSTANTIATEMASKEDSOFTMAXWITHRELPOSBIAS

template<typename T>
__global__ void transpose_attentions(
    T* attentions_out, const T* attentions_in, size_t batch_size, size_t num_layers, size_t num_heads, size_t seq_len)
{
    // attentions_in  shape [B, H, S, S]
    // attentions_out shape [B, L, H, S, S].
    // Note that we write the L dimension as if it was index 0.
    // In reality, the pointer has already been shifted to point to the correct layer.

    const auto batch_idx = blockIdx.x;
    const auto head_idx  = blockIdx.y;

    const auto dst_offset = (batch_idx * num_layers * num_heads + head_idx) * seq_len * seq_len;
    const auto src_offset = (batch_idx * num_heads + head_idx) * seq_len * seq_len;

    for (auto x = threadIdx.x; x < seq_len * seq_len; x += blockDim.x) {
        attentions_out[dst_offset + x] = attentions_in[src_offset + x];
    }
}

template<typename T>
void invokeTransposeAttentions(Tensor& attentions_out, const Tensor& attentions_in, cudaStream_t stream)
{
    const size_t batch_size = attentions_in.shape[0];
    const size_t num_heads  = attentions_in.shape[1];
    const size_t seq_len    = attentions_in.shape[2];
    const size_t num_layers = attentions_out.shape[1];

    const dim3 gridSize(batch_size, num_heads);
    const dim3 blockSize(512);

    transpose_attentions<<<gridSize, blockSize, 0, stream>>>(
        attentions_out.getPtr<T>(), attentions_in.getPtr<const T>(), batch_size, num_layers, num_heads, seq_len);
}

#define INSTANTIATETRANSPOSEATTENTIONS(T)                                                                              \
    template void invokeTransposeAttentions<T>(                                                                        \
        Tensor & attentions_out, const Tensor& attentions_in, cudaStream_t stream)
INSTANTIATETRANSPOSEATTENTIONS(float);
INSTANTIATETRANSPOSEATTENTIONS(half);
#ifdef ENABLE_BF16
INSTANTIATETRANSPOSEATTENTIONS(__nv_bfloat16);
#endif
#undef INSTANTIATETRANSPOSEATTENTIONS

}  // namespace turbomind
