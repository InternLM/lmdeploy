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

#include "bert_preprocess_kernels.h"
#include "src/turbomind/utils/cuda_bf16_fallbacks.cuh"
#include "src/turbomind/utils/cuda_fp8_utils.h"
#include "src/turbomind/utils/cuda_type_utils.cuh"

namespace turbomind {

__global__ void getPaddingOffsetAndCuSeqLensKernel(size_t*    h_valid_word_num,
                                                   int*       tmp_mask_offset,
                                                   int*       cu_seqlens,
                                                   const int* sequence_length,
                                                   const int  batch_size,
                                                   const int  max_seq_len)
{
    // do cumulated sum
    int        total_seq_len        = 0;
    int        cum_offset           = 0;
    int        index                = 0;
    const bool calculate_cu_seqlens = cu_seqlens != nullptr;
    for (int i = 0; i < batch_size; i++) {
        const int seq_len = sequence_length[i];
        if (calculate_cu_seqlens) {
            cu_seqlens[i] = total_seq_len;
        }
        for (int j = 0; j < seq_len; j++) {
            tmp_mask_offset[index] = cum_offset;
            index++;
        }
        cum_offset += max_seq_len - seq_len;
        total_seq_len += seq_len;
    }
    if (calculate_cu_seqlens) {
        cu_seqlens[batch_size] = total_seq_len;
    }
    h_valid_word_num[0] = (size_t)total_seq_len;
}

void invokeGetPaddingOffsetAndCuSeqLens(size_t*      h_pinned_token_num,
                                        size_t*      h_token_num,
                                        int*         tmp_mask_offset,
                                        int*         cu_seqlens,
                                        const int*   sequence_lengths,
                                        const int    batch_size,
                                        const int    max_seq_len,
                                        cudaStream_t stream)
{
    h_pinned_token_num[0] = 0;
    getPaddingOffsetAndCuSeqLensKernel<<<1, 1, 0, stream>>>(
        h_pinned_token_num, tmp_mask_offset, cu_seqlens, sequence_lengths, batch_size, max_seq_len);
    while (((volatile size_t*)h_pinned_token_num)[0] == 0) {};
    h_token_num[0] = h_pinned_token_num[0];
    sync_check_cuda_error();
}

template<typename T>
__global__ void buildEncoderAttentionMaskKernel(T* attention_mask, const int* sequence_lengths, const int max_seq_len)
{
    // sequence_lengths: [batch_size]
    // attention_mask: [batch_size, 1, max_seq_len, max_seq_len]
    attention_mask += blockIdx.x * max_seq_len * max_seq_len;
    const int length = sequence_lengths[blockIdx.x];
    for (int i = threadIdx.x; i < max_seq_len * max_seq_len; i += blockDim.x) {
        // int row_id = i / max_seq_len;
        int col_id = i % max_seq_len;
        // if (row_id < length && col_id < length) {
        // TODO (bhsueh) check this modification is ok or not on other rmodel
        if (col_id < length) {
            attention_mask[i] = (T)(1.0f);
        }
        else {
            attention_mask[i] = (T)(0.0f);
        }
    }
}

template<typename T>
void invokeBuildEncoderAttentionMask(
    T* attention_mask, const int* sequence_lengths, const int batch_size, const int max_seq_len, cudaStream_t stream)
{
    buildEncoderAttentionMaskKernel<<<batch_size, 256, 0, stream>>>(attention_mask, sequence_lengths, max_seq_len);
}

template void invokeBuildEncoderAttentionMask(float*       attention_mask,
                                              const int*   sequence_lengths,
                                              const int    batch_size,
                                              const int    max_seq_len,
                                              cudaStream_t stream);
template void invokeBuildEncoderAttentionMask(half*        attention_mask,
                                              const int*   sequence_lengths,
                                              const int    batch_size,
                                              const int    max_seq_len,
                                              cudaStream_t stream);
#ifdef ENABLE_FP8
template void invokeBuildEncoderAttentionMask(__nv_fp8_e4m3* attention_mask,
                                              const int*     sequence_lengths,
                                              const int      batch_size,
                                              const int      max_seq_len,
                                              cudaStream_t   stream);
#endif  // ENABLE_FP8
#ifdef ENABLE_BF16
template void invokeBuildEncoderAttentionMask(__nv_bfloat16* attention_mask,
                                              const int*     sequence_lengths,
                                              const int      batch_size,
                                              const int      max_seq_len,
                                              cudaStream_t   stream);
#endif

__global__ void getTrtPaddingOffsetKernel(int* trt_mha_padding_offset, const int* sequence_length, const int batch_size)
{
    // use for get tensorrt fused mha padding offset
    // when we remove the padding

    extern __shared__ int tmp_offset[];
    if (threadIdx.x == 0) {
        tmp_offset[0] = 0;
        for (int i = 0; i < batch_size; i++) {
            tmp_offset[i + 1] = tmp_offset[i] + sequence_length[i];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < batch_size + 1; i += blockDim.x) {
        trt_mha_padding_offset[i] = tmp_offset[i];
    }
}

void invokeGetTrtPaddingOffset(int*         trt_mha_padding_offset,
                               const int*   sequence_length,
                               const int    batch_size,
                               cudaStream_t stream)
{
    getTrtPaddingOffsetKernel<<<1, 256, sizeof(int) * (batch_size + 1), stream>>>(
        trt_mha_padding_offset, sequence_length, batch_size);
}

__global__ void getTrtPaddingOffsetKernel(int*       trt_mha_padding_offset,
                                          const int* sequence_length,
                                          const int  request_batch_size,
                                          const int  request_seq_len)
{
    // use for get tensorrt fused mha padding offset
    // when we keep the padding

    extern __shared__ int tmp_offset[];
    if (threadIdx.x == 0) {
        tmp_offset[0] = 0;
        for (int i = 0; i < request_batch_size; i++) {
            tmp_offset[i * 2 + 1] = tmp_offset[i * 2] + sequence_length[i];
            tmp_offset[i * 2 + 2] = request_seq_len * (i + 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 2 * request_batch_size + 1; i += blockDim.x) {
        trt_mha_padding_offset[i] = tmp_offset[i];
    }
}

void invokeGetTrtPaddingOffset(int*         trt_mha_padding_offset,
                               const int*   sequence_length,
                               const int    request_batch_size,
                               const int    request_seq_len,
                               cudaStream_t stream)
{
    getTrtPaddingOffsetKernel<<<1, 256, sizeof(int) * (2 * request_batch_size + 1), stream>>>(
        trt_mha_padding_offset, sequence_length, request_batch_size, request_seq_len);
}

template<typename T>
__global__ void rebuild_sequence_length_padding(const T* src, T* dst, const int* padding_offset, const int n)
{
    const int tid        = threadIdx.x;
    const int bid        = blockIdx.x;
    const int dst_seq_id = bid + padding_offset[bid];
    const int src_seq_id = bid;

    for (int i = tid; i < n; i += blockDim.x) {
        dst[dst_seq_id * n + i] = src[src_seq_id * n + i];
    }
}

template<typename T>
void invokeRebuildPadding(
    T* dst, const T* src, const int* padding_offset, const int token_num, const int hidden_dim, cudaStream_t stream)
{
    // src: [token_num, hidden_dim]
    // dst: [batch_size*max_seq_len, hidden_dim]
    rebuild_sequence_length_padding<<<token_num, 256, 0, stream>>>(src, dst, padding_offset, hidden_dim);
}

template<typename T>
void invokeRebuildPadding(
    T* dst, const T* src, const int* padding_offset, const int token_num, const int hidden_dim, cudaStream_t stream);
template void invokeRebuildPadding(float*       dst,
                                   const float* src,
                                   const int*   padding_offset,
                                   const int    token_num,
                                   const int    hidden_dim,
                                   cudaStream_t stream);
template void invokeRebuildPadding(half*        dst,
                                   const half*  src,
                                   const int*   padding_offset,
                                   const int    token_num,
                                   const int    hidden_dim,
                                   cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeRebuildPadding(__nv_bfloat16*       dst,
                                   const __nv_bfloat16* src,
                                   const int*           padding_offset,
                                   const int            token_num,
                                   const int            hidden_dim,
                                   cudaStream_t         stream);
#endif  // ENABLE_BF16

#ifdef ENABLE_FP8
template void invokeRebuildPadding(__nv_fp8_e4m3*       dst,
                                   const __nv_fp8_e4m3* src,
                                   const int*           padding_offset,
                                   const int            token_num,
                                   const int            hidden_dim,
                                   cudaStream_t         stream);
#endif  // ENABLE_FP8

template<typename T>
__global__ void remove_padding(T* tgt, const T* src, const int* padding_offset, const int n)
{
    const int tid        = threadIdx.x;
    const int bid        = blockIdx.x;
    const int src_seq_id = bid + padding_offset[bid];
    const int tgt_seq_id = bid;

    for (int i = tid; i < n; i += blockDim.x) {
        tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
    }
}

template<typename T>
void invokeRemovePadding(
    T* dst, const T* src, const int* padding_offset, const int token_num, const int hidden_dim, cudaStream_t stream)
{
    remove_padding<<<token_num, 256, 0, stream>>>(dst, src, padding_offset, hidden_dim);
}

template void invokeRemovePadding(float*       dst,
                                  const float* src,
                                  const int*   padding_offset,
                                  const int    token_num,
                                  const int    hidden_dim,
                                  cudaStream_t stream);

template void invokeRemovePadding(half*        dst,
                                  const half*  src,
                                  const int*   padding_offset,
                                  const int    token_num,
                                  const int    hidden_dim,
                                  cudaStream_t stream);
#ifdef ENABLE_FP8
template void invokeRemovePadding(__nv_fp8_e4m3*       dst,
                                  const __nv_fp8_e4m3* src,
                                  const int*           padding_offset,
                                  const int            token_num,
                                  const int            hidden_dim,
                                  cudaStream_t         stream);
#endif  // ENABLE_FP8
#ifdef ENABLE_BF16
template void invokeRemovePadding(__nv_bfloat16*       dst,
                                  const __nv_bfloat16* src,
                                  const int*           padding_offset,
                                  const int            token_num,
                                  const int            hidden_dim,
                                  cudaStream_t         stream);
#endif

template<typename T>
__global__ void buildRelativeAttentionBias(T*         relative_attention_bias,
                                           const T*   relative_attention_bias_table,
                                           const int  head_num,
                                           const int  seq_len,
                                           const int  num_bucket,
                                           const bool is_bidirectional,
                                           const int  max_distance)
{

    const int head_id = blockIdx.x;
    for (int seq_id = threadIdx.x; seq_id < seq_len * seq_len; seq_id += blockDim.x) {
        int row_id = seq_id / seq_len;
        int col_id = seq_id % seq_len;

        int relative_position = col_id - row_id;

        int relative_buckets = 0;
        int tmp_num_bucket   = num_bucket;
        if (is_bidirectional) {
            tmp_num_bucket /= 2;
            if (relative_position > 0) {
                relative_buckets += tmp_num_bucket;
            }
            else {
                relative_position *= -1;
            }
        }
        else {
            relative_position = abs(relative_position);
        }

        int  max_exact = tmp_num_bucket / 2;
        bool is_small  = relative_position < max_exact;

        int relative_position_if_large =
            max_exact
            + (int)(logf(relative_position * 1.0f / max_exact) / logf((float)max_distance / max_exact)
                    * (tmp_num_bucket - max_exact));

        relative_position_if_large = min(relative_position_if_large, tmp_num_bucket - 1);

        relative_buckets += is_small ? relative_position : relative_position_if_large;

        relative_attention_bias[head_id * seq_len * seq_len + seq_id] =
            relative_attention_bias_table[head_id * num_bucket + relative_buckets];
    }
}

template<typename T>
void invokeBuildRelativeAttentionBias(T*                          relative_attention_bias,
                                      const T*                    relative_attention_bias_table,
                                      const int                   head_num,
                                      const int                   seq_len,
                                      const int                   num_bucket,
                                      const bool                  is_bidirectional,
                                      const int                   max_distance,
                                      const PositionEmbeddingType position_embedding_type,
                                      cudaStream_t                stream)
{
    if (position_embedding_type == PositionEmbeddingType::absolute) {
        return;
    }
    dim3 grid(head_num);
    dim3 block(256);
    buildRelativeAttentionBias<<<grid, block, 0, stream>>>(relative_attention_bias,
                                                           relative_attention_bias_table,
                                                           head_num,
                                                           seq_len,
                                                           num_bucket,
                                                           is_bidirectional,
                                                           max_distance);
}

template void invokeBuildRelativeAttentionBias(float*                      relative_attention_bias,
                                               const float*                relative_attention_bias_table,
                                               const int                   head_num,
                                               const int                   seq_len,
                                               const int                   num_bucket,
                                               const bool                  is_bidirectional,
                                               const int                   max_distance,
                                               const PositionEmbeddingType position_embedding_type,
                                               cudaStream_t                stream);

template void invokeBuildRelativeAttentionBias(half*                       relative_attention_bias,
                                               const half*                 relative_attention_bias_table,
                                               const int                   head_num,
                                               const int                   seq_len,
                                               const int                   num_bucket,
                                               const bool                  is_bidirectional,
                                               const int                   max_distance,
                                               const PositionEmbeddingType position_embedding_type,
                                               cudaStream_t                stream);

#ifdef ENABLE_BF16
template void invokeBuildRelativeAttentionBias(__nv_bfloat16*              relative_attention_bias,
                                               const __nv_bfloat16*        relative_attention_bias_table,
                                               const int                   head_num,
                                               const int                   seq_len,
                                               const int                   num_bucket,
                                               const bool                  is_bidirectional,
                                               const int                   max_distance,
                                               const PositionEmbeddingType position_embedding_type,
                                               cudaStream_t                stream);
#endif

#ifdef ENABLE_FP8

template<typename T_OUT, typename T_IN>
__global__ void getLastTokenDequantize(getLastTokenDequantizeParam<T_OUT, T_IN> param)
{
    param.output[blockIdx.x * param.d_model + threadIdx.x] =
        (T_OUT)((float)param.input[blockIdx.x * param.max_seq_len * param.d_model + threadIdx.x]
                * __ldg(param.input_scale));
}

template<typename T_OUT, typename T_IN>
void invokeGetLastTokenDequantize(getLastTokenDequantizeParam<T_OUT, T_IN> param)
{
    FT_CHECK(param.d_model <= 1024);
    getLastTokenDequantize<T_OUT, T_IN><<<param.batch_size, param.d_model, 0, param.stream>>>(param);
}

template void invokeGetLastTokenDequantize<__nv_bfloat16, __nv_fp8_e4m3>(
    getLastTokenDequantizeParam<__nv_bfloat16, __nv_fp8_e4m3> param);

template<typename T_OUT, typename T_IN, QUANTIZE_MODE quantize_mode>
__global__ void quantizeMatrixRebuildPadding(QuantizeMatrixRebuildPaddingParam<T_OUT, T_IN, quantize_mode> param)
{
    for (int i = threadIdx.x; i < param.d_model; i += blockDim.x) {
        int padded_row_id = blockIdx.x + (param.padding_offset == nullptr ? 0 : param.padding_offset[blockIdx.x]);
        if (quantize_mode == QUANTIZE_MODE::PER_TENSOR) {
            param.dst[padded_row_id * param.d_model + i] =
                (T_OUT)((float)param.src[blockIdx.x * param.d_model + i] * __ldg(param.scale));
        }
        else if (quantize_mode == QUANTIZE_MODE::PER_CHANNEL) {
            param.dst[padded_row_id * param.d_model + i] =
                (T_OUT)((float)param.src[blockIdx.x * param.d_model + i] * __ldg(param.scale + i));
        }
    }
}

template<>
__global__ void
quantizeMatrixRebuildPadding(QuantizeMatrixRebuildPaddingParam<half, __nv_fp8_e4m3, QUANTIZE_MODE::PER_TENSOR> param)
{
    int padded_row_id = blockIdx.x + (param.padding_offset == nullptr ? 0 : __ldg(&param.padding_offset[blockIdx.x]));
    __nv_fp8x4_e4m3* src_ptr = ((__nv_fp8x4_e4m3*)param.src) + blockIdx.x * (param.d_model / 4);
    half2*           dst_ptr = ((half2*)param.dst) + padded_row_id * (param.d_model / 2);
    half2            scale   = cuda_cast<half2>(__ldg(param.scale));
    for (int i = threadIdx.x; i < param.d_model / 4; i += blockDim.x) {
        half2 val_0;
        half2 val_1;
        fp8x4_e4m3_to_half2(&val_0, &val_1, src_ptr + i);

        val_0 = hmul2(val_0, scale);
        val_1 = hmul2(val_1, scale);

        dst_ptr[2 * i + 0] = val_0;
        dst_ptr[2 * i + 1] = val_1;
    }
}

template<typename T_OUT, typename T_IN, QUANTIZE_MODE quantize_mode>
void invokeQuantizeMatrixRebuildPadding(QuantizeMatrixRebuildPaddingParam<T_OUT, T_IN, quantize_mode> param)
{
    dim3 grid(param.token_num);
    dim3 block(param.d_model);
    FT_CHECK(block.x <= 1024);
    if (block.x % 4 == 0) {
        block.x /= 4;
    }
    quantizeMatrixRebuildPadding<<<grid, block, 0, param.stream>>>(param);
}

template void invokeQuantizeMatrixRebuildPadding<half, __nv_fp8_e4m3, QUANTIZE_MODE::PER_TENSOR>(
    QuantizeMatrixRebuildPaddingParam<half, __nv_fp8_e4m3, QUANTIZE_MODE::PER_TENSOR> param);

#endif

}  // namespace turbomind
