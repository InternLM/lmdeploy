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
    cudaStreamSynchronize(stream);
    h_token_num[0] = h_pinned_token_num[0];
    sync_check_cuda_error();
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

}  // namespace turbomind
