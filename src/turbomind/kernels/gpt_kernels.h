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

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unordered_map>

#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<typename T>
struct inputIdsEmbeddingLookupPosEncodingSoftPromptParam {
    T*           from_tensor;
    int*         output_ids;
    int*         input_lengths;
    const T*     embedding_table;
    const T*     pos_table;
    const float* prefix_soft_prompt_embedding;
    const int*   prefix_soft_prompt_lengths;
    int*         input_ids;
    int          start_step;
    int          max_input_length;
    int          max_prefix_soft_prompt_length;
    int          batch_size;
    int          beam_width;
    int          hidden_units;
    cudaStream_t stream;
};

template<typename T>
struct pPromptTuningParam {
    // Batch number of ptrs, each ptr is the ptr of the specific p/prompt tuning weights for this sequence
    const T** p_prompt_tuning_batch_weights = nullptr;
    // The start id of p_prompt_tuning token ids (based on the tokenizer)
    // PROMPT_0 --> p_prompt_tuning_id_start; PROMPT_1 --> p_prompt_tuning_id_start + 1; ...
    const int p_prompt_tuning_id_start = 0;
    // Request prompt embeddding's max length
    const int request_prompt_max_length = 0;
    // Whether or not use the request prompt embeddings
    const bool use_request_p_prompt_embedding = false;
    // Request prompt embeddings
    const T* request_prompt_embedding = nullptr;
};

template<typename T>
void invokeInputIdsEmbeddingLookupPosEncoding(T*                    from_tensor,
                                              int*                  output_ids,
                                              const T*              embedding_table,
                                              const T*              pos_table,
                                              pPromptTuningParam<T> prompt_param,
                                              const int*            input_ids,
                                              const int             start_step,
                                              const int             length,
                                              const int             max_length,
                                              const int             batch_size,
                                              const int             hidden_units,
                                              cudaStream_t          stream);

template<typename T>
void invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(inputIdsEmbeddingLookupPosEncodingSoftPromptParam<T> param);

template<typename T>
void invokeTransposeAxis01(T* out, T* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template<typename T>
void invokeTransposeAxis01(
    T* out, T* in, const int* in_skipping_dim1, const int dim0, const int dim1, cudaStream_t stream);

template<typename T>
void invokeBuildDecoderAttentionMask(T*           attention_mask,
                                     const int*   sequence_lengths,
                                     const int*   prefix_prompt_lengths,
                                     const int    batch_size,
                                     const int    max_seq_len,
                                     const int    max_prompt_length,
                                     cudaStream_t stream);

template<typename T>
void invokeLookupHiddenStateOfLastToken(T*           from_tensor,
                                        const T*     hidden_state,
                                        const int*   input_lengths,
                                        const int    max_input_length,
                                        const int    batch_size,
                                        const int    hidden_units,
                                        cudaStream_t stream);

void invokeTileGptPromptInputs(int*         tiled_input_ids,
                               int*         tiled_input_lengths,
                               int*         tiled_prompt_lengths,
                               const int*   input_ids,
                               const int*   input_lengths,
                               const int*   prefix_prompt_lengths,
                               const int    batch_size,
                               const int    beam_width,
                               const int    max_input_length,
                               cudaStream_t stream);

void invokeTileGptInputs(int*         tiled_input_ids,
                         int*         tiled_input_lengths,
                         const int*   input_ids,
                         const int*   input_lengths,
                         const int    batch_size,
                         const int    beam_width,
                         const int    max_input_length,
                         cudaStream_t stream);

void invokeFindContextDups(int*         shared_contexts,
                           int*         batch_to_compact,
                           int*         compact_to_batch,
                           int*         compact_size,
                           const int*   input_ids,
                           const size_t batch_size,
                           const size_t input_seq_len,
                           cudaStream_t stream = 0);

template<typename T>
void handleOptArg(TensorMap* input_tensors, const std::string& arg_name, T* d_ptr, T default_value, size_t size)
{
    if (input_tensors->isExist(arg_name)) {
        FT_CHECK(input_tensors->at(arg_name).size() == size);
        cudaH2Dcpy(d_ptr, input_tensors->at(arg_name).getPtr<const T>(), size);
    }
    else {
        deviceFill(d_ptr, size, default_value);
    }
}

void setSeqLimitLen(uint32_t* seq_len_d, Tensor seq_len, int limit_len_offset, int batch_size);

template<typename T>
void invokeCompactInputs(T*           compact_input,
                         T*           compact_attention_mask,
                         int*         compact_input_lengths,
                         const T*     decoder_input,
                         const T*     decoder_mask,
                         const int*   input_lengths,
                         const int*   compact_idx,
                         size_t       compact_size,
                         size_t       seq_len,
                         size_t       hidden_dimension,
                         cudaStream_t stream = 0);

template<typename T>
void invokeUnCompactOutputs(T*           uncompact_buffer,
                            const T*     compact_buffer,
                            const int*   batch_to_compact_idx,
                            size_t       batch_size,
                            size_t       buffer_stride,
                            cudaStream_t stream = 0);

template<typename T>
void invokeUnCompactCaches(T*           uncompact_k_cache,
                           T*           uncompact_v_cache,
                           const T*     compact_k_cache,
                           const T*     compact_v_cache,
                           const int*   batch_to_compact_idx,
                           size_t       batch_size,
                           size_t       num_heads,
                           size_t       max_seq_len,
                           size_t       seq_len,
                           size_t       size_per_head,
                           size_t       local_batch_size,
                           size_t       ite,
                           cudaStream_t stream = 0);

void invokeUpdatePaddingCount(int*         total_padding_count,
                              const int*   input_lengths,
                              const int*   tiled_prompt_lengths,
                              size_t       max_input_length,
                              size_t       max_prompt_length,
                              size_t       batch_size,
                              size_t       beam_width,
                              cudaStream_t stream = 0);

inline void invokeUpdatePaddingCount(int*         total_padding_count,
                                     const int*   input_lengths,
                                     size_t       max_input_length,
                                     size_t       batch_size,
                                     size_t       beam_width,
                                     cudaStream_t stream = 0)
{
    invokeUpdatePaddingCount(
        total_padding_count, input_lengths, (const int*)nullptr, max_input_length, 0, batch_size, beam_width, stream);
}

void invokeMaskPaddingTokens(bool*        masked_tokens,
                             const int*   input_lengths,
                             const int*   tiled_prefix_prompt_lengths,
                             const size_t memory_len,
                             const size_t max_input_length,
                             const size_t initial_step,
                             size_t       batch_size,
                             size_t       beam_width,
                             cudaStream_t stream = 0);

inline void invokeMaskPaddingTokens(bool*        masked_tokens,
                                    const int*   input_lengths,
                                    const size_t memory_len,
                                    const size_t max_input_length,
                                    const size_t initial_step,
                                    size_t       batch_size,
                                    size_t       beam_width,
                                    cudaStream_t stream = 0)
{
    invokeMaskPaddingTokens(masked_tokens,
                            input_lengths,
                            (const int*)nullptr,
                            memory_len,
                            max_input_length,
                            initial_step,
                            batch_size,
                            beam_width,
                            stream);
}

template<typename T>
void invokeSumLengthDimension(float*       out_buf,
                              const T*     in_buf,
                              const size_t batch_size,
                              const size_t input_length,
                              const size_t hidden_dim,
                              cudaStream_t stream = 0);

}  // namespace turbomind
