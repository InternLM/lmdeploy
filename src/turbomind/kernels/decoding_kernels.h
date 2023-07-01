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

#include "gpt_kernels.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
void invokeDecodingInitialize(bool*        finished,
                              int*         sequence_length,
                              int*         word_ids,
                              T*           cum_log_probs,
                              const int*   sentence_ids,
                              const int    batch_size,
                              const int    beam_width,
                              const int    max_input_length,
                              cudaStream_t stream);

// get token from all_ids at step, then lookup from the embedding table
// by the token
template<typename T>
void invokeEmbeddingLookupPosEncodingPadCount(T*                    from_tensor,
                                              const T*              embedding_table,
                                              const T*              position_encoding,
                                              const int*            all_ids,
                                              const int*            padding_count,
                                              pPromptTuningParam<T> prompt_param,
                                              const int             local_token_num,
                                              const int             hidden_units,
                                              const T               scale,
                                              const int             step,
                                              const int             token_num,
                                              const int             ite,
                                              const int             seq_len,
                                              cudaStream_t          stream);

template<typename T>
void invokeEmbeddingLookupPosEncodingPadCount(T*           from_tensor,
                                              const T*     embedding_table,
                                              const T*     position_encoding,
                                              const int*   all_ids,
                                              const int*   padding_count,
                                              const int    local_token_num,
                                              const int    hidden_units,
                                              const T      scale,
                                              const int    step,
                                              const int    token_num,
                                              const int    ite,
                                              cudaStream_t stream)
{
    invokeEmbeddingLookupPosEncodingPadCount(from_tensor,
                                             embedding_table,
                                             position_encoding,
                                             all_ids,
                                             padding_count,
                                             {(const T**)nullptr, 0, 0, false, nullptr},
                                             local_token_num,
                                             hidden_units,
                                             scale,
                                             step,
                                             token_num,
                                             ite,
                                             0,
                                             stream);
}

template<typename T>
void invokePaddingEmbedding(T*           padded_embedding_kernel,
                            T*           padded_embedding_bias,
                            const T*     embedding_kernel,
                            const T*     embedding_bias,
                            const int    hidden_unit,
                            const int    vocab_size,
                            const int    vocab_size_padded,
                            cudaStream_t stream);

template<typename T>
void invokePaddingEmbeddingKernel(T*           padded_embedding_kernel,
                                  const T*     embedding_kernel,
                                  const int    hidden_unit,
                                  const int    vocab_size,
                                  const int    vocab_size_padded,
                                  cudaStream_t stream);

void invokeGatherTree(int*         beams,
                      int*         max_sequence_lengths,
                      const int    max_time,
                      const int    batch_size,
                      const int    beam_width,
                      const int*   step_ids,
                      const int*   parent_ids,
                      const int*   end_tokens,
                      cudaStream_t stream);

void invokeGatherTree(int*         beams,
                      int*         max_sequence_lengths,
                      const int    max_time,
                      const int    batch_size,
                      const int    beam_width,
                      const int*   step_ids,
                      const int*   parent_ids,
                      const int*   end_tokens,
                      const int    max_input_length,
                      cudaStream_t stream);

struct gatherTreeParam {
    int*       beams                          = nullptr;
    int*       max_sequence_lengths           = nullptr;
    int        max_sequence_length_final_step = 0;
    const int* input_lengths                  = nullptr;
    // response input lengths (used to slice the ids during postprocessing)
    int*       response_input_lengths     = nullptr;
    int        max_time                   = 0;
    int        batch_size                 = 0;
    int        beam_width                 = 0;
    const int* step_ids                   = nullptr;
    const int* parent_ids                 = nullptr;
    const int* end_tokens                 = nullptr;
    int        max_input_length           = 0;
    const int* prefix_soft_prompt_lengths = nullptr;
    // p_prompt_tuning prompt leangths, used to remove prompts during post-processing
    const int* p_prompt_tuning_prompt_lengths  = nullptr;
    int        max_input_without_prompt_length = 0;
    // prefix soft prompt
    int          max_prefix_soft_prompt_length = 0;
    int*         output_ids                    = nullptr;
    cudaStream_t stream;
};

void invokeGatherTree(gatherTreeParam param);

void invokeMinusUnfinishedSeqlen(int* sequence_lengths, const bool* finished, const int token_num, cudaStream_t stream);
void invokePlusUnfinishedSeqlen(int* sequence_lengths, const bool* finished, const int token_num, cudaStream_t stream);

template<typename T>
void invokePlusScalar(T* buf, const T val, const int size, cudaStream_t stream);

void invokeFinalize(int*         output_ids,
                    int*         sequence_lengths,
                    float*       cum_log_probs,
                    float*       output_log_probs,
                    const int*   topk_output_ids,
                    const int*   topk_sequence_lengths,
                    const float* scores,
                    const float* topk_cum_log_probs,
                    const float* topk_log_probs,
                    const int*   num_beams,
                    const int    beam_width,
                    const int    max_seq_len,
                    const int    batch_size,
                    cudaStream_t stream);

}  // namespace turbomind
