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

template<typename T>
void invokePlusScalar(T* buf, const T val, const int size, cudaStream_t stream);

}  // namespace turbomind
