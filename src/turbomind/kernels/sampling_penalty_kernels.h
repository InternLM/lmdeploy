/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/kernels/penalty_types.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

template<typename T>
void invokeApplyRepetitionPenalty(T*                          logits,
                                  const float                 penalty,
                                  const int*                  start_ids,
                                  int*                        output_ids,
                                  const int                   batch_size,
                                  const int                   local_batch_size,
                                  const int                   vocab_size,
                                  const int                   vocab_size_padd,
                                  const int*                  input_lengths,
                                  const int                   max_input_len,
                                  const int                   step,
                                  const RepetitionPenaltyType penalty_type,
                                  cudaStream_t                stream);

template<typename T>
void invokeBatchApplyRepetitionPenalty(T*                          logits,
                                       const float*                penalties,
                                       const int*                  output_ids,
                                       const int                   batch_size,
                                       const int                   local_batch_size,
                                       const int                   vocab_size,
                                       const int*                  input_lengths,
                                       const int                   max_input_length,
                                       const int                   step,
                                       const RepetitionPenaltyType penalty_type,
                                       cudaStream_t                stream);

template<typename T>
void invokeApplyTemperaturePenalty(T*           logits,
                                   const T*     bias,
                                   const float  temperature,
                                   const int    batch_size,
                                   const int    vocab_size,
                                   const int    vocab_size_padd,
                                   cudaStream_t stream);

template<typename T>
void invokeBatchApplyTemperaturePenalty(T*           logits,
                                        const T*     bias,
                                        const float* temperatures,
                                        const int    batch_size,
                                        const int    vocab_size,
                                        const int    vocab_size_padd,
                                        cudaStream_t stream);

template<typename T>
void invokeMinLengthPenalty(T*           logits,
                            const int*   min_lengths,
                            const int*   end_ids,
                            const int*   sequnece_lengths,
                            const int    max_input_length,
                            const int    batch_size,
                            const int    vocab_size_padded,
                            cudaStream_t stream);

}  // namespace fastertransformer
