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

#include "src/turbomind/kernels/penalty_types.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T>
void invokeAddBiasApplyPenalties(int                         step,
                                 T*                          logits,
                                 const int*                  current_ids,
                                 const int*                  previous_ids,
                                 const int*                  parent_ids,
                                 const int*                  input_lengths,
                                 const int*                  sequence_lengths,
                                 const T*                    bias,
                                 const int                   ite,
                                 const int                   max_input_length,
                                 const int                   local_batch_size,
                                 const int                   batch_size,
                                 const int                   beam_width,
                                 const int                   vocab_size,
                                 const int                   vocab_size_padded,
                                 const int*                  end_ids,
                                 const float                 temperature,
                                 const float                 repetition_penalty,
                                 const RepetitionPenaltyType repetition_penalty_type,
                                 const int                   min_length,
                                 cudaStream_t                stream);

}  // namespace turbomind
