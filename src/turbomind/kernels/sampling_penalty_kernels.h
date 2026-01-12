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

#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/core/core.h"

namespace turbomind {

void ApplyRepetitionPenalty(Tensor&               logits,
                            const Buffer_<float>& penalties,
                            const Buffer_<int*>&  token_ids_ptrs,
                            const Buffer_<int>&   sequence_length,
                            cudaStream_t          stream);

template<typename T>
void invokeBatchApplyTemperaturePenalty_v2(T*           logits,
                                           const T*     bias,
                                           const float* temperatures,
                                           const int    batch_size,
                                           const int    vocab_size,
                                           const int    vocab_size_padd,
                                           cudaStream_t stream);

template<typename T>
void invokeMinLengthPenalty(T*           logits,
                            const int*   min_lengths,
                            const int*   sequnece_lengths,
                            const int    vocab_size_padded,
                            const int    batch_size,
                            const int*   end_ids,
                            const int    end_ids_size,
                            cudaStream_t stream);

}  // namespace turbomind
