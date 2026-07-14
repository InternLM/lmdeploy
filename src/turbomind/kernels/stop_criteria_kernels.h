/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint>

#include <cuda_runtime.h>

namespace turbomind {

void invokeStopWordsCriterion_v2(const int**  token_ids_ptrs,
                                 const int*   sequence_length,
                                 const int*   stop_words,
                                 bool*        finished,
                                 int          stop_words_len,
                                 int          batch_size,
                                 cudaStream_t stream);

void invokeLengthCriterion_v2(bool*        finished,  //
                              const int*   sequence_length,
                              const int*   sequence_length_limit,
                              int          batch_size,
                              cudaStream_t stream);

}  // namespace turbomind
