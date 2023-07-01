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

namespace turbomind {

template<typename T>
void invokeLogProbFromLogits(float*       cum_log_probs,
                             const T*     logits,
                             const int*   input_ids,
                             const int*   input_lengths,
                             const size_t max_input_length,
                             const size_t batch_size,
                             const size_t vocab_size,
                             const size_t vocab_size_padded,
                             void*        workspace,
                             const size_t workspace_size,
                             cudaStream_t stream,
                             const bool   batch_first = false);
}  // namespace turbomind
