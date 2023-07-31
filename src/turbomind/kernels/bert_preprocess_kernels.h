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
#include "src/turbomind/utils/cuda_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef ENABLE_FP8
#include "src/turbomind/utils/cuda_fp8_utils.h"
#endif  // ENABLE_FP8

namespace turbomind {

void invokeGetPaddingOffsetAndCuSeqLens(size_t*      h_pinned_token_num,
                                        size_t*      h_token_num,
                                        int*         tmp_mask_offset,
                                        int*         cu_seqlens,
                                        const int*   sequence_length,
                                        const int    batch_size,
                                        const int    max_seq_len,
                                        cudaStream_t stream);

inline void invokeGetPaddingOffset(size_t*      h_pinned_token_num,
                                   size_t*      h_token_num,
                                   int*         tmp_mask_offset,
                                   const int*   sequence_length,
                                   const int    batch_size,
                                   const int    max_seq_len,
                                   cudaStream_t stream)
{
    invokeGetPaddingOffsetAndCuSeqLens(
        h_pinned_token_num, h_token_num, tmp_mask_offset, nullptr, sequence_length, batch_size, max_seq_len, stream);
}

template<typename T>
void invokeRebuildPadding(
    T* dst, const T* src, const int* padding_offset, const int token_num, const int hidden_dim, cudaStream_t stream);

template<typename T>
void invokeRemovePadding(
    T* dst, const T* src, const int* padding_offset, const int token_num, const int hidden_dim, cudaStream_t stream);

}  // namespace turbomind
