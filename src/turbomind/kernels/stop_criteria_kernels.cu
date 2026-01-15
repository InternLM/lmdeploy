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

#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/stop_criteria_kernels.h"

namespace turbomind {

__global__ void stop_words_criterion_v2(const int** token_ids_ptrs,
                                        const int*  sequence_length,
                                        const int*  stop_words,
                                        bool*       finished,
                                        int         stop_words_len,
                                        int         batch_size)
{
    const int id        = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;

    const int* base_stop_words = stop_words + batch_idx * 2 * stop_words_len;
    const int* base_offsets    = base_stop_words + stop_words_len;

    if (id >= stop_words_len || base_offsets[id] < 0) {
        return;
    }

    const int item_end   = base_offsets[id];
    const int item_start = (id > 0) ? base_offsets[id - 1] : 0;
    const int item_size  = item_end - item_start;

    const int  seq_len   = sequence_length[batch_idx];
    const int* token_ids = token_ids_ptrs[batch_idx];

    /* Enough previously generated tokens to look for a match */
    if (seq_len >= item_size) {
        // token_ids[seq_len - 1] is the last token
        for (int token_idx = item_size - 1, offset = seq_len - 1; token_idx >= 0; token_idx--, offset--) {
            if (token_ids[offset] != base_stop_words[item_start + token_idx]) {
                return;
            }
        }
        finished[batch_idx] = true;
    }
}

void invokeStopWordsCriterion_v2(const int**  token_ids_ptrs,
                                 const int*   sequence_length,
                                 const int*   stop_words,
                                 bool*        finished,
                                 int          stop_words_len,
                                 int          batch_size,
                                 cudaStream_t stream)
{
    // Check if we have sampled a word from the stop_words list. If so, stop the sequence.

    const int  block = std::min(round_up(stop_words_len, 32), 256);
    const dim3 grid(cdiv(stop_words_len, block), batch_size);

    stop_words_criterion_v2<<<grid, block, 0, stream>>>(
        token_ids_ptrs, sequence_length, stop_words, finished, stop_words_len, batch_size);
}

__global__ void length_criterion_v2(bool*      finished,  //
                                    const int* sequence_length,
                                    const int* sequence_length_limit,
                                    int        batch_size)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= batch_size) {
        return;
    }
    if (sequence_length[idx] >= sequence_length_limit[idx]) {
        finished[idx] = true;
    }
}

void invokeLengthCriterion_v2(bool*        finished,  //
                              const int*   sequence_length,
                              const int*   sequence_length_limit,
                              int          batch_size,
                              cudaStream_t stream)
{
    // Check if we have attained the sequence length limit. If so, stop the sequence.

    constexpr int block = 256;
    const int     grid  = cdiv(batch_size, block);

    length_criterion_v2<<<grid, block, 0, stream>>>(finished, sequence_length, sequence_length_limit, batch_size);
}

}  // namespace turbomind
