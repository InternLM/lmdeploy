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

template<bool kSpeculative>
__global__ void stop_criteria(const int* const* token_ids_ptrs,
                              const int*        sequence_length,
                              int*              accept_len,
                              const int*        stop_words,
                              int               stop_words_width,
                              const int*        sequence_length_limit,
                              bool*             finished,
                              int               batch_size)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size) {
        return;
    }

    int entry_len = 0;
    int span_len  = 0;

    if constexpr (kSpeculative) {
        if (finished[b]) {
            accept_len[b] = 0;
            return;
        }

        entry_len = sequence_length[b];
        span_len  = accept_len[b];
    }
    else {
        if (finished[b]) {
            return;
        }

        const int current_len = sequence_length[b];

        if (current_len <= 0) {
            return;
        }

        entry_len = current_len - 1;
        span_len  = 1;
    }

    if (span_len <= 0) {
        return;
    }

    const int* tokens = token_ids_ptrs[b];

    for (int j = 0; j < span_len; ++j) {
        const int effective_len = entry_len + j + 1;

        bool terminal = effective_len >= sequence_length_limit[b];

        if (!terminal && stop_words != nullptr) {
            const int* words   = stop_words + b * 2 * stop_words_width;
            const int* offsets = words + stop_words_width;

            for (int phrase = 0; phrase < stop_words_width; ++phrase) {
                const int phrase_end = offsets[phrase];

                if (phrase_end < 0) {
                    break;
                }

                const int phrase_begin = phrase == 0 ? 0 : offsets[phrase - 1];
                const int phrase_size  = phrase_end - phrase_begin;

                if (phrase_size <= 0 || effective_len < phrase_size) {
                    continue;
                }

                const int history_begin = effective_len - phrase_size;
                bool      match         = true;

                for (int t = 0; t < phrase_size; ++t) {
                    if (tokens[history_begin + t] != words[phrase_begin + t]) {
                        match = false;
                        break;
                    }
                }

                if (match) {
                    terminal = true;
                    break;
                }
            }
        }

        if (terminal) {
            if constexpr (kSpeculative) {
                accept_len[b] = j + 1;
            }

            finished[b] = true;
            return;
        }
    }
}

void invokeStopCriteria(const int* const* token_ids_ptrs,
                        const int*        sequence_length,
                        const int*        stop_words,
                        int               stop_words_width,
                        const int*        sequence_length_limit,
                        bool*             finished,
                        int               batch_size,
                        cudaStream_t      stream)
{
    if (batch_size == 0) {
        return;
    }

    constexpr int block_size = 128;
    stop_criteria<false><<<cdiv(batch_size, block_size), block_size, 0, stream>>>(token_ids_ptrs,
                                                                                  sequence_length,
                                                                                  nullptr,
                                                                                  stop_words,
                                                                                  stop_words_width,
                                                                                  sequence_length_limit,
                                                                                  finished,
                                                                                  batch_size);
}

void invokeStopCriteria(const int* const* token_ids_ptrs,
                        const int*        entry_sequence_length,
                        int*              accept_len,
                        const int*        stop_words,
                        int               stop_words_width,
                        const int*        sequence_length_limit,
                        bool*             finished,
                        int               batch_size,
                        cudaStream_t      stream)
{
    if (batch_size == 0) {
        return;
    }

    constexpr int block_size = 128;
    stop_criteria<true><<<cdiv(batch_size, block_size), block_size, 0, stream>>>(token_ids_ptrs,
                                                                                 entry_sequence_length,
                                                                                 accept_len,
                                                                                 stop_words,
                                                                                 stop_words_width,
                                                                                 sequence_length_limit,
                                                                                 finished,
                                                                                 batch_size);
}

}  // namespace turbomind
