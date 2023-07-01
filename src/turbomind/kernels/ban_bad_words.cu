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

#include "src/turbomind/kernels/ban_bad_words.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T>
__global__ void ban_bad_words(T*         logits,
                              const int* output_ids_buf,
                              const int* parent_ids_buf,
                              int        batch_size,
                              int        beam_width,
                              const int* bad_words,
                              size_t     bad_words_len,
                              bool       share_words,
                              int        id_offset,
                              int        vocab_size_padded,
                              size_t     step)
{
    const int id        = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y / beam_width;
    const int beam_idx  = blockIdx.y % beam_width;

    const int* base_bad_words         = share_words ? bad_words : bad_words + batch_idx * 2 * bad_words_len;
    const int* base_bad_words_offsets = base_bad_words + bad_words_len;

    if (id >= bad_words_len || base_bad_words_offsets[id] < 0) {
        return;
    }

    const int item_end   = base_bad_words_offsets[id];
    const int item_start = (id > 0) ? base_bad_words_offsets[id - 1] : 0;
    const int item_size  = item_end - item_start;

    /* The single-token case unconditionally bans the token */
    bool should_ban = item_size == 1;

    /* Multi-token case and enough previously generated tokens to look for a match */
    if (item_size > 1 && step >= item_size - 1) {
        should_ban             = true;
        int        parent_id   = beam_idx;
        const bool gather_beam = beam_width > 1;

        for (int token_idx = item_size - 2; token_idx >= 0; token_idx--) {
            const int previous_token = output_ids_buf[(step - (item_size - 1) + token_idx) * batch_size * beam_width
                                                      + id_offset + batch_idx * beam_width + parent_id];

            if (previous_token != base_bad_words[item_start + token_idx]) {
                should_ban = false;
                break;
            }
            if (gather_beam) {
                parent_id = parent_ids_buf[(step - (item_size - 1) + token_idx) * beam_width * batch_size + id_offset
                                           + batch_idx * beam_width + parent_id];

                if (parent_id < 0 || parent_id >= beam_width) {
                    should_ban = false;
                    break;
                }
            }
        }
    }

    if (should_ban) {
        int banned_token = base_bad_words[item_end - 1];
        if (0 < banned_token && banned_token < vocab_size_padded) {
            logits[batch_idx * beam_width * vocab_size_padded + beam_idx * vocab_size_padded + banned_token] =
                static_cast<T>(-INFINITY);
        }
    }
}

template<typename T>
void invokeBanBadWords(T*           logits,
                       const int*   output_ids_buf,
                       const int*   parent_ids_buf,
                       int          batch_size,
                       int          local_batch_size,
                       int          beam_width,
                       const int*   bad_words,
                       bool         share_words,
                       size_t       bad_words_len,
                       int          id_offset,
                       int          vocab_size_padded,
                       size_t       step,
                       cudaStream_t stream)
{
    dim3 block, grid;
    block.x = min(((bad_words_len + 32 - 1) / 32) * 32, 256UL);
    grid.x  = (bad_words_len + block.x - 1) / block.x;
    grid.y  = local_batch_size * beam_width;

    ban_bad_words<<<grid, block, 0, stream>>>(logits,
                                              output_ids_buf,
                                              parent_ids_buf,
                                              batch_size,
                                              beam_width,
                                              bad_words,
                                              bad_words_len,
                                              share_words,
                                              id_offset,
                                              vocab_size_padded,
                                              step);
    sync_check_cuda_error();
}

template void invokeBanBadWords(half*        logits,
                                const int*   output_ids_buf,
                                const int*   parent_ids_buf,
                                int          batch_size,
                                int          local_batch_size,
                                int          beam_width,
                                const int*   bad_words,
                                bool         share_words,
                                size_t       bad_words_len,
                                int          id_offset,
                                int          vocab_size_padded,
                                size_t       step,
                                cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBanBadWords(__nv_bfloat16* logits,
                                const int*     output_ids_buf,
                                const int*     parent_ids_buf,
                                int            batch_size,
                                int            local_batch_size,
                                int            beam_width,
                                const int*     bad_words,
                                bool           share_words,
                                size_t         bad_words_len,
                                int            id_offset,
                                int            vocab_size_padded,
                                size_t         step,
                                cudaStream_t   stream);
#endif
template void invokeBanBadWords(float*       logits,
                                const int*   output_ids_buf,
                                const int*   parent_ids_buf,
                                int          batch_size,
                                int          local_batch_size,
                                int          beam_width,
                                const int*   bad_words,
                                bool         share_words,
                                size_t       bad_words_len,
                                int          id_offset,
                                int          vocab_size_padded,
                                size_t       step,
                                cudaStream_t stream);

}  // namespace turbomind
