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
#include <cfloat>
// #include "src/turbomind/kernels/reduce_kernel_utils.cuh"
// #include "src/turbomind/utils/cuda_utils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace turbomind {

template<typename T>
__device__ inline T getMaxValue();

template<>
__device__ inline float getMaxValue<float>()
{
    return FLT_MAX;
}

template<>
__device__ inline half getMaxValue<half>()
{
    return __ushort_as_half((unsigned short)0x7BFFU);
}

#ifdef ENABLE_BF16
template<>
__device__ inline __nv_bfloat16 getMaxValue<__nv_bfloat16>()
{
#if __CUDA_ARCH__ >= 800
    return __ushort_as_bfloat16((unsigned short)0x7F7FU);
#endif
    return {};
}
#endif

template<class T>
__global__ void BanBadWordsKernel(T*                logits,
                                  const int* const* token_ids_ptrs,
                                  const int*        sequence_length,
                                  const int*        bad_words,
                                  size_t            bad_words_len,
                                  int               vocab_size)
{
    const int id        = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;

    const int* base_bad_words         = bad_words + batch_idx * 2 * bad_words_len;
    const int* base_bad_words_offsets = base_bad_words + bad_words_len;

    if (id >= bad_words_len || base_bad_words_offsets[id] < 0) {
        return;
    }

    const int item_end   = base_bad_words_offsets[id];
    const int item_start = (id > 0) ? base_bad_words_offsets[id - 1] : 0;
    const int item_size  = item_end - item_start;

    const int  seq_len   = sequence_length[batch_idx];
    const int* token_ids = token_ids_ptrs[batch_idx];

    /* The single-token case unconditionally bans the token */
    bool should_ban = item_size == 1;

    /* Multi-token case and enough previously generated tokens to look for a match */
    if (item_size > 1 && seq_len >= item_size - 1) {
        should_ban = true;
        for (int token_idx = item_size - 2, offset = seq_len - 1; token_idx >= 0; token_idx--, offset--) {
            if (token_ids[offset] != base_bad_words[item_start + token_idx]) {
                should_ban = false;
                break;
            }
        }
    }

    logits += batch_idx * (int64_t)vocab_size;
    if (should_ban) {
        int banned_token = base_bad_words[item_end - 1];
        if (0 < banned_token && banned_token < vocab_size) {
            logits[banned_token] = -getMaxValue<T>();
        }
    }
}

void BanBadWords(Tensor&             logits,
                 const Buffer_<int*> token_ids_ptrs,
                 const Buffer_<int>& sequence_length,
                 const Tensor_<int>& bad_words,
                 cudaStream_t        stream)
{

    auto invoke = [&](auto dtype) {
        using T = decltype(dtype);

        const auto [bsz, vocab_size] = logits.shapes(0, 1);
        const int bad_words_len      = bad_words.shape(2);

        const int  block = std::min(round_up(bad_words_len, WARP_SIZE), 256);
        const dim3 grid(cdiv(bad_words_len, block), bsz);

        BanBadWordsKernel<<<grid, block, 0, stream>>>(logits.data<T>(),
                                                      token_ids_ptrs.data(),
                                                      sequence_length.data(),
                                                      bad_words.data(),
                                                      bad_words_len,
                                                      vocab_size);
    };

    invoke(float{});
}

}  // namespace turbomind
