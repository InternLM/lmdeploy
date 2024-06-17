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

#include "src/turbomind/utils/cuda_fp8_utils.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11000)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

// PROMPT_SRC: 0 --> no prompts, 1 --> from loaded prompts, 2 --> from request prompts
template<typename T, bool OUTPUT_ID, int PROMPT_SRC>
__global__ void start_id_embedding_position_lookups_kernel(T*                    from_tensor,
                                                           int*                  output_ids,
                                                           const T*              embedding_table,
                                                           const T*              pos_table,
                                                           pPromptTuningParam<T> prompt_param,
                                                           const int*            input_ids,
                                                           const int             start_step,
                                                           const int             length,
                                                           const int             max_length,
                                                           const int             batch_size,
                                                           const int64_t         hidden_units)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * length * hidden_units;
         index += blockDim.x * gridDim.x) {
        // transpose the input_ids [batch, length] (part of [batch, max_length]) to output_ids [length, batch]
        if (OUTPUT_ID && index < batch_size * max_length) {
            // for p/prompt_tuning (have prompt templates like [input1, prompt1, input2, prompt2])
            // we have to process it to like [input1, input2, prompt1, prompt2], and then remove the prompts during post
            // processing
            if (PROMPT_SRC > 0) {
                if (index < batch_size) {
                    int no_prompt_output_seq_id = 0;
#pragma unroll 1
                    for (int seq_id = 0; seq_id < max_length; seq_id++) {
                        int current_input_id = input_ids[index * max_length + seq_id];
                        if (current_input_id < prompt_param.p_prompt_tuning_id_start) {
                            output_ids[no_prompt_output_seq_id * batch_size + index] = current_input_id;
                            no_prompt_output_seq_id++;
                        }
                    }
                }
            }
            else {
                const int seq_id   = index % max_length;
                const int batch_id = index / max_length;
                if (seq_id < length) {
                    output_ids[seq_id * batch_size + batch_id] = input_ids[index];
                }
            }
        }

        // embedding lookup from word ids [batch, length] (part of [batch, max_length]) and [vocab, hidden] to generate
        // embedding [batch, length, hidden]
        const int word_index      = index / hidden_units;
        const int word_index_row  = word_index / length;  // batch_id
        const int word_index_col  = word_index % length;
        const int real_word_index = word_index_row * max_length + word_index_col;
        const int step            = start_step + word_index % length;
        const int col_index       = index % hidden_units;
        const int input_id        = input_ids == nullptr ? real_word_index : input_ids[real_word_index];
        const int prompt_id       = input_id - prompt_param.p_prompt_tuning_id_start;
        T         embedding       = (T)0.0f;
        if (PROMPT_SRC > 0 && prompt_id >= 0) {
            if (PROMPT_SRC == 1) {
                // from loaded prompt embedding tables
                embedding =
                    prompt_param.p_prompt_tuning_batch_weights[word_index_row][prompt_id * hidden_units + col_index];
            }
            else {
                // from request prompt embedding
                embedding =
                    prompt_param
                        .request_prompt_embedding[word_index_row * prompt_param.request_prompt_max_length * hidden_units
                                                  + prompt_id * hidden_units + col_index];
            }
        }
        else {
            embedding = embedding_table[input_id * hidden_units + col_index];
        }
        T pos_embed        = pos_table == nullptr ? (T)0.f : pos_table[(step - 1) * hidden_units + col_index];
        from_tensor[index] = embedding + pos_embed;
    }
}

#define WORD_POS_EMBEDDING_LOOPUP_KERNEL(OUTPUT_ID, PROMPT_SRC)                                                        \
    start_id_embedding_position_lookups_kernel<T, OUTPUT_ID, PROMPT_SRC><<<grid, block, 0, stream>>>(from_tensor,      \
                                                                                                     output_ids,       \
                                                                                                     embedding_table,  \
                                                                                                     pos_table,        \
                                                                                                     prompt_param,     \
                                                                                                     input_ids,        \
                                                                                                     start_step,       \
                                                                                                     length,           \
                                                                                                     max_length,       \
                                                                                                     batch_size,       \
                                                                                                     hidden_units);

template<typename T>
void invokeInputIdsEmbeddingLookupPosEncoding(T*                    from_tensor,
                                              int*                  output_ids,
                                              const T*              embedding_table,  // can also be inputs_embeds
                                              const T*              pos_table,
                                              pPromptTuningParam<T> prompt_param,
                                              const int*            input_ids,
                                              const int             start_step,
                                              const int             length,
                                              const int             max_length,
                                              const int             batch_size,
                                              const int             hidden_units,
                                              cudaStream_t          stream)
{
    dim3       grid(min(batch_size * length, 65536));
    dim3       block(min(hidden_units, 512));
    const bool has_output_ids = output_ids != nullptr;
    FT_CHECK(!(has_output_ids && input_ids == nullptr));

    if (has_output_ids) {
        if (prompt_param.use_request_p_prompt_embedding) {
            WORD_POS_EMBEDDING_LOOPUP_KERNEL(true, 2);
        }
        else if (prompt_param.p_prompt_tuning_batch_weights != nullptr) {
            WORD_POS_EMBEDDING_LOOPUP_KERNEL(true, 1);
        }
        else {
            WORD_POS_EMBEDDING_LOOPUP_KERNEL(true, 0);
        }
    }
    else {
        if (prompt_param.use_request_p_prompt_embedding) {
            WORD_POS_EMBEDDING_LOOPUP_KERNEL(false, 2);
        }
        else if (prompt_param.p_prompt_tuning_batch_weights != nullptr) {
            WORD_POS_EMBEDDING_LOOPUP_KERNEL(false, 1);
        }
        else {
            WORD_POS_EMBEDDING_LOOPUP_KERNEL(false, 0);
        }
    }
}

#ifdef ENABLE_FP32
template void invokeInputIdsEmbeddingLookupPosEncoding(float*                    from_tensor,
                                                       int*                      output_ids,
                                                       const float*              embedding_table,
                                                       const float*              pos_table,
                                                       pPromptTuningParam<float> prompt_param,
                                                       const int*                input_ids,
                                                       const int                 start_step,
                                                       const int                 length,
                                                       const int                 max_length,
                                                       const int                 batch_size,
                                                       const int                 hidden_units,
                                                       cudaStream_t              stream);
#endif

template void invokeInputIdsEmbeddingLookupPosEncoding(half*                    from_tensor,
                                                       int*                     output_ids,
                                                       const half*              embedding_table,
                                                       const half*              pos_table,
                                                       pPromptTuningParam<half> prompt_param,
                                                       const int*               input_ids,
                                                       const int                start_step,
                                                       const int                length,
                                                       const int                max_length,
                                                       const int                batch_size,
                                                       const int                hidden_units,
                                                       cudaStream_t             stream);

#ifdef ENABLE_BF16
template void invokeInputIdsEmbeddingLookupPosEncoding(__nv_bfloat16*                    from_tensor,
                                                       int*                              output_ids,
                                                       const __nv_bfloat16*              embedding_table,
                                                       const __nv_bfloat16*              pos_table,
                                                       pPromptTuningParam<__nv_bfloat16> prompt_param,
                                                       const int*                        input_ids,
                                                       const int                         start_step,
                                                       const int                         length,
                                                       const int                         max_length,
                                                       const int                         batch_size,
                                                       const int                         hidden_units,
                                                       cudaStream_t                      stream);
#endif

// TODO Add half2 implementation
template<typename T>
__global__ void transposeAxis01(T* out, T* in, const int dim0, const int dim1, const int dim2)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < dim0 * dim1 * dim2) {
        const int input_dim2_index = index % dim2;
        index                      = (index - input_dim2_index) / dim2;
        const int input_dim1_index = index % dim1;
        index                      = (index - input_dim1_index) / dim1;
        const int input_dim0_index = index % dim0;

        out[input_dim1_index * dim0 * dim2 + input_dim0_index * dim2 + input_dim2_index] =
            in[input_dim0_index * dim1 * dim2 + input_dim1_index * dim2 + input_dim2_index];
    }
}

template<typename T>
void invokeTransposeAxis01(T* out, T* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(dim0 * dim1 * dim2 / 512.)));
    transposeAxis01<<<grid, block, 0, stream>>>(out, in, dim0, dim1, dim2);
}

template void
invokeTransposeAxis01(float* out, float* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template void
invokeTransposeAxis01(half* out, half* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template void
invokeTransposeAxis01(int* out, int* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template<typename T>
__global__ void transposeAxis01(T* out, T* in, const int* in_skipping_dim1, const int dim0, const int dim1)
{
    // out: [dim1, dim0]
    // in: [dim0, dim1]
    // in_skipping_dim1: [dim1]

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < dim0 * dim1) {
        const int input_dim1_index = index % dim1;
        index                      = (index - input_dim1_index) / dim1;
        const int input_dim0_index = index % dim0;
        const int in_offset        = in_skipping_dim1 == nullptr ? 0 : in_skipping_dim1[input_dim1_index] * dim1;

        out[input_dim1_index * dim0 + input_dim0_index] = in[in_offset + input_dim0_index * dim1 + input_dim1_index];
    }
}

template<typename T>
void invokeTransposeAxis01(
    T* out, T* in, const int* in_skipping_dim1, const int dim0, const int dim1, cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(dim0 * dim1 / 512.)));
    transposeAxis01<<<grid, block, 0, stream>>>(out, in, in_skipping_dim1, dim0, dim1);
}

template void invokeTransposeAxis01(
    int* out, int* in, const int* in_skipping_dim1, const int dim0, const int dim1, cudaStream_t stream);

}  // namespace turbomind
