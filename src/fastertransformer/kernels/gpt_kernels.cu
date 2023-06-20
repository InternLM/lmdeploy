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

#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

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

template<typename T>
__global__ void inputIdsEmbeddingLookupPosEncodingSoftPrompt(inputIdsEmbeddingLookupPosEncodingSoftPromptParam<T> param)
{
    // 1. Copy the input ids to output ids and transpose output ids to [seq_len, batch_size, beam_width].
    // 2. Embedding lookup by input ids and concat with soft prompt. The axis of concatenation is on axis of seq_len.

    // Assume batch size is 2 and prompts are [[t1, t2], [t3], [t4, t5]], input_ids are [[s1, s2], [s3], [s4]]
    // then the order of output_ids is
    // [ [?, ?, s1, s2]
    //   [?, s3, padding, padding]
    //   [?, ?, s4, padding] ]
    // and the order of embedding is
    // [ [t1, t2, s1, s2]
    //   [t3, s3, padding, padding]
    //   [t4, t5, s4, padding] ]
    // where "?" means undefined values and we should attach it.

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < param.batch_size * param.beam_width * (param.max_prefix_soft_prompt_length + param.max_input_length)
                     * param.hidden_units;
         index += blockDim.x * gridDim.x) {
        // transpose the input_ids [batch, length] (part of [batch, beam, max_input_length]) to
        // output_ids [length, batch, beam].
        // ouptut_ids need to add padding in the beginning for soft prompting.

        if (index < param.batch_size * param.beam_width * param.max_input_length) {
            int       tmp_index = index;
            const int seq_id    = tmp_index % param.max_input_length;
            tmp_index           = (tmp_index - seq_id) / param.max_input_length;
            const int beam_id   = tmp_index % param.beam_width;
            tmp_index           = (tmp_index - beam_id) / param.beam_width;
            const int batch_id  = tmp_index % param.batch_size;
            if (seq_id < param.max_input_length) {
                param.output_ids[(param.prefix_soft_prompt_lengths[batch_id] + seq_id) * param.batch_size
                                     * param.beam_width
                                 + batch_id * param.beam_width + beam_id] = param.input_ids[index];
            }
        }

        // embedding lookup from word ids [batch, beam, length] (part of [batch, beam, max_input_length]), [vocab,
        // hidden] and [batch, max_prefix_soft_prompt_length, hidden] to generate embedding [batch, beam, length +
        // max_prefix_soft_prompt_length, hidden]
        int       tmp_index = index;
        const int hidden_id = tmp_index % param.hidden_units;
        tmp_index           = (tmp_index - hidden_id) / param.hidden_units;
        const int seq_id    = tmp_index % (param.max_prefix_soft_prompt_length + param.max_input_length);
        tmp_index           = (tmp_index - seq_id) / (param.max_prefix_soft_prompt_length + param.max_input_length);
        const int beam_id   = tmp_index % param.beam_width;
        tmp_index           = (tmp_index - beam_id) / param.beam_width;
        const int batch_id  = tmp_index % param.batch_size;
        const int64_t hidden_units = param.hidden_units;
        T         embedding =
            (seq_id < param.prefix_soft_prompt_lengths[batch_id]) ?
                        (T)param.prefix_soft_prompt_embedding[batch_id * param.max_prefix_soft_prompt_length * hidden_units
                                                      + seq_id * hidden_units + hidden_id] :
                            param.embedding_table[param.input_ids[batch_id * param.beam_width * param.max_input_length
                                                      + beam_id * param.max_input_length
                                                      + (seq_id - param.prefix_soft_prompt_lengths[batch_id])]
                                          * hidden_units
                                      + hidden_id];

        T pos_embed              = param.pos_table == nullptr ?
                                       (T)0.0f :
                                       param.pos_table[(param.start_step + seq_id - 1) * hidden_units + hidden_id];
        param.from_tensor[index] = embedding + pos_embed;

        if (seq_id == 0 && hidden_id == 0) {
            param.input_lengths[batch_id * param.beam_width + beam_id] += param.prefix_soft_prompt_lengths[batch_id];
        }
    }
}

template<typename T>
void invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(inputIdsEmbeddingLookupPosEncodingSoftPromptParam<T> param)
{
    dim3 grid(min(param.batch_size * param.beam_width * (param.max_input_length + param.max_prefix_soft_prompt_length),
                  65536));
    dim3 block(min(param.hidden_units, 512));
    inputIdsEmbeddingLookupPosEncodingSoftPrompt<T><<<grid, block, 0, param.stream>>>(param);
}

template void
invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(inputIdsEmbeddingLookupPosEncodingSoftPromptParam<float> param);

template void
invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(inputIdsEmbeddingLookupPosEncodingSoftPromptParam<half> param);

#ifdef ENABLE_BF16
template void invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(
    inputIdsEmbeddingLookupPosEncodingSoftPromptParam<__nv_bfloat16> param);
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

template<typename T, bool PREFIX_PROMPT>
__global__ void buildDecoderAttentionMaskKernel(T*         attention_mask,
                                                const int* sequence_lengths,
                                                const int* prefix_prompt_lengths,
                                                const int  max_seq_len,
                                                const int  max_prompt_length)
{
    // sequence_lengths: [batch_size]
    // attention_mask: [batch_size, 1, max_seq_len, max_seq_len + max_prompt_length]
    const int max_prompt_seq_length = max_seq_len + max_prompt_length;
    const int mask_size_per_seq     = max_seq_len * max_prompt_seq_length;
    attention_mask += blockIdx.x * mask_size_per_seq;
    const int seq_length    = sequence_lengths[blockIdx.x];
    const int prompt_length = PREFIX_PROMPT ? prefix_prompt_lengths[blockIdx.x] : 0;
    for (int i = threadIdx.x; i < mask_size_per_seq; i += blockDim.x) {
        int row_id = i / max_prompt_seq_length;
        int col_id = i % max_prompt_seq_length;
        if (row_id < seq_length && col_id <= (row_id + prompt_length)) {
            attention_mask[i] = (T)(1.0f);
        }
        else {
            attention_mask[i] = (T)(0.0f);
        }
    }
}

template<typename T>
void invokeBuildDecoderAttentionMask(T*           attention_mask,
                                     const int*   sequence_lengths,
                                     const int*   prefix_prompt_lengths,
                                     const int    batch_size,
                                     const int    max_seq_len,
                                     const int    max_prompt_length,
                                     cudaStream_t stream)
{
    if (max_prompt_length == 0) {
        buildDecoderAttentionMaskKernel<T, false><<<batch_size, 256, 0, stream>>>(
            attention_mask, sequence_lengths, prefix_prompt_lengths, max_seq_len, max_prompt_length);
    }
    else {
        buildDecoderAttentionMaskKernel<T, true><<<batch_size, 256, 0, stream>>>(
            attention_mask, sequence_lengths, prefix_prompt_lengths, max_seq_len, max_prompt_length);
    }
}

template void invokeBuildDecoderAttentionMask(float*       attention_mask,
                                              const int*   sequence_lengths,
                                              const int*   prefix_prompt_lengths,
                                              const int    batch_size,
                                              const int    max_seq_len,
                                              const int    max_prompt_length,
                                              cudaStream_t stream);
template void invokeBuildDecoderAttentionMask(half*        attention_mask,
                                              const int*   sequence_lengths,
                                              const int*   prefix_prompt_lengths,
                                              const int    batch_size,
                                              const int    max_seq_len,
                                              const int    max_prompt_length,
                                              cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBuildDecoderAttentionMask(__nv_bfloat16* attention_mask,
                                              const int*     sequence_lengths,
                                              const int*     prefix_prompt_lengths,
                                              const int      batch_size,
                                              const int      max_seq_len,
                                              const int      max_prompt_length,
                                              cudaStream_t   stream);
#endif
#ifdef ENABLE_FP8
template void invokeBuildDecoderAttentionMask(__nv_fp8_e4m3* attention_mask,
                                              const int*     sequence_lengths,
                                              const int*     prefix_prompt_lengths,
                                              const int      batch_size,
                                              const int      max_seq_len,
                                              const int      max_prompt_length,
                                              cudaStream_t   stream);
#endif

template<typename T>
__launch_bounds__(1024, 1) __global__ void lookupHiddenStateOfLastToken(T*         from_tensor,
                                                                        const T*   hidden_state,
                                                                        const int* input_lengths,
                                                                        const int  max_input_length,
                                                                        const int  batch_size,
                                                                        const int  hidden_units)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * hidden_units;
         index += blockDim.x * gridDim.x) {
        const int col_index = index % hidden_units;
        const int batch_id  = index / hidden_units;
        from_tensor[index]  = hidden_state[batch_id * max_input_length * hidden_units
                                          + (input_lengths[batch_id] - 1) * hidden_units + col_index];
    }
}

template<typename T>
void invokeLookupHiddenStateOfLastToken(T*           from_tensor,
                                        const T*     hidden_state,
                                        const int*   input_lengths,
                                        const int    max_input_length,
                                        const int    batch_size,
                                        const int    hidden_units,
                                        cudaStream_t stream)
{
    const int grid_size = (int)(ceil(batch_size * hidden_units / 1024.));
    dim3      grid(min(grid_size, 65536));
    dim3      block(min(hidden_units, 1024));
    lookupHiddenStateOfLastToken<T><<<grid, block, 0, stream>>>(
        from_tensor, hidden_state, input_lengths, max_input_length, batch_size, hidden_units);
}

template void invokeLookupHiddenStateOfLastToken(float*       from_tensor,
                                                 const float* hidden_state,
                                                 const int*   input_lengths,
                                                 const int    max_input_length,
                                                 const int    batch_size,
                                                 const int    hidden_units,
                                                 cudaStream_t stream);

template void invokeLookupHiddenStateOfLastToken(half*        from_tensor,
                                                 const half*  hidden_state,
                                                 const int*   input_lengths,
                                                 const int    max_input_length,
                                                 const int    batch_size,
                                                 const int    hidden_units,
                                                 cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeLookupHiddenStateOfLastToken(__nv_bfloat16*       from_tensor,
                                                 const __nv_bfloat16* hidden_state,
                                                 const int*           input_lengths,
                                                 const int            max_input_length,
                                                 const int            batch_size,
                                                 const int            hidden_units,
                                                 cudaStream_t         stream);
#endif

template<bool PREFIX_PROMPT>
__global__ void tileGptPromptInputs(int*       tiled_input_ids,
                                    int*       tiled_input_lengths,
                                    int*       tiled_prompt_lengths,
                                    const int* input_ids,
                                    const int* input_lengths,
                                    const int* prefix_prompt_lengths,
                                    const int  max_input_length)
{
    if (threadIdx.x == 0) {
        tiled_input_lengths[blockIdx.x * gridDim.y + blockIdx.y] = input_lengths[blockIdx.x];
        if (PREFIX_PROMPT) {
            tiled_prompt_lengths[blockIdx.x * gridDim.y + blockIdx.y] = prefix_prompt_lengths[blockIdx.x];
        }
    }
    for (int index = threadIdx.x; index < max_input_length; index += blockDim.x) {
        tiled_input_ids[(blockIdx.x * gridDim.y + blockIdx.y) * max_input_length + index] =
            input_ids[blockIdx.x * max_input_length + index];
    }
}

void invokeTileGptPromptInputs(int*         tiled_input_ids,
                               int*         tiled_input_lengths,
                               int*         tiled_prompt_lengths,
                               const int*   input_ids,
                               const int*   input_lengths,
                               const int*   prefix_prompt_lengths,
                               const int    batch_size,
                               const int    beam_width,
                               const int    max_input_length,
                               cudaStream_t stream)
{
    dim3 grid(batch_size, beam_width);
    dim3 block(min(1024, max_input_length));
    if (prefix_prompt_lengths != nullptr) {
        tileGptPromptInputs<true><<<grid, block, 0, stream>>>(tiled_input_ids,
                                                              tiled_input_lengths,
                                                              tiled_prompt_lengths,
                                                              input_ids,
                                                              input_lengths,
                                                              prefix_prompt_lengths,
                                                              max_input_length);
    }
    else {
        tileGptPromptInputs<false><<<grid, block, 0, stream>>>(tiled_input_ids,
                                                               tiled_input_lengths,
                                                               tiled_prompt_lengths,
                                                               input_ids,
                                                               input_lengths,
                                                               prefix_prompt_lengths,
                                                               max_input_length);
    }
}

void invokeTileGptInputs(int*         tiled_input_ids,
                         int*         tiled_input_lengths,
                         const int*   input_ids,
                         const int*   input_lengths,
                         const int    batch_size,
                         const int    beam_width,
                         const int    max_input_length,
                         cudaStream_t stream)
{
    invokeTileGptPromptInputs(tiled_input_ids,
                              tiled_input_lengths,
                              nullptr,
                              input_ids,
                              input_lengths,
                              nullptr,
                              batch_size,
                              beam_width,
                              max_input_length,
                              stream);
}

void setSeqLimitLen(uint32_t* seq_len_d, Tensor seq_len, int limit_len_offset, int batch_size)
{
    std::vector<uint32_t> seq_len_h(batch_size);
    for (int i = 0; i < batch_size; i++) {
        seq_len_h[i] = seq_len.getPtr<uint32_t>()[i] + limit_len_offset;
    }
    cudaH2Dcpy(seq_len_d, seq_len_h.data(), batch_size);
}

template<int TB_SIZE>
__global__ void
find_context_dups(int* shared_contexts, const int* input_ids, const size_t batch_size, const size_t input_seq_len)
{
    /* We compare all context pairs (i, j), with i (tgt) < j (src) , to detect duplicate
     * inputs. If there's a match between i and j, we store i at the
     * j-th position of shared_context. So that we know that j can be
     * represented by i. shared_contexts is initialized like shared_contexts[i] = i
     * and when there's a match, we actually use shared_contexts[j] = min(shared_contexts[j], i)
     * so that in the end, shared_contexts effectively contains an index
     * to the match with the lowest index context.
     * Note that shared_contexts[i] <= i, a property that will be used when uncompacting
     * inputs.
     */
    typedef cub::BlockReduce<int, TB_SIZE>       BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ bool                              match;

    /* Each block is responsible for a (i, j) pair. To map the block space to
     * the i < j space, we need to convert a linear addressing to a triangle, of
     * size (batch_size * (batch_size - 1)) / 2
     * For more information, check https://en.wikipedia.org/wiki/Triangular_number
     */

    // blockIdx = [0, 1, 2, ... n(n-1)/2] -> base_index = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, ..., n - 2]
    const int base_index = floorf(0.5f * (sqrtf(1 + 8 * blockIdx.x) - 1));
    const int src_idx    = base_index + 1;  // base_index \in [1, batch_size)

    const int rev_base_index = base_index * (base_index + 1) / 2;
    const int tgt_idx        = blockIdx.x - rev_base_index;  // tgt_idx \in [0, src_idx)

    const int padded_length = TB_SIZE * ((input_seq_len + TB_SIZE - 1) / TB_SIZE);

    int sum = 0;
    for (int i = threadIdx.x; i < padded_length; i += TB_SIZE) {
        int compare =
            (i >= input_seq_len) ? 1 : input_ids[tgt_idx * input_seq_len + i] == input_ids[src_idx * input_seq_len + i];

        sum = BlockReduce(temp_storage).Sum(compare);

        if (threadIdx.x == 0) {
            match = (sum == TB_SIZE);
        }

        __syncthreads();

        if (!match) {
            break;
        }
    }

    if (threadIdx.x == 0 && match) {
        atomicMin(&shared_contexts[src_idx], tgt_idx);
    }
}

constexpr int DUPS_INDICES_BLOCK_SIZE = 128;

__global__ void generate_dups_indices(int*         batch_to_compact,
                                      int*         compact_to_batch,
                                      int*         compact_size,
                                      const int*   shared_contexts,
                                      const size_t batch_size,
                                      const size_t input_seq_len)
{
    const int padded_batchsize = blockDim.x * ((batch_size + blockDim.x - 1) / blockDim.x);

    typedef cub::BlockScan<int, DUPS_INDICES_BLOCK_SIZE, cub::BLOCK_SCAN_WARP_SCANS> BlockScan;
    __shared__ typename BlockScan::TempStorage                                       temp_storage;
    __shared__ int                                                                   scan_offset;

    int scan = 0;
    for (int batch = threadIdx.x; batch < padded_batchsize; batch += blockDim.x) {
        bool masked     = (batch >= batch_size);
        bool first_iter = batch < blockDim.x;

        int is_first_occur = masked ? 0 : shared_contexts[batch] == batch;
        BlockScan(temp_storage).ExclusiveSum(is_first_occur, scan);

        if (!masked && is_first_occur) {
            int compact_idx = scan + (first_iter ? 0 : scan_offset);
            // Context rep. writes initial index
            batch_to_compact[batch]       = compact_idx;
            compact_to_batch[compact_idx] = batch;
        }

        if (threadIdx.x == blockDim.x - 1) {
            scan_offset = scan + is_first_occur + (first_iter ? 0 : scan_offset);
        }

        __syncthreads();

        if (!masked && !is_first_occur) {
            // Fill the rest of batch_to_compact based on what rep. wrote
            const int src_idx       = batch_to_compact[shared_contexts[batch]];
            batch_to_compact[batch] = src_idx;
        }
    }

    if (threadIdx.x == 0) {
        *compact_size = scan_offset;
    }
}

__global__ void init_shared_contexts(int* shared_contexts, const size_t batch_size)
{
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= batch_size) {
        return;
    }
    shared_contexts[global_idx] = global_idx;
}

void invokeFindContextDups(int*         shared_contexts,
                           int*         batch_to_compact,
                           int*         compact_to_batch,
                           int*         compact_size,
                           const int*   input_ids,
                           const size_t batch_size,
                           const size_t input_seq_len,
                           cudaStream_t stream)
{
    dim3 block{512};
    dim3 grid{((int)batch_size + block.x - 1) / block.x};
    init_shared_contexts<<<grid, block, 0, stream>>>(shared_contexts, batch_size);

    grid = dim3{(unsigned int)(batch_size * (batch_size - 1)) / 2};
    if (input_seq_len <= 128) {
        block = 128;
        find_context_dups<128><<<grid, block, 0, stream>>>(shared_contexts, input_ids, batch_size, input_seq_len);
    }
    else {
        block = 256;
        find_context_dups<256><<<grid, block, 0, stream>>>(shared_contexts, input_ids, batch_size, input_seq_len);
    }

    generate_dups_indices<<<1, DUPS_INDICES_BLOCK_SIZE, 0, stream>>>(
        batch_to_compact, compact_to_batch, compact_size, shared_contexts, batch_size, input_seq_len);
}

template<typename T>
__global__ void compact_inputs(T*         compact_input,
                               T*         compact_attention_mask,
                               int*       compact_input_lengths,
                               const T*   decoder_input,
                               const T*   decoder_mask,
                               const int* input_lengths,
                               const int* compact_idx,
                               size_t     compact_size,
                               size_t     seq_len,
                               size_t     hidden_dimension)
{
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < compact_size * seq_len * hidden_dimension) {
        const int h_id     = global_idx % hidden_dimension;
        const int seq_id   = (global_idx / hidden_dimension) % seq_len;
        const int batch_id = global_idx / (hidden_dimension * seq_len);

        compact_input[global_idx] = decoder_input[(compact_idx[batch_id] * seq_len + seq_id) * hidden_dimension + h_id];
    }

    if (global_idx < compact_size * seq_len * seq_len) {
        const int seq1_id  = global_idx % seq_len;
        const int seq2_id  = (global_idx / seq_len) % seq_len;
        const int batch_id = global_idx / (seq_len * seq_len);

        compact_attention_mask[global_idx] =
            decoder_mask[(compact_idx[batch_id] * seq_len + seq2_id) * seq_len + seq1_id];
    }

    if (global_idx < compact_size) {
        compact_input_lengths[global_idx] = input_lengths[compact_idx[global_idx]];
    }
}

template<typename T>
void invokeCompactInputs(T*           compact_input,
                         T*           compact_attention_mask,
                         int*         compact_input_lengths,
                         const T*     decoder_input,
                         const T*     decoder_mask,
                         const int*   input_lengths,
                         const int*   compact_idx,
                         size_t       compact_size,
                         size_t       seq_len,
                         size_t       hidden_dimension,
                         cudaStream_t stream)
{
    /* Compact relevant decoder_layer inputs based on the identical contexts.
     * For example, decoder_input is [batch_size, seq_len, H]. It's compacted
     * into compact_input [compact_size, seq_len, H] such that
     * compact_input[i, ...] = decoder_input[compact_idx[i], ...] */
    const size_t elems_n = compact_size * seq_len * max(hidden_dimension, seq_len);
    const dim3   blockDim(512);
    const dim3   gridDim((elems_n + 512 - 1) / 512);

    compact_inputs<T><<<gridDim, blockDim, 0, stream>>>(compact_input,
                                                        compact_attention_mask,
                                                        compact_input_lengths,
                                                        decoder_input,
                                                        decoder_mask,
                                                        input_lengths,
                                                        compact_idx,
                                                        compact_size,
                                                        seq_len,
                                                        hidden_dimension);
}

#define INSTANTIATE_INVOKE_COMPACT_INPUTS(T)                                                                           \
    template void invokeCompactInputs<T>(T * compact_input,                                                            \
                                         T * compact_attention_mask,                                                   \
                                         int*         compact_input_lengths,                                           \
                                         const T*     decoder_input,                                                   \
                                         const T*     decoder_mask,                                                    \
                                         const int*   input_lengths,                                                   \
                                         const int*   compact_idx,                                                     \
                                         size_t       compact_size,                                                    \
                                         size_t       seq_len,                                                         \
                                         size_t       hidden_dimension,                                                \
                                         cudaStream_t stream)
INSTANTIATE_INVOKE_COMPACT_INPUTS(half);
INSTANTIATE_INVOKE_COMPACT_INPUTS(float);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_COMPACT_INPUTS(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_COMPACT_INPUTS

template<typename T>
__global__ void uncompact_outputs(T*         uncompact_buffer,
                                  const T*   compact_buffer,
                                  const int* batch_to_compact_idx,
                                  size_t     batch_size,
                                  size_t     buffer_stride)
{
    /* Uncompact a buffer IN of size [Compact, Stride] into OUT of size [Batch, Stride]
     * so that \forall i, OUT[i, :] = IN[batch_to_compact_idx[i], :]
     */
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= batch_size * buffer_stride) {
        return;
    }

    const int stride_idx = global_idx % buffer_stride;
    const int batch_idx  = global_idx / buffer_stride;

    const int src                = batch_to_compact_idx[batch_idx];
    uncompact_buffer[global_idx] = compact_buffer[src * buffer_stride + stride_idx];
}

template<typename T>
void invokeUnCompactOutputs(T*           uncompact_buffer,
                            const T*     compact_buffer,
                            const int*   batch_to_compact_idx,
                            size_t       batch_size,
                            size_t       buffer_stride,
                            cudaStream_t stream)
{
    const size_t num_elems = batch_size * buffer_stride;
    const dim3   blockDim(1024);
    const dim3   gridDim((num_elems + blockDim.x - 1) / blockDim.x);

    uncompact_outputs<T><<<gridDim, blockDim, 0, stream>>>(
        uncompact_buffer, compact_buffer, batch_to_compact_idx, batch_size, buffer_stride);
}

#define INSTANTIATE_INVOKE_UNCOMPACT_OUTPUTS(T)                                                                        \
    template void invokeUnCompactOutputs(T*           uncompact_buffer,                                                \
                                         const T*     compact_buffer,                                                  \
                                         const int*   batch_to_compact_idx,                                            \
                                         size_t       batch_size,                                                      \
                                         size_t       buffer_stride,                                                   \
                                         cudaStream_t stream)
INSTANTIATE_INVOKE_UNCOMPACT_OUTPUTS(half);
INSTANTIATE_INVOKE_UNCOMPACT_OUTPUTS(float);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_UNCOMPACT_OUTPUTS(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_UNCOMPACT_OUTPUTS

template<typename T>
__global__ void uncompact_caches(T*         uncompact_k_cache,
                                 T*         uncompact_v_cache,
                                 const T*   compact_k_cache,
                                 const T*   compact_v_cache,
                                 const int* batch_to_compact_idx,
                                 size_t     batch_size,
                                 size_t     num_heads,
                                 size_t     max_seq_len,
                                 size_t     seq_len,
                                 size_t     size_per_head,
                                 size_t     local_batch_size,
                                 size_t     ite)
{
    const int hidden_dimension    = num_heads * size_per_head;
    const int num_elems_per_batch = seq_len * hidden_dimension;
    const int num_elems_cache     = batch_size * num_elems_per_batch;
    const int x_size              = 16 / sizeof(T);

    for (int global_idx = blockIdx.x * blockDim.x + threadIdx.x; global_idx < 2 * num_elems_cache;
         global_idx += blockDim.x * gridDim.x) {

        const bool     handle_k  = global_idx < num_elems_cache;
        const T* const cache_src = handle_k ? compact_k_cache : compact_v_cache;
        T* const       cache_dst = handle_k ? uncompact_k_cache : uncompact_v_cache;
        const int      idx       = handle_k ? global_idx : global_idx - num_elems_cache;

        const int src_offset = idx % num_elems_per_batch;
        const int batch_idx  = idx / num_elems_per_batch;
        const int batch_src  = batch_to_compact_idx[batch_idx] - ite * local_batch_size;

        if (batch_src < 0 || batch_src >= local_batch_size) {
            continue;
        }

        int dst_offset;
        if (handle_k) {
            const int i0 = idx % (x_size * seq_len);
            const int i1 = (idx / (x_size * seq_len)) % (num_heads * size_per_head / x_size);
            dst_offset   = i1 * max_seq_len * x_size + i0;
        }
        else {
            const int i0 = idx % (size_per_head * seq_len);
            const int i1 = (idx / (size_per_head * seq_len)) % (num_heads);
            dst_offset   = i1 * max_seq_len * size_per_head + i0;
        }

        cache_dst[batch_idx * max_seq_len * hidden_dimension + dst_offset] =
            cache_src[batch_src * num_elems_per_batch + src_offset];
    }
}

template<typename T>
void invokeUnCompactCaches(T*           uncompact_k_cache,
                           T*           uncompact_v_cache,
                           const T*     compact_k_cache,
                           const T*     compact_v_cache,
                           const int*   batch_to_compact_idx,
                           size_t       batch_size,
                           size_t       num_heads,
                           size_t       max_seq_len,
                           size_t       seq_len,
                           size_t       size_per_head,
                           size_t       local_batch_size,
                           size_t       ite,
                           cudaStream_t stream)
{
    const dim3 blockDim(512);
    const dim3 gridDim(1024);
    uncompact_caches<T><<<gridDim, blockDim, 0, stream>>>(uncompact_k_cache,
                                                          uncompact_v_cache,
                                                          compact_k_cache,
                                                          compact_v_cache,
                                                          batch_to_compact_idx,
                                                          batch_size,
                                                          num_heads,
                                                          max_seq_len,
                                                          seq_len,
                                                          size_per_head,
                                                          local_batch_size,
                                                          ite);
}

#define INSTANTIATE_INVOKE_UNCOMPACT_CACHES(T)                                                                         \
    template void invokeUnCompactCaches(T*           uncompact_k_cache,                                                \
                                        T*           uncompact_v_cache,                                                \
                                        const T*     compact_k_cache,                                                  \
                                        const T*     compact_v_cache,                                                  \
                                        const int*   batch_to_compact_idx,                                             \
                                        size_t       batch_size,                                                       \
                                        size_t       num_heads,                                                        \
                                        size_t       max_seq_len,                                                      \
                                        size_t       seq_len,                                                          \
                                        size_t       size_per_head,                                                    \
                                        size_t       local_batch_size,                                                 \
                                        size_t       ite,                                                              \
                                        cudaStream_t stream)
INSTANTIATE_INVOKE_UNCOMPACT_CACHES(half);
INSTANTIATE_INVOKE_UNCOMPACT_CACHES(float);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_UNCOMPACT_CACHES(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_UNCOMPACT_CACHES

template<bool PREFIX_PROMPT>
__global__ void update_padding_count(int*       total_padding_count,
                                     const int* input_lengths,
                                     const int* tiled_prompt_lengths,
                                     size_t     max_input_length,
                                     size_t     max_prompt_length,
                                     size_t     batch_size,
                                     size_t     beam_width)
{
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gidx >= batch_size * beam_width) {
        return;
    }

    const int batch_idx = gidx / beam_width;

    total_padding_count[gidx] +=
        PREFIX_PROMPT ? (max_input_length + max_prompt_length - input_lengths[batch_idx] - tiled_prompt_lengths[gidx]) :
                        (max_input_length - input_lengths[batch_idx]);
}

void invokeUpdatePaddingCount(int*         total_padding_count,
                              const int*   input_lengths,
                              const int*   tiled_prompt_lengths,
                              size_t       max_input_length,
                              size_t       max_prompt_length,
                              size_t       batch_size,
                              size_t       beam_width,
                              cudaStream_t stream)
{
    dim3 blockSize(256);
    dim3 gridSize((batch_size * beam_width + blockSize.x - 1) / blockSize.x);

    if (tiled_prompt_lengths != nullptr) {
        update_padding_count<true><<<gridSize, blockSize, 0, stream>>>(total_padding_count,
                                                                       input_lengths,
                                                                       tiled_prompt_lengths,
                                                                       max_input_length,
                                                                       max_prompt_length,
                                                                       batch_size,
                                                                       beam_width);
    }
    else {
        update_padding_count<false><<<gridSize, blockSize, 0, stream>>>(total_padding_count,
                                                                        input_lengths,
                                                                        tiled_prompt_lengths,
                                                                        max_input_length,
                                                                        max_prompt_length,
                                                                        batch_size,
                                                                        beam_width);
    }
}

template<bool PREFIX_PROMPT>
__global__ void mask_padding_tokens(bool*        masked_tokens,
                                    const int*   input_lengths,
                                    const int*   tiled_prefix_prompt_lengths,
                                    const size_t memory_len,
                                    const size_t max_input_length,
                                    const size_t initial_step,
                                    size_t       beam_width)
{
    const int seq_len = PREFIX_PROMPT ?
                            (input_lengths[blockIdx.x / beam_width] + tiled_prefix_prompt_lengths[blockIdx.x]) :
                            input_lengths[blockIdx.x / beam_width];
    for (int step = initial_step + seq_len + threadIdx.x; step < initial_step + max_input_length; step += blockDim.x) {
        masked_tokens[blockIdx.x * memory_len + step % memory_len] = true;
    }
}

void invokeMaskPaddingTokens(bool*        masked_tokens,
                             const int*   input_lengths,
                             const int*   tiled_prefix_prompt_lengths,
                             const size_t memory_len,
                             const size_t max_input_length,
                             const size_t initial_step,
                             size_t       batch_size,
                             size_t       beam_width,
                             cudaStream_t stream)
{
    dim3 blockSize(128);
    dim3 gridSize(batch_size * beam_width);
    if (tiled_prefix_prompt_lengths != nullptr) {
        mask_padding_tokens<true><<<gridSize, blockSize, 0, stream>>>(masked_tokens,
                                                                      input_lengths,
                                                                      tiled_prefix_prompt_lengths,
                                                                      memory_len,
                                                                      max_input_length,
                                                                      initial_step,
                                                                      beam_width);
    }
    else {
        mask_padding_tokens<false><<<gridSize, blockSize, 0, stream>>>(masked_tokens,
                                                                       input_lengths,
                                                                       tiled_prefix_prompt_lengths,
                                                                       memory_len,
                                                                       max_input_length,
                                                                       initial_step,
                                                                       beam_width);
    }
}

template<typename T>
__global__ void sum_length_dimension(
    float* out_buf, const T* in_buf, const size_t batch_size, const size_t input_length, const size_t hidden_dim)
{
    const int bidx = blockIdx.x;

    for (int hidx = threadIdx.x; hidx < hidden_dim; hidx += blockDim.x) {
        float accum = 0.0f;
        for (int step = 0; step < input_length; step++) {
            accum += static_cast<float>(in_buf[(bidx * input_length + step) * hidden_dim + hidx]);
        }
        out_buf[bidx * hidden_dim + hidx] = accum;
    }
}

template<typename T>
void invokeSumLengthDimension(float*       out_buf,
                              const T*     in_buf,
                              const size_t batch_size,
                              const size_t input_length,
                              const size_t hidden_dim,
                              cudaStream_t stream)
{
    dim3 gridSize(batch_size);
    dim3 blockSize(256);

    sum_length_dimension<<<gridSize, blockSize, 0, stream>>>(out_buf, in_buf, batch_size, input_length, hidden_dim);
}

#define INSTANTIATE_INVOKE_SUM_LENGTH_DIMENSION(T)                                                                     \
    template void invokeSumLengthDimension(float*       out_buf,                                                       \
                                           const T*     in_buf,                                                        \
                                           const size_t batch_size,                                                    \
                                           const size_t input_length,                                                  \
                                           const size_t hidden_dim,                                                    \
                                           cudaStream_t stream)
INSTANTIATE_INVOKE_SUM_LENGTH_DIMENSION(half);
INSTANTIATE_INVOKE_SUM_LENGTH_DIMENSION(float);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_SUM_LENGTH_DIMENSION(__nv_bfloat16);
#endif
#undef INSTANTIATE_INVOKE_SUM_LENGTH_DIMENSION

}  // namespace fastertransformer
