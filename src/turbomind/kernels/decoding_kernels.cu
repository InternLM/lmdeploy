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

#include "src/turbomind/kernels/decoding_kernels.h"
#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
#include "src/turbomind/utils/cuda_type_utils.cuh"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

// PROMPT_SRC: 0 --> no prompts, 1 --> from loaded prompts, 2 --> from request prompts
template<typename T>
__global__ void embeddingLookupPosEncoding(T*            from_tensor,
                                           const T*      embedding_table,
                                           const T*      position_encoding,
                                           const int*    all_ids,
                                           const int*    padding_count,
                                           const int*    input_lengths,
                                           const int     local_token_num,
                                           const int64_t hidden_units,
                                           const int     step,
                                           const int     max_input_length,
                                           const int     token_num,
                                           const int     ite,
                                           const T       scale)
{
    // 1. lookup from embedding table
    // 2. multiply scale
    // 3. add the position encoding
    const int id_offset = step * token_num + ite * local_token_num;

    const bool use_padding_count = padding_count != nullptr;
    const bool use_input_len     = input_lengths != nullptr;

    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < local_token_num * hidden_units;
         index += blockDim.x * gridDim.x) {
        const int row_index   = index / hidden_units;
        const int col_index   = index % hidden_units;
        int       step_offset = step;
        if (use_padding_count) {
            step_offset -= padding_count[row_index];
        }
        else if (use_input_len) {
            step_offset -= max_input_length - input_lengths[row_index];
        }
        step_offset *= hidden_units;

        T val = embedding_table[all_ids[id_offset + row_index] * hidden_units + col_index] * scale;
        val   = val + position_encoding[step_offset + col_index];

        from_tensor[index] = val;
    }
}

// No absolute position embedding
// PROMPT_SRC: 0 --> no prompts, 1 --> from loaded prompts, 2 --> from request prompts
template<typename T, int PROMPT_SRC>
__global__ void embeddingLookup(T*                    from_tensor,
                                const T*              embedding_table,
                                const int*            all_ids,
                                pPromptTuningParam<T> prompt_param,
                                const int             local_token_num,
                                const int64_t         hidden_units,
                                const int             step,
                                const int             token_num,
                                const int             ite,
                                const int             seq_len,
                                const T               scale)
{
    // 1. lookup from embedding table
    // 2. multiply scale
    const int id_offset = step * token_num + ite * local_token_num;

    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < local_token_num * hidden_units;
         index += blockDim.x * gridDim.x) {

        const int word_index     = index / hidden_units;
        const int word_index_row = word_index / seq_len;  // batch_id
        const int col_index      = index % hidden_units;
        const int input_id       = all_ids == nullptr ? word_index : all_ids[id_offset + word_index];
        const int prompt_id      = input_id - prompt_param.p_prompt_tuning_id_start;
        T         embedding      = (T)0.0f;
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
        from_tensor[index] = embedding * scale;
    }
}

#define EMBEDDING_LOOKUP(PROMPT_SRC)                                                                                   \
    embeddingLookup<T, PROMPT_SRC><<<grid, block, 0, stream>>>(from_tensor,                                            \
                                                               embedding_table,                                        \
                                                               all_ids,                                                \
                                                               prompt_param,                                           \
                                                               local_token_num,                                        \
                                                               hidden_units,                                           \
                                                               step,                                                   \
                                                               token_num,                                              \
                                                               ite,                                                    \
                                                               seq_len,                                                \
                                                               scale);

/* Adapter function for invokeEmbeddingLookupPosEncoding{PadCount,InputLen} */
template<typename T>
void invokeEmbeddingLookupPosEncoding(T*                    from_tensor,
                                      const T*              embedding_table,
                                      const T*              position_encoding,
                                      const int*            all_ids,
                                      const int*            padding_count,
                                      const int*            input_lengths,
                                      pPromptTuningParam<T> prompt_param,
                                      const int             local_token_num,
                                      const int             hidden_units,
                                      const T               scale,
                                      const int             step,
                                      const int             max_input_length,
                                      const int             token_num,
                                      const int             ite,
                                      const int             seq_len,
                                      cudaStream_t          stream)
{
    dim3 grid(min(local_token_num, 65536));
    dim3 block(min(hidden_units, 1024));
    if (position_encoding != nullptr) {
        FT_CHECK_WITH_INFO(prompt_param.use_request_p_prompt_embedding == false
                               && prompt_param.p_prompt_tuning_batch_weights == nullptr,
                           fmtstr("embeddingLookupPosEncoding still not support prompt tuning"));
        embeddingLookupPosEncoding<T><<<grid, block, 0, stream>>>(from_tensor,
                                                                  embedding_table,
                                                                  position_encoding,
                                                                  all_ids,
                                                                  padding_count,
                                                                  input_lengths,
                                                                  local_token_num,
                                                                  hidden_units,
                                                                  step,
                                                                  max_input_length,
                                                                  token_num,
                                                                  ite,
                                                                  scale);
    }
    else {
        if (prompt_param.use_request_p_prompt_embedding) {
            EMBEDDING_LOOKUP(2);
        }
        else if (prompt_param.p_prompt_tuning_batch_weights != nullptr) {
            EMBEDDING_LOOKUP(1);
        }
        else {
            EMBEDDING_LOOKUP(0);
        }
    }
}

#undef EMBEDDING_LOOKUP

template<typename T>
void invokeEmbeddingLookupPosEncodingPadCount(T*                    from_tensor,
                                              const T*              embedding_table,
                                              const T*              position_encoding,
                                              const int*            all_ids,
                                              const int*            pad_count,
                                              pPromptTuningParam<T> prompt_param,
                                              const int             local_token_num,
                                              const int             hidden_units,
                                              const T               scale,
                                              const int             step,
                                              const int             token_num,
                                              const int             ite,
                                              const int             seq_len,
                                              cudaStream_t          stream)
{
    invokeEmbeddingLookupPosEncoding<T>(from_tensor,
                                        embedding_table,
                                        position_encoding,
                                        all_ids,
                                        pad_count,
                                        nullptr,
                                        prompt_param,
                                        local_token_num,
                                        hidden_units,
                                        scale,
                                        step,
                                        0,
                                        token_num,
                                        ite,
                                        seq_len,
                                        stream);
}

#define INSTANTIATE_LOOKUP_POS_ENCODING_PAD_COUNT(T)                                                                   \
    template void invokeEmbeddingLookupPosEncodingPadCount(T*                    from_tensor,                          \
                                                           const T*              embedding_table,                      \
                                                           const T*              position_encoding,                    \
                                                           const int*            all_ids,                              \
                                                           const int*            pad_count,                            \
                                                           pPromptTuningParam<T> prompt_param,                         \
                                                           const int             local_token_num,                      \
                                                           const int             hidden_units,                         \
                                                           const T               scale,                                \
                                                           const int             step,                                 \
                                                           const int             token_num,                            \
                                                           const int             ite,                                  \
                                                           const int             seq_len,                              \
                                                           cudaStream_t          stream)
INSTANTIATE_LOOKUP_POS_ENCODING_PAD_COUNT(float);
INSTANTIATE_LOOKUP_POS_ENCODING_PAD_COUNT(half);
#ifdef ENABLE_BF16
INSTANTIATE_LOOKUP_POS_ENCODING_PAD_COUNT(__nv_bfloat16);
#endif
#undef INSTANTIATE_LOOKUP_POS_ENCODING_PAD_COUNT

template<typename T>
__global__ void paddingEmbedding(T*            padded_embedding_kernel,
                                 T*            padded_embedding_bias,
                                 const T*      embedding_kernel,
                                 const T*      embedding_bias,
                                 const int64_t hidden_unit,
                                 const int64_t vocab_size,
                                 const int64_t vocab_size_padded)
{
    for (int64_t id = threadIdx.x + blockIdx.x * blockDim.x; id < hidden_unit * vocab_size_padded;
         id += blockDim.x * gridDim.x) {
        int row_id = id / vocab_size_padded;
        int col_id = id % vocab_size_padded;
        if (col_id < vocab_size) {
            padded_embedding_kernel[id] = embedding_kernel[row_id * vocab_size + col_id];
        }
        else {
            padded_embedding_kernel[id] = (T)(0.0f);
        }
    }

    for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < vocab_size_padded; id += blockDim.x * gridDim.x) {
        if (id < vocab_size) {
            padded_embedding_bias[id] = embedding_bias[id];
        }
        else {
            padded_embedding_bias[id] = (T)(0.0f);
        }
    }
}

template<typename T>
void invokePaddingEmbedding(T*           padded_embedding_kernel,
                            T*           padded_embedding_bias,
                            const T*     embedding_kernel,
                            const T*     embedding_bias,
                            const int    hidden_unit,
                            const int    vocab_size,
                            const int    vocab_size_padded,
                            cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(hidden_unit * vocab_size_padded / 512.)));
    paddingEmbedding<<<grid, block, 0, stream>>>(padded_embedding_kernel,
                                                 padded_embedding_bias,
                                                 embedding_kernel,
                                                 embedding_bias,
                                                 hidden_unit,
                                                 vocab_size,
                                                 vocab_size_padded);
}

// template void invokePaddingEmbedding(float*       padded_embedding_kernel,
//                                      float*       padded_embedding_bias,
//                                      const float* embedding_kernel,
//                                      const float* embedding_bias,
//                                      const int    hidden_unit,
//                                      const int    vocab_size,
//                                      const int    vocab_size_padded,
//                                      cudaStream_t stream);

// template void invokePaddingEmbedding(half*        padded_embedding_kernel,
//                                      half*        padded_embedding_bias,
//                                      const half*  embedding_kernel,
//                                      const half*  embedding_bias,
//                                      const int    hidden_unit,
//                                      const int    vocab_size,
//                                      const int    vocab_size_padded,
//                                      cudaStream_t stream);
// #ifdef ENABLE_BF16
// template void invokePaddingEmbedding(__nv_bfloat16*       padded_embedding_kernel,
//                                      __nv_bfloat16*       padded_embedding_bias,
//                                      const __nv_bfloat16* embedding_kernel,
//                                      const __nv_bfloat16* embedding_bias,
//                                      const int            hidden_unit,
//                                      const int            vocab_size,
//                                      const int            vocab_size_padded,
//                                      cudaStream_t         stream);
// #endif

template<typename T>
__global__ void paddingEmbeddingKernel(T*        padded_embedding_kernel,
                                       const T*  embedding_kernel,
                                       const int hidden_unit,
                                       const int vocab_size,
                                       const int vocab_size_padded)
{
    for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < hidden_unit * vocab_size_padded;
         id += blockDim.x * gridDim.x) {
        int row_id = id / hidden_unit;
        int col_id = id % hidden_unit;
        if (row_id < vocab_size) {
            padded_embedding_kernel[id] = embedding_kernel[row_id * hidden_unit + col_id];
        }
        else {
            padded_embedding_kernel[id] = (T)(0.0f);
        }
    }
}

template<typename T>
void invokePaddingEmbeddingKernel(T*           padded_embedding_kernel,
                                  const T*     embedding_kernel,
                                  const int    hidden_unit,
                                  const int    vocab_size,
                                  const int    vocab_size_padded,
                                  cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(hidden_unit * vocab_size_padded / 512.)));
    paddingEmbeddingKernel<<<grid, block, 0, stream>>>(
        padded_embedding_kernel, embedding_kernel, hidden_unit, vocab_size, vocab_size_padded);
}

// template void invokePaddingEmbeddingKernel(float*       padded_embedding_kernel,
//                                            const float* embedding_kernel,
//                                            const int    hidden_unit,
//                                            const int    vocab_size,
//                                            const int    vocab_size_padded,
//                                            cudaStream_t stream);

// template void invokePaddingEmbeddingKernel(half*        padded_embedding_kernel,
//                                            const half*  embedding_kernel,
//                                            const int    hidden_unit,
//                                            const int    vocab_size,
//                                            const int    vocab_size_padded,
//                                            cudaStream_t stream);

// #ifdef ENABLE_BF16
// template void invokePaddingEmbeddingKernel(__nv_bfloat16*       padded_embedding_kernel,
//                                            const __nv_bfloat16* embedding_kernel,
//                                            const int            hidden_unit,
//                                            const int            vocab_size,
//                                            const int            vocab_size_padded,
//                                            cudaStream_t         stream);
// #endif

template<typename T>
__global__ void plusScalar(T* buf, const T val, const int size)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        buf[i] += val;
    }
}

template<typename T>
void invokePlusScalar(T* buf, const T val, const int size, cudaStream_t stream)
{
    dim3 block(min(256, size));
    dim3 grid(ceil(size / 256.));
    plusScalar<<<block, grid, 0, stream>>>(buf, val, size);
}

template void invokePlusScalar(int* buf, const int val, const int size, cudaStream_t stream);

}  // namespace turbomind
