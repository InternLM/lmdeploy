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

#pragma once

#include "src/turbomind/utils/cuda_bf16_wrapper.h"
#include "src/turbomind/utils/cuda_fp8_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status_ = call;                                                                                    \
        if (status_ != cudaSuccess) {                                                                                  \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_));              \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////

// The structure of parameters for the masked multihead attention kernel.
//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.

template<typename T>
struct Multihead_attention_params_base {

    // The output buffer. Dimensions B x D.
    T* out = nullptr;

    // The input Qs and the associated bias. Dimensions B x D and D, resp.
    const T *q = nullptr, *q_bias = nullptr;
    // The input Ks and the associated bias. Dimensions B x D and D, resp.
    const T *k = nullptr, *k_bias = nullptr;
    // The input Vs and the associated bias. Dimensions B x D and D, resp.
    const T *v = nullptr, *v_bias = nullptr;

    // scales
    const float* query_weight_output_scale               = nullptr;
    const float* attention_qk_scale                      = nullptr;
    const float* attention_output_weight_input_scale_inv = nullptr;

    // Stride to handle the case when KQV is a single buffer
    int stride = 0;

    // The batch size.
    int batch_size = 0;
    // The beam width
    int beam_width = 0;
    // The sequence length.
    int memory_max_len = 0;
    // The number of heads (H).
    int num_heads = 0;
    // The hidden dimension per head (Dh).
    int hidden_size_per_head = 0;
    // The per-head latent space reserved for rotary embeddings.
    int rotary_embedding_dim = 0;
    // The maximum length of input sentences.
    int max_input_length = 0;
    // The current timestep. TODO(bhsueh) Check that do we only this param in cross attention?
    int timestep = 0;
    // The current timestep of each sentences (support different timestep for different sentences)

    // The 1.f / sqrt(Dh). Computed on the host.
    float inv_sqrt_dh = 0.0f;

    // Used when we have some input context like gpt
    const int* total_padding_tokens = nullptr;

    const bool* masked_tokens            = nullptr;
    const int*  prefix_prompt_lengths    = nullptr;
    int         max_prefix_prompt_length = 0;

    const T* relative_attention_bias        = nullptr;
    int      relative_attention_bias_stride = 0;
    // The slope per head of linear position bias to attention score (H).
    const T* linear_bias_slopes = nullptr;

    const float* qkv_scale_out       = nullptr;
    const float* attention_out_scale = nullptr;
    int          int8_mode           = 0;
    float        attention_k_scale   = 0.f;
    float        attention_k_zp      = 0.f;
    float        attention_v_scale   = 0.f;
    float        attention_v_zp      = 0.f;
};

template<typename T>
struct Multihead_attention_params: public Multihead_attention_params_base<T> {
    bool*      finished                   = nullptr;
    const int* length_per_sample          = nullptr;
    T**        k_cache_per_sample         = nullptr;
    T**        v_cache_per_sample         = nullptr;
    size_t     kv_cache_per_sample_offset = 0;
    int        num_kv_heads               = 0;
    int        max_position_embeddings    = 0;
    bool       use_dynamic_ntk            = false;
    bool       use_logn_attn              = false;
    float      rotary_embedding_base      = 10000.0f;
};

template<class T>
using Masked_multihead_attention_params = Multihead_attention_params<T>;

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<float>& params, const cudaStream_t& stream);
void masked_multihead_attention(const Masked_multihead_attention_params<uint16_t>& params, const cudaStream_t& stream);
#ifdef ENABLE_BF16
void masked_multihead_attention(const Masked_multihead_attention_params<__nv_bfloat16>& params,
                                const cudaStream_t&                                     stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
