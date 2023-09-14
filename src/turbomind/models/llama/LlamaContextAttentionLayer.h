/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/layers/attention_layers/GptContextAttentionLayer.h

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/nccl_utils.h"

namespace turbomind {

template<typename T>
class LlamaContextAttentionLayer {
public:
    void freeBuffer();
    void allocateBuffer(size_t batch_size, size_t num_token, size_t max_q_len, size_t max_kv_len);

    LlamaContextAttentionLayer(size_t               head_num,
                               size_t               kv_head_num,
                               size_t               size_per_head,
                               LlamaAttentionParams attn_params,
                               NcclParam            tensor_para,
                               cudaStream_t         stream,
                               cublasMMWrapper*     cublas_wrapper,
                               IAllocator*          allocator,
                               bool                 is_free_buffer_after_forward,
                               bool                 use_fmha,
                               int                  quant_policy):
        head_num_(head_num),
        size_per_head_(size_per_head),
        hidden_units_(head_num * size_per_head),
        local_head_num_(head_num / tensor_para.world_size_),
        local_kv_head_num_(kv_head_num / tensor_para.world_size_),
        head_n_rep_(head_num / kv_head_num),
        params_(attn_params),
        tensor_para_(tensor_para),
        stream_(stream),
        cublas_wrapper_(cublas_wrapper),
        linear_(cublas_wrapper, stream),
        allocator_(allocator),
        is_free_buffer_after_forward_(is_free_buffer_after_forward),
        use_fmha_(use_fmha),
        quant_policy_(quant_policy)
    {
        FT_CHECK(head_num % kv_head_num == 0);
    }

    void forward(TensorMap* output_tensors, const TensorMap* input_tensors, const LlamaAttentionWeight<T>* weights);

    void fusedMultiHeadAttention(T**    key_cache_ptrs,
                                 T**    val_cache_ptrs,
                                 size_t cache_layer_offset,
                                 T*     attention_mask,
                                 int*   cu_seqlens,
                                 int*   context_lengths,
                                 int    batch_size,
                                 int    max_q_len,
                                 int    max_k_len,
                                 int    max_seq_len);

    void unfusedMultiHeadAttention(T**          key_cache_ptrs,
                                   T**          val_cache_ptrs,
                                   size_t       cache_layer_offset,
                                   const T*     attention_mask,
                                   const int*   padding_offset,
                                   const int*   context_length,
                                   int          batch_size,
                                   int          num_token,
                                   int          max_q_len,
                                   int          max_k_len,
                                   int          max_seq_len,
                                   int          quant_policy,
                                   const float* kv_scale);

private:
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t hidden_units_;
    const size_t local_kv_head_num_;
    const size_t local_head_num_;
    const size_t head_n_rep_;
    const bool   is_free_buffer_after_forward_;

    const LlamaAttentionParams params_;

    const bool use_fmha_;
    const int  quant_policy_;

    NcclParam tensor_para_;

    cudaStream_t     stream_;
    IAllocator*      allocator_;
    cublasMMWrapper* cublas_wrapper_;
    LlamaLinear<T>   linear_;

    T*     qkv_buf_{};
    T*     q_buf_2_{};
    T*     k_buf_2_{};
    T*     v_buf_2_{};
    T*     k_cache_buf_{};
    T*     v_cache_buf_{};
    T*     qk_buf_{};
    float* qk_buf_float_{};
    T*     qkv_buf_2_{};
    T*     qkv_buf_3_{};

    bool is_allocate_buffer_ = false;
};

}  // namespace turbomind
