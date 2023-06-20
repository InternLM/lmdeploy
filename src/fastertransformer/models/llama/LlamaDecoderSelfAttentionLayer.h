/*
 * Copyright (c) OpenMMLab. All rights reserved.
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

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/attention_layers/DecoderSelfAttentionLayer.h

#pragma once

#include "src/fastertransformer/models/llama/LlamaDenseWeight.h"
#include "src/fastertransformer/models/llama/LlamaLinear.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace fastertransformer {

template<typename T>
class LlamaDecoderSelfAttentionLayer {
public:
    void freeBuffer();
    void allocateBuffer(size_t batch_size, int key_len, int max_memory_len);

    LlamaDecoderSelfAttentionLayer(size_t           head_num,
                                   size_t           size_per_head,
                                   size_t           rotary_embedding_dim,
                                   bool             neox_rotary_style,
                                   NcclParam        tensor_para,
                                   cudaStream_t     stream,
                                   cublasMMWrapper* cublas_wrapper,
                                   IAllocator*      allocator,
                                   bool             is_free_buffer_after_forward):
        head_num_(head_num),
        size_per_head_(size_per_head),
        hidden_units_(head_num * size_per_head),
        local_head_num_(head_num / tensor_para.world_size_),
        local_hidden_units_(hidden_units_ / tensor_para.world_size_),
        rotary_embedding_dim_(rotary_embedding_dim),
        neox_rotary_style_(neox_rotary_style),
        tensor_para_(tensor_para),
        stream_(stream),
        linear_(cublas_wrapper, stream),
        allocator_(allocator),
        is_free_buffer_after_forward_(is_free_buffer_after_forward)
    {
    }

    ~LlamaDecoderSelfAttentionLayer()
    {
        freeBuffer();
    }

    void forward(TensorMap* output_tensors, const TensorMap* input_tensors, const LlamaAttentionWeight<T>* weights);

private:
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t hidden_units_;
    const size_t local_head_num_;
    const size_t local_hidden_units_;
    const size_t rotary_embedding_dim_;
    const bool   is_free_buffer_after_forward_;

    const bool neox_rotary_style_;

    NcclParam tensor_para_;

    cudaStream_t   stream_;
    IAllocator*    allocator_;
    LlamaLinear<T> linear_;

    T* qkv_buf_     = nullptr;
    T* context_buf_ = nullptr;
    // T*   weight_buf_  = nullptr;
    // T* k_cache_buf_{};
    // T* v_cache_buf_{};

    // T* tmp_k_cache_buf_{};
    // T* tmp_v_cache_buf_{};
    // T* tmp_cache_buf_{};

    bool is_allocate_buffer_{};
};

}  // namespace fastertransformer