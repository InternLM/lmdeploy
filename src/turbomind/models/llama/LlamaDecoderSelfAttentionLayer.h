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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/layers/attention_layers/DecoderSelfAttentionLayer.h

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/nccl_utils.h"

namespace turbomind {

template<typename T>
class LlamaDecoderSelfAttentionLayer {
public:
    void freeBuffer();
    void allocateBuffer(size_t batch_size, int key_len, int max_memory_len);

    LlamaDecoderSelfAttentionLayer(size_t                      head_num,
                                   size_t                      kv_head_num,
                                   size_t                      size_per_head,
                                   const LlamaAttentionParams& attn_params,
                                   NcclParam                   tensor_para,
                                   cudaStream_t                stream,
                                   cublasMMWrapper*            cublas_wrapper,
                                   IAllocator*                 allocator,
                                   bool                        is_free_buffer_after_forward,
                                   int                         quant_policy):
        head_num_(head_num),
        kv_head_num_(kv_head_num),
        size_per_head_(size_per_head),
        hidden_units_(head_num * size_per_head),
        local_head_num_(head_num / tensor_para.world_size_),
        local_kv_head_num_(kv_head_num_ / tensor_para.world_size_),
        local_hidden_units_(hidden_units_ / tensor_para.world_size_),
        params_(attn_params),
        tensor_para_(tensor_para),
        stream_(stream),
        linear_(cublas_wrapper, stream),
        allocator_(allocator),
        kv_cache_block_len_(128),  ///
        is_free_buffer_after_forward_(is_free_buffer_after_forward),
        quant_policy_(quant_policy)
    {
    }

    ~LlamaDecoderSelfAttentionLayer()
    {
        freeBuffer();
    }

    void forward(TensorMap* output_tensors, const TensorMap* input_tensors, const LlamaAttentionWeight<T>* weights);

private:
    const size_t head_num_;
    const size_t kv_head_num_;
    const size_t size_per_head_;
    const size_t hidden_units_;
    const size_t local_head_num_;
    const size_t local_kv_head_num_;
    const size_t local_hidden_units_;
    const size_t kv_cache_block_len_;
    const bool   is_free_buffer_after_forward_;
    const int    quant_policy_;

    const LlamaAttentionParams& params_;

    NcclParam tensor_para_;

    cudaStream_t   stream_;
    IAllocator*    allocator_;
    LlamaLinear<T> linear_;

    T* qkv_buf_     = nullptr;
    T* context_buf_ = nullptr;

    bool is_allocate_buffer_{};
};

}  // namespace turbomind
