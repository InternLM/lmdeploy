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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGptContextDecoder.h

#pragma once

// #include "src/turbomind/kernels/add_residual_kernels.h"
// #include "src/turbomind/kernels/layernorm_kernels.h"
#include "src/turbomind/layers/BaseLayer.h"
// #include "src/turbomind/layers/FfnLayer.h"
// #include "src/turbomind/layers/attention_layers/BaseAttentionLayer.h"
#include "src/turbomind/models/llama/LlamaContextAttentionLayer.h"
#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/custom_ar_comm.h"
#include "src/turbomind/utils/nccl_utils.h"

namespace turbomind {

template<typename T>
class LlamaContextDecoder: public BaseLayer {
protected:
    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t num_token, size_t max_q_len, size_t max_kv_len);
    void freeBuffer() override;

    void initialize(bool use_fmha, int quant_policy);

    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t rotary_embedding_dim_;
    size_t hidden_units_;
    float  rmsnorm_eps_;

    NcclParam tensor_para_;

    T*   attn_ffn_io_{};
    T*   attention_mask_{};
    int* padding_offset_{};
    int* cu_seqlens_{};  // cu for cumulative

    size_t* h_pinned_token_num_ptr_{};

    LlamaContextAttentionLayer<T>* context_attention_layer_{};
    LlamaFfnLayer<T>*              silu_ffn_layer_{};

    const DataType data_type_;

    struct Session {
        size_t  batch_size;
        size_t  token_num;
        size_t  max_query_len;
        size_t  max_key_len;
        Tensor* k_cache;
        Tensor* v_cache;
        int*    input_length{};
        int*    history_length{};
        int*    context_length{};

        const std::vector<LlamaDecoderLayerWeight<T>*>* weights;
    };

    void forwardSelfAttn(const Session&                                 sess,
                         const std::unordered_map<std::string, Tensor>* input_tensors,
                         int                                            layer,
                         bool                                           is_final);

public:
    LlamaContextDecoder(size_t           head_num,
                        size_t           size_per_head,
                        size_t           inter_size,
                        size_t           num_layer,
                        size_t           rotary_embedding_dim,
                        float            rmsnorm_eps,
                        NcclParam        tensor_para,
                        cudaStream_t     stream,
                        cublasMMWrapper* cublas_wrapper,
                        IAllocator*      allocator,
                        bool             is_free_buffer_after_forward,
                        bool             use_fmha,
                        int              quant_policy);

    ~LlamaContextDecoder() override;

    virtual void forward(std::unordered_map<std::string, Tensor>*        output_tensors,
                         const std::unordered_map<std::string, Tensor>*  input_tensors,
                         const std::vector<LlamaDecoderLayerWeight<T>*>* decoder_layer_weights);

    virtual void forward(std::vector<Tensor>*                            output_tensors,
                         const std::vector<Tensor>*                      input_tensors,
                         const std::vector<LlamaDecoderLayerWeight<T>*>* decoder_layer_weights);
};

}  // namespace turbomind
