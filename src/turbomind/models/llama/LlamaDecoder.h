/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2022, SK Telecom Authored by A. Dialog
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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoder.h

#include "src/fastertransformer/layers/BaseLayer.h"
// #include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/models/llama/LlamaDecoderLayerWeight.h"
#include "src/fastertransformer/models/llama/LlamaDecoderSelfAttentionLayer.h"
#include "src/fastertransformer/models/llama/LlamaFfnLayer.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace fastertransformer {

template<typename T>
class LlamaDecoder: public BaseLayer {
protected:
    void allocateBuffer() override;  // deprecated
    void allocateBuffer(size_t batch_size);
    void freeBuffer() override;
    void initialize(int quant_policy);

    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t rotary_embedding_dim_;
    size_t hidden_units_;
    float  rmsnorm_eps_;

    NcclParam tensor_para_;

    LlamaDecoderSelfAttentionLayer<T>* self_attention_layer_{};
    LlamaFfnLayer<T>*                  silu_ffn_layer_{};

    const DataType data_type_;

    struct Session {
        size_t                                          batch_size;
        int                                             ite;
        size_t                                          max_memory_len;
        Tensor*                                         k_cache;
        Tensor*                                         v_cache;
        const std::vector<LlamaDecoderLayerWeight<T>*>* weights;
    };

    void forwardSelfAttn(const Session&                                 sess,
                         T*                                             attn_io,
                         const std::unordered_map<std::string, Tensor>* input_tensors,
                         size_t                                         layer);

    void forwardFfn(const LlamaDecoder::Session& sess, T* ffn_io, size_t layer);

public:
    LlamaDecoder(size_t           head_num,
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
                 int              quant_policy),

        ~LlamaDecoder() override;

    virtual void forward(std::unordered_map<std::string, Tensor>*        output_tensors,
                         const std::unordered_map<std::string, Tensor>*  input_tensors,
                         const std::vector<LlamaDecoderLayerWeight<T>*>* decoder_layer_weights);

    virtual void forward(std::vector<Tensor>*                            output_tensors,
                         const std::vector<Tensor>*                      input_tensors,
                         const std::vector<LlamaDecoderLayerWeight<T>*>* decoder_layer_weights);
};

}  // namespace fastertransformer
