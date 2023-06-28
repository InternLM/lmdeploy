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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGptContextDecoder.cc

#include "src/fastertransformer/models/llama/LlamaContextDecoder.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/models/llama/LlamaContextDecoder.h"
#include "src/fastertransformer/models/llama/llama_decoder_kernels.h"
#include "src/fastertransformer/models/llama/llama_kernels.h"
#include "src/fastertransformer/utils/Tensor.h"

namespace fastertransformer {

template<typename T>
void LlamaContextDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void LlamaContextDecoder<T>::allocateBuffer(size_t batch_size, size_t num_token, size_t max_q_len, size_t max_kv_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    attn_ffn_io_    = (T*)allocator_->reMalloc(attn_ffn_io_, sizeof(T) * num_token * hidden_units_, false);
    attention_mask_ = (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * batch_size * max_q_len * max_kv_len, false);
    padding_offset_ = (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * max_q_len, false);
    cu_seqlens_     = (int*)allocator_->reMalloc(cu_seqlens_, sizeof(int) * (batch_size + 1), false);

    is_allocate_buffer_ = true;
}

template<typename T>
void LlamaContextDecoder<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)&attn_ffn_io_);
        allocator_->free((void**)&padding_offset_);
        allocator_->free((void**)&cu_seqlens_);
        allocator_->free((void**)&attention_mask_);
        allocator_->free((void**)&h_pinned_token_num_ptr_, true);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void LlamaContextDecoder<T>::initialize(bool use_fmha, int quant_policy)
{
    h_pinned_token_num_ptr_ = (size_t*)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true, true);

    context_attention_layer_ = new LlamaContextAttentionLayer<T>(head_num_,
                                                                 size_per_head_,
                                                                 rotary_embedding_dim_,
                                                                 false,  // neox_rotary_style
                                                                 tensor_para_,
                                                                 stream_,
                                                                 cublas_wrapper_,
                                                                 allocator_,
                                                                 is_free_buffer_after_forward_,
                                                                 use_fmha,
                                                                 quant_policy);

    silu_ffn_layer_ = new LlamaFfnLayer<T>(head_num_,
                                           size_per_head_,
                                           inter_size_,
                                           tensor_para_,
                                           stream_,
                                           cublas_wrapper_,
                                           allocator_,
                                           is_free_buffer_after_forward_);
}

template<typename T>
void LlamaContextDecoder<T>::forwardSelfAttn(const Session&                                 sess,
                                             const std::unordered_map<std::string, Tensor>* input_tensors,
                                             int                                            layer,
                                             bool                                           is_final)
{
    // FT_LOG_ERROR(__PRETTY_FUNCTION__);
    TensorMap self_attention_input_tensors{
        {"input_query", Tensor{MEMORY_GPU, data_type_, {sess.token_num, hidden_units_}, attn_ffn_io_}},
        {"attention_mask",
         {MEMORY_GPU, data_type_, {sess.batch_size, 1, sess.max_query_len, sess.max_key_len}, attention_mask_}},
        {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &layer}},
        {"is_final_layer", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_final}},
        {"padding_offset", {MEMORY_GPU, TYPE_INT32, {sess.token_num}, padding_offset_}},
        {"cu_seqlens", {MEMORY_GPU, TYPE_INT32, {sess.batch_size + 1}, cu_seqlens_}},
        {"input_lengths", {MEMORY_GPU, TYPE_INT32, {sess.batch_size}, sess.input_length}},
        {"history_lengths", {MEMORY_GPU, TYPE_INT32, {sess.batch_size}, sess.history_length}},
        {"context_lengths", {MEMORY_GPU, TYPE_INT32, {sess.batch_size}, sess.context_length}},
        {"max_seq_len", input_tensors->at("max_seq_len")}};

    auto& k_cache = *sess.k_cache;
    auto& v_cache = *sess.v_cache;

    TensorMap self_attention_output_tensors{
        {"hidden_features", {MEMORY_GPU, data_type_, {sess.token_num, hidden_units_}, attn_ffn_io_}},
        {"key_cache", k_cache},
        {"value_cache", v_cache},
    };

    context_attention_layer_->forward(&self_attention_output_tensors,  //
                                      &self_attention_input_tensors,
                                      &sess.weights->at(layer)->self_attn_weights);
}

template<typename T>
LlamaContextDecoder<T>::LlamaContextDecoder(size_t           head_num,
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
                                            int              quant_policy):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    hidden_units_(head_num * size_per_head),
    num_layer_(num_layer),
    rotary_embedding_dim_(rotary_embedding_dim),
    rmsnorm_eps_(rmsnorm_eps),
    tensor_para_(tensor_para),
    data_type_(getTensorType<T>())
{
    initialize(use_fmha, quant_policy);
}

template<typename T>
LlamaContextDecoder<T>::~LlamaContextDecoder()
{
    delete context_attention_layer_;
    delete silu_ffn_layer_;
    freeBuffer();
}

template<typename T>
void LlamaContextDecoder<T>::forward(std::vector<Tensor>*                            output_tensors,
                                     const std::vector<Tensor>*                      input_tensors,
                                     const std::vector<LlamaDecoderLayerWeight<T>*>* decoder_layer_weights)
{
    FT_CHECK(false);
}

template<typename T>
void LlamaContextDecoder<T>::forward(std::unordered_map<std::string, Tensor>*        output_tensors,
                                     const std::unordered_map<std::string, Tensor>*  input_tensors,
                                     const std::vector<LlamaDecoderLayerWeight<T>*>* decoder_layer_weights)
{
    /**
     * input tensors:
     *   \param decoder_input [num_token, hidden_units], float
     *   \param input_lengths [batch_size], int
     *   \param history_lengths [batch_size], int
     *   \param context_legnths [batch_size], int
     *   \param output_norm_weight [hidden_dims], float
     *   \param max_q_len [1], int on cpu
     *   \param max_kv_len [1], int on cpu
     *   \param max_seq_len [1], int on cpu
     *
     * output tensors:
     *   \param decoder_output [batch_size, seq_len, hidden_units],
     *   \param key_cache [num_layer, batch, local_head_num, size_per_head // x, max_seq_len, x]
     *   \param value_cache [num_layer, batch, local_head_num, max_seq_len, size_per_head]
     *   \param last_token_hidden_units [batch_size, hidden_units]
     */

    Session sess{};

    sess.token_num     = input_tensors->at("decoder_input").shape[0];
    sess.batch_size    = input_tensors->at("input_lengths").shape[0];
    sess.max_query_len = input_tensors->at("max_q_len").getVal<int>();
    sess.max_key_len   = input_tensors->at("max_kv_len").getVal<int>();
    sess.weights       = decoder_layer_weights;

    sess.input_length   = input_tensors->at("input_lengths").getPtr<int>();
    sess.history_length = input_tensors->at("history_lengths").getPtr<int>();
    sess.context_length = input_tensors->at("context_lengths").getPtr<int>();

    T* decoder_input_output = input_tensors->at("decoder_input").getPtr<T>();
    // T* decoder_output = output_tensors->at("decoder_output").getPtr<T>();

    sess.k_cache = &output_tensors->at("key_cache");
    sess.v_cache = &output_tensors->at("value_cache");

    allocateBuffer(sess.batch_size, sess.token_num, sess.max_query_len, sess.max_key_len);

    size_t tmp_token_num{};
    invokeGetPaddingOffsetAndCuSeqLens(h_pinned_token_num_ptr_,
                                       &tmp_token_num,  // updated token num
                                       padding_offset_,
                                       cu_seqlens_,
                                       input_tensors->at("input_lengths").getPtr<int>(),
                                       sess.batch_size,
                                       sess.max_query_len,
                                       stream_);
    sync_check_cuda_error();
    FT_CHECK(tmp_token_num == sess.token_num);

    invokeCreateCausalMasks(attention_mask_,
                            sess.input_length,
                            sess.context_length,
                            sess.max_query_len,
                            sess.max_key_len,
                            sess.batch_size,
                            stream_);
    sync_check_cuda_error();

    /////////////////////////////////////////////
    /// RMSNorm
    invokeRootMeanSquareNorm(attn_ffn_io_,
                             decoder_input_output,
                             decoder_layer_weights->at(0)->self_attn_norm_weights,
                             rmsnorm_eps_,
                             sess.token_num,
                             hidden_units_,
                             stream_);
    sync_check_cuda_error();

    for (size_t layer = 0; layer < num_layer_; ++layer) {
        /////////////////////////////////////////////
        /// self-attention
        forwardSelfAttn(sess, input_tensors, layer, false);

        invokeFusedAddBiasResidualRMSNorm(decoder_input_output,
                                          attn_ffn_io_,
                                          decoder_layer_weights->at(layer)->self_attn_weights.output.bias,
                                          decoder_layer_weights->at(layer)->ffn_norm_weights,
                                          rmsnorm_eps_,
                                          sess.token_num,
                                          hidden_units_,
                                          stream_);
        sync_check_cuda_error();

        ////////////////////////////////////////////
        /// feed-forward network
        TensorMap ffn_inputs{{"ffn_input", {MEMORY_GPU, data_type_, {sess.token_num, hidden_units_}, attn_ffn_io_}}};
        TensorMap ffn_outputs{{"ffn_output", {MEMORY_GPU, data_type_, {sess.token_num, hidden_units_}, attn_ffn_io_}}};
        silu_ffn_layer_->forward(&ffn_outputs, &ffn_inputs, &decoder_layer_weights->at(layer)->ffn_weights);

        auto scale_weight = layer < num_layer_ - 1 ? decoder_layer_weights->at(layer + 1)->self_attn_norm_weights :
                                                     input_tensors->at("output_norm_weight").getPtr<T>();
        invokeFusedAddBiasResidualRMSNorm(decoder_input_output,  //
                                          attn_ffn_io_,
                                          decoder_layer_weights->at(layer)->ffn_weights.output.bias,
                                          scale_weight,
                                          rmsnorm_eps_,
                                          sess.token_num,
                                          hidden_units_,
                                          stream_);
        sync_check_cuda_error();
    }

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class LlamaContextDecoder<float>;
template class LlamaContextDecoder<half>;

}  // namespace fastertransformer