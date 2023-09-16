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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/layers/attention_layers/DecoderSelfAttentionLayer.cc
#include "src/turbomind/models/llama/LlamaDecoderSelfAttentionLayer.h"
#include "src/turbomind/kernels/decoder_masked_multihead_attention.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/nvtx_utils.h"
#include <string>
// #include <glog/logging.h>

namespace turbomind {

template<typename T>
struct SATypeConverter {
    using Type = T;
};

template<>
struct SATypeConverter<half> {
    using Type = uint16_t;
};

template<typename T>
static inline void fusedQKV_masked_attention_dispatch(const T*     qkv_buf,
                                                      const T*     qkv_bias,
                                                      const T*     relative_attention_bias,
                                                      T*           key_cache,
                                                      T*           value_cache,
                                                      T**          k_cache_per_sample,
                                                      T**          v_cache_per_sample,
                                                      size_t       kv_cache_per_sample_offset,
                                                      const int*   cache_indir,
                                                      T*           context_buf,
                                                      const bool*  finished,
                                                      const int*   sequence_lengths,
                                                      const int    max_batch_size,
                                                      const int    inference_batch_size,
                                                      const int    beam_width,
                                                      const int    head_num,
                                                      const int    kv_head_num,
                                                      const int    size_per_head,
                                                      const int    rotary_embedding_dim,
                                                      const float  rotary_embedding_base,
                                                      const int    max_position_embeddings,
                                                      const bool   use_dynamic_ntk,
                                                      const bool   use_logn_attn,
                                                      const int    memory_max_len,
                                                      const int*   prefix_prompt_lengths,
                                                      const int    max_prefix_prompt_length,
                                                      const int    max_input_len,
                                                      const int*   total_padding_tokens,
                                                      const int    step,
                                                      const float  q_scaling,
                                                      const int    relative_attention_bias_stride,
                                                      const T*     linear_bias_slopes,
                                                      const bool*  masked_tokens,
                                                      const int*   ia3_tasks,
                                                      const T*     ia3_key_weights,
                                                      const T*     ia3_value_weights,
                                                      const float* qkv_scale_out,
                                                      const float* attention_out_scale,
                                                      const int    int8_mode,
                                                      const float* attention_kv_scale,
                                                      cudaStream_t stream)
{
    using DataType = typename SATypeConverter<T>::Type;
    // Prepare the parameters.
    Masked_multihead_attention_params<DataType> params;
    memset(&params, 0, sizeof(params));
    // int hidden_units = head_num * size_per_head;
    if (qkv_bias != nullptr) {
        params.q_bias = reinterpret_cast<const DataType*>(qkv_bias);
        params.k_bias = reinterpret_cast<const DataType*>(qkv_bias) + head_num * size_per_head;
        params.v_bias = reinterpret_cast<const DataType*>(qkv_bias) + (head_num + kv_head_num) * size_per_head;
    }
    else {
        params.q_bias = nullptr;
        params.k_bias = nullptr;
        params.v_bias = nullptr;
    }

    // Set the output buffer.
    params.out = reinterpret_cast<DataType*>(context_buf);

    // Set the input buffers.
    // [B, nH + kvH, D]
    params.q = reinterpret_cast<const DataType*>(qkv_buf);
    params.k = reinterpret_cast<const DataType*>(qkv_buf) + head_num * size_per_head;
    params.v = reinterpret_cast<const DataType*>(qkv_buf) + (head_num + kv_head_num) * size_per_head;

    params.stride   = (head_num + 2 * kv_head_num) * size_per_head;
    params.finished = const_cast<bool*>(finished);

    FT_CHECK(k_cache_per_sample && v_cache_per_sample);

    params.k_cache_per_sample         = reinterpret_cast<DataType**>(k_cache_per_sample);
    params.v_cache_per_sample         = reinterpret_cast<DataType**>(v_cache_per_sample);
    params.kv_cache_per_sample_offset = kv_cache_per_sample_offset;
    params.batch_size                 = inference_batch_size;
    params.beam_width                 = beam_width;
    params.memory_max_len             = memory_max_len;
    params.prefix_prompt_lengths      = prefix_prompt_lengths;
    params.max_prefix_prompt_length   = max_prefix_prompt_length;
    params.length_per_sample          = sequence_lengths;  // max_input_length + current output length
    // timestep adding max_prefix_prompt_length for shared memory size calculation and rotary embedding computation
    params.timestep     = step + max_prefix_prompt_length - 1;
    params.num_heads    = head_num;
    params.num_kv_heads = kv_head_num;

    params.hidden_size_per_head    = size_per_head;
    params.rotary_embedding_dim    = rotary_embedding_dim;
    params.rotary_embedding_base   = rotary_embedding_base;
    params.max_position_embeddings = max_position_embeddings;
    params.use_dynamic_ntk         = use_dynamic_ntk;
    params.use_logn_attn           = use_logn_attn;

    // Note: keep norm factor (sqrt(K_dim)) when adopting megatron T5 structure (may adjust)
    params.inv_sqrt_dh = 1.F / (sqrtf((float)params.hidden_size_per_head) * q_scaling);

    params.total_padding_tokens = total_padding_tokens;
    if (relative_attention_bias != nullptr) {
        params.relative_attention_bias = reinterpret_cast<const DataType*>(relative_attention_bias);
    }
    params.relative_attention_bias_stride = relative_attention_bias_stride;
    params.masked_tokens                  = masked_tokens;

    // The slope of linear position bias per head, e.g., ALiBi.
    if (linear_bias_slopes != nullptr) {
        params.linear_bias_slopes = reinterpret_cast<const DataType*>(linear_bias_slopes);
    }
    params.max_input_length = max_input_len;

    params.int8_mode = int8_mode;

    if (int8_mode & QuantPolicy::kCacheKVInt8) {
        params.attention_k_scale = attention_kv_scale[0];
        params.attention_k_zp    = attention_kv_scale[1];
        params.attention_v_scale = attention_kv_scale[2];
        params.attention_v_zp    = attention_kv_scale[3];
    }

    PUSH_RANGE("scaled dot-product fusion");
    masked_multihead_attention(params, stream);
    POP_RANGE;
}

template<typename T>
void LlamaDecoderSelfAttentionLayer<T>::allocateBuffer(size_t batch_size, int key_len, int max_memory_len)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const size_t local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    qkv_buf_ = reinterpret_cast<T*>(
        allocator_->reMalloc(qkv_buf_, sizeof(T) * batch_size * local_q_kv_head_num * size_per_head_, false));
    context_buf_ =
        reinterpret_cast<T*>(allocator_->reMalloc(context_buf_, sizeof(T) * batch_size * local_hidden_units_, false));

    is_allocate_buffer_ = true;
}

template<typename T>
void LlamaDecoderSelfAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&context_buf_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void LlamaDecoderSelfAttentionLayer<T>::forward(TensorMap*                     output_tensors,
                                                const TensorMap*               input_tensors,
                                                const LlamaAttentionWeight<T>* weights)
{
    /**
     * input tensors:
     *    \param input_query [batch_size, hidden_units],
     *    \param sequence_lengths [batch_size]
     *    \param step [1] on cpu
     *    \param finished [batch_size]
     *    \param total_padding_tokens [batch_size]
     *    \param layer_id [1], int on cpu
     *    \param max_seq_len [1] on cpu
     *    \param masked_tokens [batch_size, memory_len], (optional), NOT USED YET
     *    \param cache_indirection [batch_size / beam_width, beam_width, memory_max_len] (optional)
     *
     * output tensors:
     *    \param attention_output [batch_size, hidden_units],
     *    \param key_cache [batch, local_head_num, memory_max_len, size_per_head]
     *    \param value_cache [batch, local_head_num, memory_max_len, size_per_head]
     */

    const T*    input_query_data      = input_tensors->getPtr<T>("input_query");
    const int*  sequence_lengths_data = input_tensors->getPtr<int>("sequence_lengths");
    const int*  total_padding_len     = input_tensors->getPtr<int>("total_padding_tokens");
    const bool* finished_data         = input_tensors->getPtr<bool>("finished", nullptr);
    const bool* masked_tokens_data    = input_tensors->getPtr<bool>("masked_tokens", nullptr);
    const int*  cache_indir           = input_tensors->getPtr<int>("cache_indirection", nullptr);

    T*  hidden_features_data = output_tensors->getPtr<T>("attention_output");
    T** key_cache_ptrs       = output_tensors->getPtr<T*>("key_cache");
    T** value_cache_ptrs     = output_tensors->getPtr<T*>("value_cache");

    const int layer_id = input_tensors->getVal<int>("layer_id");

    const int max_seq_len = input_tensors->getVal<int>("max_seq_len");
    const int step        = input_tensors->getVal<int>("step");

    const int step_1 = step - 1;

    const int batch_size = input_tensors->at("input_query").shape[0];
    const int beam_width = cache_indir != nullptr ? input_tensors->at("cache_indirection").shape[1] : 1;

    allocateBuffer(batch_size, step, max_seq_len);

    PUSH_RANGE("qkv_gemm");
    linear_.forward(qkv_buf_, input_query_data, batch_size, weights->qkv);
    POP_RANGE;

    const auto kv_cache_layer_offset = layer_id * local_kv_head_num_ * max_seq_len * size_per_head_;
    const int  memory_len            = max_seq_len;

    fusedQKV_masked_attention_dispatch<T>(
        qkv_buf_,
        weights->qkv.bias,  // query_weight.bias,
        nullptr,            // relative_attention_bias,
        nullptr,
        nullptr,
        key_cache_ptrs,
        value_cache_ptrs,
        kv_cache_layer_offset,
        cache_indir,
        context_buf_,
        finished_data,
        sequence_lengths_data,  // NOTE: current seq len including padding (fixed after meeting the finished id)
        batch_size,
        batch_size,
        beam_width,
        local_head_num_,
        local_kv_head_num_,
        size_per_head_,
        params_.rotray_embedding_dim,
        params_.rotary_embedding_base,
        params_.max_position_embeddings,
        params_.use_dynamic_ntk,
        params_.use_logn_attn,
        memory_len,
        nullptr,  // prefix_prompt_lengths
        0,        // max_prefix_prompt_length
        0,        // max_input_length, not used w/o linear_bias_slopes
        input_tensors->getPtr<int>("total_padding_tokens", nullptr),
        step,
        1.f,                            // q_scaling
        0,                              // relative_attention_bias_stride
        nullptr,                        // linear_bias_slopes
        nullptr,                        //  masked_tokens_data,
        nullptr,                        // ia3_tasks
        nullptr,                        // ia3_key_weights
        nullptr,                        // ia3_value_weights
        nullptr,                        // qkv_scale_out
        nullptr,                        // attention_out_scale
        quant_policy_,                  // int8_mode
        weights->past_kv_scale.data(),  // attention kv scale
        stream_);
    sync_check_cuda_error();

    linear_.forward(hidden_features_data, context_buf_, batch_size, weights->output);

    if (tensor_para_.world_size_ > 1) {
        NcclGuard nccl_guard(tensor_para_, stream_);
        ftNcclAllReduceSum(
            hidden_features_data, hidden_features_data, batch_size * hidden_units_, tensor_para_, stream_);
        sync_check_cuda_error();
    }

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }

    // LOG(WARNING);
}

template class LlamaDecoderSelfAttentionLayer<float>;
template class LlamaDecoderSelfAttentionLayer<half>;

}  // namespace turbomind
