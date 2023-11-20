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
#include "src/turbomind/kernels/decoder_multihead_attention/decoder_multihead_attention.h"
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
void LlamaDecoderSelfAttentionLayer<T>::allocateBuffer(size_t batch_size)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const size_t local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    qkv_buf_ = reinterpret_cast<T*>(
        allocator_->reMalloc(qkv_buf_, sizeof(T) * batch_size * local_q_kv_head_num * size_per_head_, false));
    context_buf_ =
        reinterpret_cast<T*>(allocator_->reMalloc(context_buf_, sizeof(T) * batch_size * local_hidden_units_, false));

    workspace_ = (float*)allocator_->reMalloc(
        workspace_, sizeof(float) * batch_size * local_head_num_ * kMaxSplitK * (size_per_head_ + 2), false);

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
    const bool* finished_data         = input_tensors->getPtr<bool>("finished");

    const int sum_seq_len = input_tensors->getVal<int>("sum_seq_len");
    const int max_seq_len = input_tensors->getVal<int>("max_seq_len");

    T*  hidden_features_data = output_tensors->getPtr<T>("attention_output");
    T** key_cache_ptrs       = output_tensors->getPtr<T*>("key_cache");
    T** value_cache_ptrs     = output_tensors->getPtr<T*>("value_cache");

    int* cu_block_counts = input_tensors->at("cu_block_counts").getPtr<int>();

    const int layer_id = input_tensors->getVal<int>("layer_id");

    const int step = input_tensors->getVal<int>("step");
    // const int step_1 = step - 1;

    const int batch_size = input_tensors->at("input_query").shape[0];

    const float* rope_theta = input_tensors->getPtr<const float>("rope_theta", nullptr);

    allocateBuffer(batch_size);

    // for (int i = 0; i < batch_size; ++i) {
    //     if (gSequenceIds(i) == 1) {
    //         Compare((T*)input_query_data + hidden_units_ * i,
    //                 hidden_units_,
    //                 Concat("query", gSequenceIds(i), seqlens[i], layer_id),
    //                 compare_mode,
    //                 stream_);
    //     }
    // }

    {
        NvtxScope scope("qkv_gemm");
        linear_.forward(qkv_buf_, input_query_data, batch_size, weights->qkv);
    }

    // if (layer_id == 0) {
    //     Compare(qkv_buf_, batch_size * 3 * hidden_units_, Concat("qkv_buf", step, layer_id), kCmpRead, stream_);
    // }

    const auto layer_offset = layer_id * local_kv_head_num_ * kv_cache_block_len_ * size_per_head_;
    // const int  memory_len   = max_seq_len;

    DecoderMultiHeadAttentionParams<T> params{};

    params.out    = context_buf_;
    params.q      = qkv_buf_;
    params.k      = params.q + local_head_num_ * size_per_head_;
    params.v      = params.k + local_kv_head_num_ * size_per_head_;
    params.stride = (local_head_num_ + 2 * local_kv_head_num_) * size_per_head_;

    params.q_bias = weights->qkv.bias;
    params.k_bias = params.q_bias + local_head_num_ * size_per_head_;
    params.v_bias = params.k_bias + local_kv_head_num_ * size_per_head_;

    params.batch_size    = batch_size;
    params.cu_block_cnts = cu_block_counts;

    params.k_cache_block_ptrs  = (void**)key_cache_ptrs;
    params.v_cache_block_ptrs  = (void**)value_cache_ptrs;
    params.kv_cache_block_size = kv_cache_block_len_;

    params.finished          = finished_data;
    params.per_sample_length = sequence_lengths_data;
    params.rope_theta        = rope_theta;

    params.layer_offset = layer_offset;

    params.num_heads     = local_head_num_;
    params.num_kv_heads  = local_kv_head_num_;
    params.size_per_head = size_per_head_;
    params.inv_sqrt_dh   = 1.f / std::sqrt((float)params.size_per_head);

    params.rotary_embedding_dim    = size_per_head_;
    params.rotary_embedding_base   = params_.rotary_embedding_base;
    params.max_position_embeddings = params_.max_position_embeddings;
    // params.use_dynamic_ntk = params_.use_dynamic_ntk;
    params.use_logn_attn = params_.use_logn_attn;

    params.partial_O = workspace_;
    params.partial_M = params.partial_O + batch_size * local_head_num_ * kMaxSplitK * size_per_head_;
    params.partial_L = params.partial_M + batch_size * local_head_num_ * kMaxSplitK;

    // avg_batch_size = sum_seq_len / max_seq_len
    // max_split_k    = kMaxSplitK  / avg_batch_size
    // max_split_k'   = min(max_split_k, max_seq_lens / kSliceLen)

    const float avg_batch_size = max_seq_len ? (float)sum_seq_len / max_seq_len : 1;
    FT_CHECK(avg_batch_size >= 1.f);

    const int max_split_k = std::max(1, (int)std::ceil(kMaxSplitK / avg_batch_size));

    // if (layer_id == 0) {
    //     TM_LOG_INFO("avg_batch_size = %.1f, max_split_k = %d", avg_batch_size, max_split_k);
    // }

    params.max_split_k = max_split_k;
    params.max_seq_len = max_seq_len;

    params.arch   = arch_;
    params.stream = stream_;

    params.quant_policy = quant_policy_;
    std::copy(weights->past_kv_scale.begin(), weights->past_kv_scale.end(), std::begin(params.kv_quant_params));

    {
        NvtxScope scope("decoder_multihead_attention");
        DispatchDecoderMultiheadAttention<T>(params);
    }

    // for (int i = 0; i < batch_size; ++i) {
    //     if (gSequenceIds(i) == 1) {
    //         Compare((T*)context_buf_ + hidden_units_ * i,
    //                 hidden_units_,
    //                 Concat("context_buf", gSequenceIds(i), seqlens[i], layer_id),
    //                 compare_mode,
    //                 stream_);
    //     }
    // }

    // if (layer_id == 0) {
    //     Compare(context_buf_, batch_size * hidden_units_, Concat("context_buf", step, layer_id), kCmpRead, stream_);
    // }

    {
        NvtxScope scope("o_gemm");
        linear_.forward(hidden_features_data, context_buf_, batch_size, weights->output);
    }

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
