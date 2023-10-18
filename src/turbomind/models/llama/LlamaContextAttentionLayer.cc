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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/layers/attention_layers/GptContextAttentionLayer.cc

#include "src/turbomind/models/llama/LlamaContextAttentionLayer.h"
#include "src/turbomind/kernels/bert_preprocess_kernels.h"
#include "src/turbomind/kernels/unfused_attention_kernels.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

template<typename T>
void LlamaContextAttentionLayer<T>::allocateBuffer(size_t batch_size,
                                                   size_t num_token,
                                                   size_t max_q_len,
                                                   size_t max_k_len)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    // no padding
    qkv_buf_ = (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * num_token * local_q_kv_head_num * size_per_head_, true);

    // padding is rebuilt for q/k/v_buf_2_
    // [qH + 2kvH, B, S, D]
    q_buf_2_ = (T*)allocator_->reMalloc(
        q_buf_2_, sizeof(T) * local_q_kv_head_num * batch_size * max_q_len * size_per_head_, true);
    k_buf_2_ = q_buf_2_ + local_head_num_ * batch_size * max_q_len * size_per_head_;
    v_buf_2_ = k_buf_2_ + local_kv_head_num_ * batch_size * max_q_len * size_per_head_;

    if (use_fmha_) {
        FlashAttentionOp<T> flash_attention(batch_size, local_head_num_, max_k_len, max_q_len, size_per_head_);
        if (flash_attention.get_workspace_size() > 0) {
            qk_buf_float_ = (float*)allocator_->reMalloc(qk_buf_float_, flash_attention.get_workspace_size(), true);
        }
    }
    else {
        // kv heads are repeated for unfused attention
        k_cache_buf_ = (T*)allocator_->reMalloc(
            k_cache_buf_, 2 * sizeof(T) * batch_size * local_head_num_ * max_k_len * size_per_head_, true);
        v_cache_buf_ = k_cache_buf_ + batch_size * local_head_num_ * max_k_len * size_per_head_;

        qk_buf_ =
            (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * local_head_num_ * max_q_len * max_k_len, true);

        // qkv_buf_2_ has padding
        qkv_buf_2_ = (T*)allocator_->reMalloc(
            qkv_buf_2_, sizeof(T) * batch_size * max_q_len * local_head_num_ * size_per_head_, true);
    }

    // qkv_buf_3_ padding is removed
    qkv_buf_3_ = (T*)allocator_->reMalloc(qkv_buf_3_, sizeof(T) * num_token * local_head_num_ * size_per_head_, true);

    is_allocate_buffer_ = true;
}

template<typename T>
void LlamaContextAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        TM_LOG_DEBUG(__PRETTY_FUNCTION__);

        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&q_buf_2_));
        if (use_fmha_) {
            allocator_->free((void**)&qk_buf_float_);
        }
        else {
            allocator_->free((void**)(&k_cache_buf_));
            allocator_->free((void**)(&qk_buf_));
            allocator_->free((void**)(&qkv_buf_2_));
        }
        allocator_->free((void**)(&qkv_buf_3_));

        is_allocate_buffer_ = false;
    }
}

template<typename T>
inline void LlamaContextAttentionLayer<T>::forward(TensorMap*                     output_tensors,
                                                   const TensorMap*               input_tensors,
                                                   const LlamaAttentionWeight<T>* weights)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    /**
     * input_tensors:
     *   \param input_query [token_num, hidden_dim]
     *   \param attention_mask [batch_size, 1, max_q_len, max_kv_len]
     *   \param padding_offset [token_num], int
     *   \param input_lengths [batch_size], int
     *   \param history_lengths [batch_size], int
     *   \param context_lengths [batch_size], int
     *   \param cu_seqlens [batch_size+1], int
     *   \param max_seq_len [1], int on cpu
     *   \param is_final_layer [1], bool on cpu
     *   \param layer_id [1], int on cpu
     *
     * output_tensors:
     *   \param hidden_features [token_num, hidden_dim]
     *   \param key_cache [batch_size], uint64
     *   \param value_cache [batch_size], uint64
     */

    /////////////////////////////////////////////
    /// parse inputs
    const int batch_size = input_tensors->at("attention_mask").shape[0];
    const int max_q_len  = input_tensors->at("attention_mask").shape[2];
    const int max_k_len  = input_tensors->at("attention_mask").shape[3];
    const int layer_id   = input_tensors->getVal<int>("layer_id");

    const int num_token = input_tensors->at("input_query").shape[0];

    const int max_seq_len = input_tensors->at("max_seq_len").getVal<int>();

    T* attention_out   = output_tensors->at("hidden_features").getPtr<T>();
    T* attention_input = input_tensors->at("input_query").getPtr<T>();
    T* attention_mask  = input_tensors->at("attention_mask").getPtr<T>();

    const auto input_length   = input_tensors->at("input_lengths").getPtr<const int>();
    const auto history_length = input_tensors->at("history_lengths").getPtr<const int>();
    const auto context_length = input_tensors->at("context_lengths").getPtr<const int>();
    int*       cu_seqlens     = input_tensors->at("cu_seqlens").getPtr<int>();

    const auto padding_offset = input_tensors->at("padding_offset").getPtr<int>();

    /////////////////////////////////////////////
    /// allocate buffers
    allocateBuffer(batch_size, num_token, max_q_len, max_k_len);

    //////////////////////////////////////////////
    /// qkv gemm
    // [token_num, hidden_dim] -> [token_num, 3, local_hidden_dim]
    linear_.forward(qkv_buf_, attention_input, num_token, weights->qkv);

    //////////////////////////////////////////////
    /// transpose qkv & apply rotary embedding & rebuild padding
    /// qkv [B, s, H + 2kvH, D] -> (q [B, H, s, D], k [B, kvH, s, D], v [B, kvH, s, D])
    invokeAddFusedQKVBiasTranspose(q_buf_2_,
                                   k_buf_2_,
                                   v_buf_2_,
                                   qkv_buf_,
                                   weights->qkv.bias,
                                   padding_offset,  // padding_offset,
                                   history_length,  // used for applying rotary embedding
                                   input_length,
                                   batch_size,
                                   max_q_len,  // seq_len
                                   num_token,  // batch_size * seq_len
                                   local_head_num_,
                                   local_kv_head_num_,
                                   size_per_head_,
                                   params_.rotray_embedding_dim,
                                   params_.rotary_embedding_base,
                                   params_.max_position_embeddings,
                                   params_.use_dynamic_ntk,
                                   params_.use_logn_attn,
                                   stream_);
    sync_check_cuda_error();

    const size_t layer_offset = layer_id * local_kv_head_num_ * max_seq_len * size_per_head_;

    auto k_cache_ptrs = output_tensors->getPtr<T*>("key_cache");
    auto v_cache_ptrs = output_tensors->getPtr<T*>("value_cache");
    //////////////////////////////////////////////////////////
    /// insert the k/v computed from inputs into k/v cache
    /// transpose kv -> kv cache
    // put k/v_buf from shape [B, kvH, s, D] to
    // k_buf_2 [B, kvH, s, D] -> key_cache [B, kvH, S[t:t+s], D/x, x]
    // v_buf_2 [B, kvH, s, D] -> val_cache [B, kvH, S[t:t+s], D/x, x]
    invokeExtendKVCache(k_cache_ptrs,
                        v_cache_ptrs,
                        layer_offset,
                        k_buf_2_,
                        v_buf_2_,
                        batch_size,
                        input_length,
                        max_q_len,
                        history_length,
                        max_seq_len,
                        size_per_head_,
                        local_kv_head_num_,
                        stream_,
                        quant_policy_,
                        weights->past_kv_scale.data());

    sync_check_cuda_error();
    if (use_fmha_) {
        fusedMultiHeadAttention(k_cache_ptrs,
                                v_cache_ptrs,
                                layer_offset,
                                attention_mask,
                                cu_seqlens,
                                input_tensors->at("context_lengths").getPtr<int>(),
                                batch_size,
                                max_q_len,
                                max_k_len,
                                max_seq_len);
    }
    else {
        unfusedMultiHeadAttention(k_cache_ptrs,
                                  v_cache_ptrs,
                                  layer_offset,
                                  attention_mask,
                                  padding_offset,
                                  context_length,
                                  batch_size,
                                  num_token,
                                  max_q_len,
                                  max_k_len,
                                  max_seq_len,
                                  quant_policy_,
                                  weights->past_kv_scale.data());
    }

    //////////////////////////////////////////////
    /// output gemm <Bs,HD> -> <Bs,HD>
    linear_.forward(attention_out, qkv_buf_3_, num_token, weights->output);

    if (tensor_para_.world_size_ > 1) {
        NcclGuard nccl_guard(tensor_para_, stream_);
        ftNcclAllReduceSum(attention_out, attention_out, num_token * hidden_units_, tensor_para_, stream_);
        sync_check_cuda_error();
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
void LlamaContextAttentionLayer<T>::fusedMultiHeadAttention(T**    key_cache_ptrs,
                                                            T**    val_cache_ptrs,
                                                            size_t cache_layer_offset,
                                                            T*     attention_mask,
                                                            int*   cu_seqlens,
                                                            int*   context_lengths,
                                                            int    batch_size,
                                                            int    max_q_len,
                                                            int    max_k_len,
                                                            int    max_seq_len)
{
    //////////////////////////////////////////////
    // flash attention
    // flash attention 2 only support half inputs
    using AttentionOp = FlashAttentionOp<T>;
    using Layout      = typename AttentionOp::AttentionLayout;
    Layout layout_q{
        int(local_head_num_ * max_q_len * size_per_head_), int(size_per_head_), int(max_q_len * size_per_head_)};
    Layout layout_k{int(local_head_num_ * max_seq_len * size_per_head_),
                    int(size_per_head_),
                    int(max_seq_len * size_per_head_),
                    false,
                    cache_layer_offset,
                    key_cache_ptrs};
    Layout layout_v{int(local_head_num_ * max_seq_len * size_per_head_),
                    int(size_per_head_),
                    int(max_seq_len * size_per_head_),
                    false,
                    cache_layer_offset,
                    val_cache_ptrs};
    Layout layout_o{
        int(local_head_num_ * max_q_len * size_per_head_),
        int(local_head_num_ * size_per_head_),
        int(size_per_head_),
        true,
    };
    size_t                       group_size = size_t(local_head_num_ / local_kv_head_num_);
    AttentionOp                  flash_attention(batch_size, local_head_num_, max_k_len, max_q_len, size_per_head_);
    typename AttentionOp::Params attn_params{qkv_buf_3_,
                                             q_buf_2_,
                                             k_cache_buf_,
                                             v_cache_buf_,
                                             attention_mask,
                                             qk_buf_float_,
                                             cu_seqlens,
                                             nullptr,
                                             nullptr,
                                             context_lengths,
                                             group_size,
                                             layout_q,
                                             layout_k,
                                             layout_v,
                                             layout_o};

    //
    flash_attention(attn_params, stream_);
}

template<typename T>
void LlamaContextAttentionLayer<T>::unfusedMultiHeadAttention(T**          key_cache_ptrs,
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
                                                              int          quant,
                                                              const float* kv_scale)
{
    // key_cache [B, kvH, S[:t+s], D/x, x] -> [B, qH, t+s, D]
    // val_cache [B, kvH, S[:t+s], D/x, x] -> [B, qH, t+s, D]
    invokeTransposeKVCache(k_cache_buf_,
                           v_cache_buf_,
                           (const T**)key_cache_ptrs,
                           (const T**)val_cache_ptrs,
                           cache_layer_offset,
                           batch_size,
                           context_length,  // history_len + input_len = context_len
                           max_k_len,
                           max_seq_len,
                           size_per_head_,
                           local_head_num_,
                           head_n_rep_,
                           stream_,
                           quant,
                           kv_scale);
    sync_check_cuda_error();

    const T qk_scale = static_cast<T>(1.f / sqrtf(size_per_head_ * 1.f));

    //////////////////////////////////////////////
    /// Q*K batch gemm
    /// -> [B, H, s, t + s]
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        max_k_len,                      // m
                                        max_q_len,                      // n
                                        size_per_head_,                 // k
                                        k_cache_buf_,                   // A
                                        size_per_head_,                 // lda
                                        max_k_len * size_per_head_,     // strideA
                                        q_buf_2_,                       // B
                                        size_per_head_,                 // ldb
                                        max_q_len * size_per_head_,     // strideB
                                        qk_buf_,                        // C
                                        max_k_len,                      // ldc
                                        max_q_len * max_k_len,          // strideC
                                        batch_size * local_head_num_);  // batchCount

    //////////////////////////////////////////////
    /// ! masked softmax (kernel asserts k_length <= 4096)
    MaskedSoftmaxParam<T, T> param{};
    param.attention_score    = qk_buf_;
    param.qk                 = qk_buf_;
    param.attention_mask     = attention_mask;
    param.batch_size         = batch_size;
    param.q_length           = max_q_len;
    param.k_length           = max_k_len;
    param.num_heads          = local_head_num_;
    param.qk_scale           = qk_scale;
    param.linear_bias_slopes = nullptr;
    invokeMaskedSoftmax(param, stream_);
    sync_check_cuda_error();

    //////////////////////////////////////////////
    /// softmax(QK)*V batch gemm
    // -> [B, H, S, D]
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        size_per_head_,                 // m
                                        max_q_len,                      // n
                                        max_k_len,                      // k
                                        v_cache_buf_,                   // A
                                        size_per_head_,                 // lda
                                        max_k_len * size_per_head_,     // strideA,
                                        qk_buf_,                        // B
                                        max_k_len,                      // ldb
                                        max_k_len * max_q_len,          // strideB
                                        qkv_buf_2_,                     // C
                                        size_per_head_,                 // ldc,
                                        max_q_len * size_per_head_,     // strideC
                                        batch_size * local_head_num_);  // batchCount

    //////////////////////////////////////////////
    /// transpose <B,h,s,D> -> <B,s,h,D>
    invokeTransposeAttentionOutRemovePadding(qkv_buf_2_,
                                             qkv_buf_3_,
                                             num_token,
                                             batch_size,
                                             max_q_len,
                                             local_head_num_,
                                             size_per_head_,
                                             padding_offset,
                                             nullptr,
                                             0,
                                             stream_);
    sync_check_cuda_error();
}

template class LlamaContextAttentionLayer<float>;
template class LlamaContextAttentionLayer<half>;

}  // namespace turbomind
