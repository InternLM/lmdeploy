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

#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/kernels/bert_preprocess_kernels.h"
#include "src/turbomind/kernels/decoder_multihead_attention/decoder_multihead_attention.h"
#include "src/turbomind/kernels/decoder_multihead_attention/kv_cache.h"
#include "src/turbomind/kernels/unfused_attention_kernels.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/debug_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

template<typename T>
// void UnifiedAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t num_token, size_t max_q_len, size_t
// max_k_len)
void UnifiedAttentionLayer<T>::allocateBuffer(size_t num_token,
                                              size_t pf_batch_size,
                                              size_t pf_max_q_len,
                                              size_t pf_max_k_len,
                                              size_t dc_batch_size,
                                              size_t dc_max_split_k)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    // no padding
    qkv_buf_ = (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * num_token * local_q_kv_head_num * size_per_head_, false);

    // qkv_buf_3_ padding is removed
    qkv_buf_3_ = (T*)allocator_->reMalloc(qkv_buf_3_, sizeof(T) * num_token * local_head_num_ * size_per_head_, false);

    if (pf_batch_size) {
        [&](size_t bsz, size_t max_q, size_t max_k) {
            // padding is rebuilt for q/k/v_buf_2_
            // [qH + 2kvH, B, S, D]
            q_buf_2_ = (T*)allocator_->reMalloc(
                q_buf_2_, sizeof(T) * local_q_kv_head_num * bsz * max_q * size_per_head_, false);
            k_buf_2_ = q_buf_2_ + local_head_num_ * bsz * max_q * size_per_head_;
            v_buf_2_ = k_buf_2_ + local_kv_head_num_ * bsz * max_q * size_per_head_;

            if (use_fmha_) {
                FlashAttentionOp<T> flash_attention(bsz, local_head_num_, max_k, max_q, size_per_head_);
                if (flash_attention.get_workspace_size() > 0) {
                    qk_buf_float_ =
                        (float*)allocator_->reMalloc(qk_buf_float_, flash_attention.get_workspace_size(), false);
                }
            }
            else {
                // kv heads are repeated for unfused attention
                k_cache_buf_ = (T*)allocator_->reMalloc(
                    k_cache_buf_, 2 * sizeof(T) * bsz * local_head_num_ * max_k * size_per_head_, false);
                v_cache_buf_ = k_cache_buf_ + bsz * local_head_num_ * max_k * size_per_head_;

                qk_buf_ = (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * bsz * local_head_num_ * max_q * max_k, false);

                // qkv_buf_2_ has padding
                qkv_buf_2_ = (T*)allocator_->reMalloc(
                    qkv_buf_2_, sizeof(T) * bsz * max_q * local_head_num_ * size_per_head_, false);
            }
        }(pf_batch_size, pf_max_q_len, pf_max_k_len);
    }

    if (dc_batch_size) {
        dc_workspace_ = (float*)allocator_->reMalloc(dc_workspace_,
                                                     sizeof(float) * dc_batch_size * local_head_num_ * dc_max_split_k
                                                         * (size_per_head_ + 2),
                                                     false);
    }

    is_allocate_buffer_ = true;
}

template<typename T>
void UnifiedAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        TM_LOG_DEBUG(__PRETTY_FUNCTION__);

        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&q_buf_2_));
        allocator_->free((void**)(&qkv_buf_3_));

        allocator_->free((void**)&qk_buf_float_);
        allocator_->free((void**)(&k_cache_buf_));
        allocator_->free((void**)(&qk_buf_));
        allocator_->free((void**)(&qkv_buf_2_));

        allocator_->free((void**)&dc_workspace_);

        is_allocate_buffer_ = false;
    }
}

template<typename T>
inline void UnifiedAttentionLayer<T>::forward(TensorMap* outputs, const TensorMap* inputs, const WeightType* weights)
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
     *   \param cu_block_counts [batch_size+1], int
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
    const int num_token   = inputs->at("input_query").shape[0];
    const int layer_id    = inputs->getVal<int>("layer_id");
    const int session_len = inputs->getVal<int>("session_len");

    int pf_batch_size = 0;
    int pf_max_q_len  = 0;
    int pf_max_k_len  = 0;
    T*  attention_mask{};
    if (inputs->isExist("attention_mask")) {
        pf_batch_size  = inputs->at("attention_mask").shape[0];
        pf_max_q_len   = inputs->at("attention_mask").shape[2];
        pf_max_k_len   = inputs->at("attention_mask").shape[3];
        attention_mask = inputs->getPtr<T>("attention_mask");
    }

    const int dc_batch_size  = inputs->getVal<int>("dc_batch_size");
    const int dc_sum_seq_len = inputs->getVal<int>("dc_sum_seq_len");
    const int dc_max_seq_len = inputs->getVal<int>("dc_max_seq_len");

    T*     attention_input = inputs->getPtr<T>("input_query");
    int*   input_length    = inputs->getPtr<int>("input_lengths");
    int*   context_length  = inputs->getPtr<int>("context_lengths");
    bool*  is_finished     = inputs->getPtr<bool>("finished");
    int*   cu_block_count  = inputs->getPtr<int>("cu_block_counts");
    int*   cu_seqlens      = inputs->getPtr<int>("cu_seqlens", nullptr);
    int*   padding_offset  = inputs->getPtr<int>("padding_offset", nullptr);
    float* rope_theta      = inputs->getPtr<float>("rope_theta", nullptr);

    auto k_cache_ptrs = outputs->getPtr<void*>("key_cache");
    auto v_cache_ptrs = outputs->getPtr<void*>("value_cache");
    auto tmp_k_ptrs   = outputs->getPtr<T*>("tmp_k");
    auto tmp_v_ptrs   = outputs->getPtr<T*>("tmp_v");

    T* attention_out = outputs->getPtr<T>("hidden_features");

    /////////////////////////////////////////////
    /// allocate buffers
    allocateBuffer(num_token,  //
                   pf_batch_size,
                   pf_max_q_len,
                   pf_max_k_len,
                   dc_batch_size,
                   kDecodeMaxSplits);

    // [2, L, H, s, D]
    const size_t layer_offset = layer_id * local_kv_head_num_ * kv_cache_block_len_ * size_per_head_;

    //////////////////////////////////////////////
    /// qkv gemm
    // [token_num, hidden_dim] -> [token_num, 3, local_hidden_dim]
    linear_.forward(qkv_buf_, attention_input, num_token, weights->qkv);

    if (pf_batch_size) {
        const int offset       = dc_batch_size;
        const int pf_num_token = num_token - offset;
        prefill(qkv_buf_3_ + offset * weights->output.input_dims,
                qkv_buf_ + offset * weights->qkv.output_dims,
                k_cache_ptrs,
                v_cache_ptrs,
                attention_mask,
                cu_seqlens,
                padding_offset,
                tmp_k_ptrs + offset,
                tmp_v_ptrs + offset,
                input_length + offset,
                context_length + offset,
                cu_block_count + offset,
                rope_theta + offset,
                pf_batch_size,
                pf_num_token,
                layer_offset,
                pf_max_q_len,
                pf_max_k_len,
                session_len,
                weights);
    }

    if (dc_batch_size) {
        decode(qkv_buf_3_,
               qkv_buf_,
               k_cache_ptrs,
               v_cache_ptrs,
               cu_block_count,
               context_length,
               is_finished,
               rope_theta,
               layer_offset,
               dc_batch_size,
               dc_sum_seq_len,
               dc_max_seq_len,
               kDecodeMaxSplits,
               weights);
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
void UnifiedAttentionLayer<T>::prefill(T*                output,
                                       const T*          qkv,
                                       void**            k_cache_ptrs,
                                       void**            v_cache_ptrs,
                                       const T*          attention_mask,
                                       const int*        cu_seqlens,
                                       const int*        padding_offset,
                                       T**               tmp_k_ptrs,
                                       T**               tmp_v_ptrs,
                                       const int*        input_length,
                                       const int*        context_length,
                                       const int*        cu_block_count,
                                       const float*      rope_theta,
                                       int               pf_batch_size,
                                       int               pf_num_token,
                                       size_t            layer_offset,
                                       int               pf_max_q_len,
                                       int               pf_max_k_len,
                                       int               pf_session_len,
                                       const WeightType* weights)
{
    //////////////////////////////////////////////
    /// transpose qkv & apply rotary embedding & rebuild padding
    /// qkv [B, s, H + 2kvH, D] -> (q [B, H, s, D], k [B, kvH, s, D], v [B, kvH, s, D])
    invokeAddFusedQKVBiasTranspose(q_buf_2_,
                                   k_buf_2_,
                                   v_buf_2_,
                                   (T*)qkv,
                                   weights->qkv.bias,
                                   padding_offset,  // padding_offset,
                                   context_length,  // used for applying rotary embedding
                                   input_length,
                                   rope_theta,
                                   pf_batch_size,
                                   pf_max_q_len,  // seq_len
                                   pf_num_token,
                                   local_head_num_,
                                   local_kv_head_num_,
                                   size_per_head_,
                                   params_.rotary_embedding_dim,
                                   params_.rotary_embedding_base,
                                   params_.max_position_embeddings,
                                   false,  // params_.use_dynamic_ntk,
                                   params_.use_logn_attn,
                                   stream_);
    sync_check_cuda_error();

    //////////////////////////////////////////////////////////
    /// insert the k/v computed from inputs into k/v cache
    /// transpose kv -> kv cache
    // put k/v_buf from shape [B, kvH, s, D] to
    // k_buf_2 [B, kvH, s, D] -> key_cache [B, kvH, S[t:t+s], D/x, x]
    // v_buf_2 [B, kvH, s, D] -> val_cache [B, kvH, S[t:t+s], D/x, x]
    invokeExtendKVCache(k_cache_ptrs,
                        v_cache_ptrs,
                        k_buf_2_,
                        v_buf_2_,
                        cu_block_count,
                        input_length,
                        context_length,
                        pf_batch_size,
                        kv_cache_block_len_,
                        layer_offset,
                        pf_max_q_len,
                        size_per_head_,
                        local_kv_head_num_,
                        quant_policy_,
                        weights->past_kv_scale.data(),
                        stream_);
    sync_check_cuda_error();

    const int kv_cache_elem_bits = quant_policy_ & QuantPolicy::kCacheKVInt8 ? 8 : sizeof(T) * 8;

    FT_CHECK(weights->past_kv_scale.size() == 4);
    ConvertKvCacheBlocksToLinear2((const void**)k_cache_ptrs,
                                  (const void**)v_cache_ptrs,
                                  (T**)tmp_k_ptrs,
                                  (T**)tmp_v_ptrs,
                                  cu_block_count,
                                  context_length,
                                  layer_offset,
                                  kv_cache_block_len_,
                                  pf_session_len,
                                  local_kv_head_num_,
                                  size_per_head_,
                                  pf_batch_size,
                                  quant_policy_,
                                  weights->past_kv_scale.data(),
                                  stream_);
    sync_check_cuda_error();

    if (use_fmha_) {
        fusedMultiHeadAttention(output,
                                q_buf_2_,
                                tmp_k_ptrs,
                                tmp_v_ptrs,
                                0,
                                (T*)attention_mask,
                                (int*)cu_seqlens,
                                (int*)context_length,
                                pf_batch_size,
                                pf_max_q_len,
                                pf_max_k_len,
                                pf_session_len);
    }
    else {
        unfusedMultiHeadAttention(output,
                                  q_buf_2_,
                                  tmp_k_ptrs,
                                  tmp_v_ptrs,
                                  0,
                                  attention_mask,
                                  padding_offset,
                                  context_length,
                                  pf_batch_size,
                                  pf_num_token,
                                  pf_max_q_len,
                                  pf_max_k_len,
                                  pf_session_len,
                                  quant_policy_,
                                  weights->past_kv_scale.data());
    }
}

template<typename T>
void UnifiedAttentionLayer<T>::decode(T*                output,
                                      const T*          qkv,
                                      void**            k_cache_ptrs,
                                      void**            v_cache_ptrs,
                                      const int*        cu_block_count,
                                      const int*        context_length,
                                      const bool*       is_finished,
                                      const float*      rope_theta,
                                      size_t            layer_offset,
                                      int               batch_size,
                                      int               dc_sum_seq_len,
                                      int               dc_max_seq_len,
                                      int               max_split_k,
                                      const WeightType* weights)
{
    DecoderMultiHeadAttentionParams<T> params{};

    params.out    = output;
    params.q      = (T*)qkv;
    params.k      = params.q + local_head_num_ * size_per_head_;
    params.v      = params.k + local_kv_head_num_ * size_per_head_;
    params.stride = (local_head_num_ + 2 * local_kv_head_num_) * size_per_head_;

    params.q_bias = weights->qkv.bias;
    params.k_bias = params.q_bias + local_head_num_ * size_per_head_;
    params.v_bias = params.k_bias + local_kv_head_num_ * size_per_head_;

    params.batch_size    = batch_size;
    params.cu_block_cnts = (int*)cu_block_count;

    params.k_cache_block_ptrs  = (void**)k_cache_ptrs;
    params.v_cache_block_ptrs  = (void**)v_cache_ptrs;
    params.kv_cache_block_size = kv_cache_block_len_;

    params.finished       = is_finished;
    params.context_length = context_length;
    params.rope_theta     = rope_theta;

    params.layer_offset = layer_offset;

    params.num_heads     = local_head_num_;
    params.num_kv_heads  = local_kv_head_num_;
    params.size_per_head = size_per_head_;
    params.inv_sqrt_dh   = 1.f / std::sqrt((float)params.size_per_head);

    params.rotary_embedding_dim    = size_per_head_;
    params.rotary_embedding_base   = params_.rotary_embedding_base;
    params.max_position_embeddings = params_.max_position_embeddings;
    params.use_dynamic_ntk = params_.use_dynamic_ntk;
    // when dynamic_ntk is false, `rope_scaling_factor` refers to the linear scaling factor if it is not 0
    params.scaling_factor = params_.rope_scaling_factor;
    params.use_logn_attn = params_.use_logn_attn;

    params.partial_O = dc_workspace_;
    params.partial_M = params.partial_O + batch_size * local_head_num_ * max_split_k * size_per_head_;
    params.partial_L = params.partial_M + batch_size * local_head_num_ * max_split_k;

    const float avg_batch_size = dc_max_seq_len ? (float)dc_sum_seq_len / dc_max_seq_len : 1;
    FT_CHECK(avg_batch_size >= 1.f);

    max_split_k = std::max(1, (int)std::ceil(max_split_k / avg_batch_size));

    params.max_split_k = max_split_k;
    params.max_seq_len = dc_max_seq_len;

    params.arch   = arch_;
    params.stream = stream_;

    params.quant_policy = quant_policy_;
    FT_CHECK(std::size(weights->past_kv_scale) == std::size(params.kv_quant_params));
    std::copy(weights->past_kv_scale.begin(), weights->past_kv_scale.end(), std::begin(params.kv_quant_params));

    {
        NvtxScope scope("decoder_multihead_attention");
        DispatchDecoderMultiheadAttention<T>(params);
    }
}

template<typename T>
void UnifiedAttentionLayer<T>::fusedMultiHeadAttention(T*       output,
                                                       const T* query,
                                                       T**      key_cache_ptrs,
                                                       T**      val_cache_ptrs,
                                                       size_t   cache_layer_offset,
                                                       T*       attention_mask,
                                                       int*     cu_seqlens,
                                                       int*     context_lengths,
                                                       int      batch_size,
                                                       int      max_q_len,
                                                       int      max_k_len,
                                                       int      max_seq_len)
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
    typename AttentionOp::Params attn_params{output,
                                             (T*)query,
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
void UnifiedAttentionLayer<T>::unfusedMultiHeadAttention(T*           output,
                                                         const T*     query,
                                                         T**          key_cache_ptrs,
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
                           0,  // dequant handled in block->linear conversion
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
                                        query,                          // B
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
                                             output,
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

template class UnifiedAttentionLayer<float>;
template class UnifiedAttentionLayer<half>;
#ifdef ENABLE_BF16
template class UnifiedAttentionLayer<__nv_bfloat16>;
#endif // ENABLE_BF16

}  // namespace turbomind
