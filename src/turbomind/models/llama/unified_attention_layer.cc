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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.cc

#include <algorithm>
#include <math.h>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/mla_utils.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<class T>
UnifiedAttentionLayer<T>::UnifiedAttentionLayer(
    const ModelParam& model, const AttentionParam& attn, const LoraParam& lora, size_t tp_size, const Context<T>& ctx):
    head_num_(model.head_num),
    kv_head_num_(model.kv_head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    local_head_num_(head_num_ / tp_size),
    local_kv_head_num_(model.kv_head_num / tp_size),
    param_(attn),
    model_param_(model),
    lora_param_(lora),
    context_(ctx),
    stream_(ctx.stream),
    linear_(ctx.linear.get()),
    allocator_(ctx.allocator.get()),
    arch_(getSMVersion())
{
    TM_CHECK(head_num_ % kv_head_num_ == 0);

    check_cuda_error(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    check_cuda_error(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

    streams_[0] = stream_;
    streams_[1] = aux_stream_;

    init_rope_kernel_param(param_.rope, rope_param_);

    allocateWorkspace();
}

template<typename T>
void UnifiedAttentionLayer<T>::allocateWorkspace()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TM_CHECK(!is_allocate_workspace_);
    partial_M_ = (float*)allocator_->malloc(sizeof(float) * kMaxWorkspaceTokens * local_head_num_);
    partial_L_ = (float*)allocator_->malloc(sizeof(float) * kMaxWorkspaceTokens * local_head_num_);
    partial_O_ = (float*)allocator_->malloc(sizeof(float) * kMaxWorkspaceTokens * local_head_num_ * size_per_head_);
    split_cnt_ = (int*)allocator_->malloc(sizeof(int) * kMaxWorkspaceTokens);
    barriers_  = (int*)allocator_->malloc(sizeof(int) * kMaxWorkspaceTokens * local_head_num_, true, false);
    is_allocate_workspace_ = true;
}

template<typename T>
void UnifiedAttentionLayer<T>::freeWorkspace()
{
    if (is_allocate_workspace_) {
        TM_LOG_DEBUG(__PRETTY_FUNCTION__);

        allocator_->free((void**)&partial_M_);
        allocator_->free((void**)&partial_L_);
        allocator_->free((void**)&partial_O_);
        allocator_->free((void**)&split_cnt_);
        allocator_->free((void**)&barriers_);

        is_allocate_workspace_ = false;
    }
}

template<typename T>
inline void UnifiedAttentionLayer<T>::forward(ForwardParam&& param)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    /**
     * input_tensors:
     *   \param input_query [token_num, hidden_dim]
     *   \param cu_q_len [batch_size+1], int
     *   \param cu_k_len [batch_size+1], int
     *   \param cu_block_counts [batch_size+1], int
     *   \param finished [batch_size], bool
     *   \param rope_theta [batch_size], float
     *   \param h_q_len [batch_size], int on cpu
     *   \param h_k_len [batch_size], int on cpu
     *   \param h_cu_q_len [batch_size+1], int on cpu
     *   \param h_cu_k_len [batch_size+1], int on cpu
     *   \param dc_batch_size [1], int on cpu
     *   \param pf_batch_size [1], int on cpu
     *   \param layer_id [1], int on cpu
     *
     * output_tensors:
     *   \param hidden_features [token_num, hidden_dim], float
     *   \param block_ptrs [total_block_counts], void*
     */

    /////////////////////////////////////////////
    /// parse inputs
    const int token_num = param.input.shape(0);

    if (token_num == 0) {
        return;
    }

    const int layer_id = param.layer_id;

    const int dc_batch_size = param.decode_num;
    const int pf_batch_size = param.prefil_num;
    const int batch_size    = dc_batch_size + pf_batch_size;

    const auto weights = param.weights;

    const auto device = param.input.device();
    const auto dtype  = getTensorType<T>();

    int* h_q_len    = param.h_q_len;
    int* h_k_len    = param.h_k_len;
    int* cu_q_len   = param.cu_q_len;
    int* cu_k_len   = param.cu_k_len;
    int* h_cu_q_len = param.h_cu_q_len;
    int* h_cu_k_len = param.h_cu_k_len;

    bool*  is_finished = param.is_finished;
    float* rope_theta  = param.rope_base;

    void** block_ptrs     = param.block_ptrs;
    int*   cu_block_count = param.cu_block_count;

    const int q_count = token_num;
    const int k_count = h_cu_k_len[batch_size] - h_cu_k_len[dc_batch_size];

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    // [L, 2, H, s, D]
    const size_t layer_offset = layer_id * 2 * local_kv_head_num_ * param_.cache_block_seq_len * size_per_head_;

    core::Tensor qkv;

    if (weights->qkv.output_dims) {
        //////////////////////////////////////////////
        /// qkv gemm
        // [token_num, hidden_dim] -> [token_num, local_q_kv_head_num, head_dim]
        qkv = linear_->forward(param.input, weights->qkv, LlamaLinear<T>::kGemm);
        sync_check_cuda_error();

        if (model_param_.qk_norm) {
            qk_norm(qkv, *weights);
        }
    }
    else {
        qkv = forward_mla(param.input, *weights);
    }

    count_and_fix(qkv.data<T>(), qkv.size(), Concat("qkv", layer_id), 3);

    core::Tensor attn{{q_count, (int)local_head_num_, (int)size_per_head_}, dtype, device};
    core::Tensor tmp_kv{{2, (int)local_kv_head_num_, k_count + MAX_CTA_S, (int)size_per_head_}, dtype, device};

    auto stream_ptr = streams_.data();

    auto CreateParams = [&](int offset, int batch_size, int max_kv_splits, cudaStream_t stream) {
        AttentionParams<T> params{};

        // Batch offset for `out` and `q` are computed inside the kernel
        params.out = (T*)attn.raw_data();

        params.q      = (T*)qkv.raw_data();
        params.k      = params.q + local_head_num_ * size_per_head_;
        params.v      = params.k + local_kv_head_num_ * size_per_head_;
        params.stride = (local_head_num_ + 2 * local_kv_head_num_) * size_per_head_;

        if (weights->qkv.bias) {
            params.q_bias = weights->qkv.bias;
            params.k_bias = params.q_bias + local_head_num_ * size_per_head_;
            params.v_bias = params.k_bias + local_kv_head_num_ * size_per_head_;
        }

        params.token_num  = h_cu_q_len[offset + batch_size] - h_cu_q_len[offset];
        params.batch_size = batch_size;
        params.max_q_len  = *std::max_element(h_q_len + offset, h_q_len + offset + batch_size);
        params.max_k_len  = *std::max_element(h_k_len + offset, h_k_len + offset + batch_size);

        // Decoding use only
        params.block_iter_params = BlockIteratorParams{(char**)block_ptrs,  //
                                                       (int*)cu_block_count + offset,
                                                       layer_id,
                                                       (int)param_.cache_block_seq_len};

        // Prefilling use only
        const int sum_k_len       = h_cu_k_len[offset + pf_batch_size] - h_cu_k_len[offset];
        params.linear_iter_params = LinearIteratorParams{tmp_kv.raw_data(),  //
                                                         int(2 * sum_k_len * size_per_head_),
                                                         int(sum_k_len * size_per_head_)};

        params.finished = is_finished + offset;
        params.cu_q_len = cu_q_len + offset;
        params.cu_k_len = cu_k_len + offset;

        params.num_heads     = local_head_num_;
        params.num_kv_heads  = local_kv_head_num_;
        params.size_per_head = size_per_head_;

        // MSVC does not have M_LOG2E
        params.inv_sqrt_dh = (float)std::log2(expf(1.));
        if (param_.softmax_scale) {  // model predefined softmax scale
            params.inv_sqrt_dh *= param_.softmax_scale;
        }
        else {  // default value
            params.inv_sqrt_dh /= std::sqrt((float)params.size_per_head);
        }

        // rotary embedding
        if (rope_param_.type == RopeType::kDynamic) {
            rope_param_.base = rope_theta + offset;
        }
        params.rope_param = rope_param_;

        // logn attn
        params.use_logn_attn           = param_.use_logn_attn;
        params.max_position_embeddings = param_.max_position_embeddings;

        // Decoding use only for now
        TM_CHECK_NOTNULL(barriers_);
        params.split_cnt   = split_cnt_;
        params.partial_L   = partial_L_;
        params.partial_M   = partial_M_;
        params.partial_O   = partial_O_;
        params.locks       = barriers_;
        params.max_split_k = std::min(std::max(1, kMaxWorkspaceTokens / params.token_num), max_kv_splits);

        params.arch   = arch_;
        params.stream = stream;

        params.quant_policy = model_param_.quant_policy;
        return params;
    };

    cudaStream_t pf_stream = stream_;
    cudaStream_t dc_stream = stream_;

    if (pf_batch_size && dc_batch_size) {
        pf_stream = aux_stream_;
        check_cuda_error(cudaEventRecord(qkv_event_, stream_));
        check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));
    }

    if (pf_batch_size && !isTuning()) {
        const int offset    = dc_batch_size;
        const int sum_k_len = h_cu_k_len[offset + pf_batch_size] - h_cu_k_len[offset];
        // We are executing prefill & decoding kernels concurrently, but only have 1 workspace
        // disable split kv for prefill for now
        auto params = CreateParams(offset, pf_batch_size, 1, pf_stream);
        if constexpr (sizeof(T) == 2) {
            invokeProcessKV_v2_(params);
            sync_check_cuda_error();

            /// TODO: skip flattening for `sm_80`
            invokeFlattenKV_v2_(params, sum_k_len);
            sync_check_cuda_error();

            dispatchAttention(params);
            sync_check_cuda_error();
        }
    }

    if (dc_batch_size && !isTuning()) {
        auto params = CreateParams(0, dc_batch_size, kMaxKVSplits, dc_stream);
        if constexpr (sizeof(T) == 2) {
            dispatchDecoding<T>(params);
            sync_check_cuda_error();
        }
    }

    if (pf_batch_size && dc_batch_size) {
        check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
        check_cuda_error(cudaStreamWaitEvent(stream_, aux_event_));
    }

    if (isTuning()) {
        rng_.set_stream(stream_);
        rng_.GenerateUniform(attn.data<T>(), token_num * weights->output.input_dims, .02f, -.01f);
    }

    count_and_fix(attn.data<T>(), attn.size(), Concat("attn", layer_id), 3);

    //////////////////////////////////////////////
    /// output gemm <Bs,HD> -> <Bs,HD>
    (void)linear_->forward(attn.view({q_count, -1}), weights->output, LlamaLinear<T>::kGemm, &param.output);
    sync_check_cuda_error();

    count_and_fix((T*)param.output.raw_data(), param.output.size(), Concat("wo", layer_id), 3);
}

template<typename T>
core::Tensor UnifiedAttentionLayer<T>::forward_mla(const core::Tensor& hidden_state, const WeightType& w)
{
    const int q_lora_rank  = w.q_a_proj.output_dims;
    const int kv_lora_rank = w.kv_b_proj.input_dims;
    const int qk_rope_dim  = w.kv_a_proj.output_dims - kv_lora_rank;
    const int qk_nope_dim  = std::max(w.q_b_proj.output_dims, w.q_proj.output_dims) / local_head_num_ - qk_rope_dim;
    const int v_head_dim   = w.kv_b_proj.output_dims / local_head_num_ - qk_nope_dim;

    const auto token_num = hidden_state.shape(0);
    const auto dtype     = getTensorType<T>();

    core::Tensor q;

    if (w.q_proj.kernel) {
        q = linear_->forward(hidden_state, w.q_proj);
        sync_check_cuda_error();
    }
    else {
        core::Tensor q_a = linear_->forward(hidden_state, w.q_a_proj);
        sync_check_cuda_error();

        invokeRMSNorm(q_a, q_a, w.q_a_layernorm, model_param_.norm_eps, stream_);
        sync_check_cuda_error();

        q = linear_->forward(q_a, w.q_b_proj);
        sync_check_cuda_error();
    }

    core::Tensor kv_a = linear_->forward(hidden_state, w.kv_a_proj);
    sync_check_cuda_error();

    invokeRMSNorm(kv_a, kv_a, w.kv_a_layernorm, model_param_.norm_eps, stream_);
    sync_check_cuda_error();

    core::Tensor kv_b = linear_->forward(kv_a.slice({0, 0}, {token_num, kv_lora_rank}), w.kv_b_proj);
    sync_check_cuda_error();

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    core::Tensor qkv{{token_num, local_q_kv_head_num, (int)size_per_head_}, dtype, hidden_state.device()};
    dispatchMLACopyQKV(qkv.data<T>(),
                       q.data<T>(),
                       kv_a.data<T>(),
                       kv_b.data<T>(),
                       token_num,
                       local_head_num_,
                       qk_nope_dim,
                       qk_rope_dim,
                       kv_lora_rank,
                       v_head_dim,
                       stream_);
    sync_check_cuda_error();

    return qkv;
}

template<typename T>
void UnifiedAttentionLayer<T>::qk_norm(core::Tensor& qkv, const WeightType& weights)
{
    check_cuda_error(cudaEventRecord(qkv_event_, stream_));
    check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));

    FT_CHECK(model_param_.attn_bias == false);

    const auto token_num = qkv.shape(0);

    auto qkv3 = qkv.view({token_num, -1, (int)size_per_head_});

    auto q = qkv3.slice({0, 0, 0}, {-1, (int)local_head_num_, -1});
    invokeRMSNormQK(q, weights.q_a_layernorm, model_param_.norm_eps, stream_);
    sync_check_cuda_error();

    auto k = qkv3.slice({0, (int)local_head_num_, 0}, {-1, (int)local_kv_head_num_, -1});
    invokeRMSNormQK(k, weights.kv_a_layernorm, model_param_.norm_eps, aux_stream_);
    sync_check_cuda_error();

    check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
    check_cuda_error(cudaStreamWaitEvent(stream_, aux_event_));
}

#ifdef ENABLE_FP32
template class UnifiedAttentionLayer<float>;
#endif
template class UnifiedAttentionLayer<half>;
#ifdef ENABLE_BF16
template class UnifiedAttentionLayer<__nv_bfloat16>;
#endif  // ENABLE_BF16

}  // namespace turbomind
