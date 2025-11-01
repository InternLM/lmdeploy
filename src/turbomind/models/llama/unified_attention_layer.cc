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
#include <numeric>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"

#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/macro.h"

#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/mla_utils.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

UnifiedAttentionLayer::~UnifiedAttentionLayer()
{
    for (auto& s : streams_) {
        s = {};
    }

    check_cuda_error(cudaEventDestroy(aux_event_));
    check_cuda_error(cudaEventDestroy(qkv_event_));
    check_cuda_error(cudaStreamDestroy(aux_stream_));

    aux_event_ = qkv_event_ = {};
    aux_stream_             = {};
}

UnifiedAttentionLayer::UnifiedAttentionLayer(const ModelParam&     model,
                                             const AttentionParam& attn,
                                             const EngineParam&    engine,
                                             const LoraParam&      lora,
                                             int                   tp_size,
                                             const Context&        ctx):
    head_num_(model.head_num),
    kv_head_num_(model.kv_head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    local_head_num_(head_num_ / tp_size),
    local_kv_head_num_(model.kv_head_num / tp_size),
    param_(attn),
    model_param_(model),
    engine_param_(engine),
    attn_cp_group_(ctx.comm.d_cp_group),
    cp_fn_ctx_(ctx.comm.d_comm, ctx.comm.d_cp_group),
    d_comm_(ctx.comm.d_comm),
    lora_param_(lora),
    context_(ctx),
    stream_(ctx.stream),
    linear_(*ctx.linear),
    arch_(getSMVersion())
{
    TM_CHECK_EQ(head_num_ % tp_size, 0) << head_num_ << " " << tp_size;
    TM_CHECK_EQ(head_num_ % kv_head_num_, 0) << head_num_ << " " << kv_head_num_;

    check_cuda_error(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    check_cuda_error(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

    streams_[0] = stream_;
    streams_[1] = aux_stream_;

    init_rope_kernel_param(param_.rope, rope_param_);

    partial_M_ = Tensor_<float>({kMaxWorkspaceTokens, local_head_num_}, kDEVICE);
    partial_L_ = Tensor_<float>({kMaxWorkspaceTokens, local_head_num_}, kDEVICE);
    partial_O_ = Tensor_<float>({kMaxWorkspaceTokens, local_head_num_, size_per_head_}, kDEVICE);
    split_cnt_ = Tensor_<int>({kMaxWorkspaceTokens}, kDEVICE);
    barriers_  = Tensor_<int>({kMaxWorkspaceTokens, local_head_num_}, kDEVICE);

    if (engine_param_.attn_cp_size > 1) {
        const int cp_workspace_tokens = kMaxWorkspaceTokens + engine_param_.max_forward_token_num;
        cp_k_ML_                      = Tensor_<float>({cp_workspace_tokens, local_head_num_, 2}, kDEVICE);
    }

    Clear(split_cnt_.buffer());
    Clear(barriers_.buffer());

    const auto max_batch_size = engine.max_batch_size;

    d_cu_x_len_ = {2 * (max_batch_size + 1), kDEVICE};
    h_cu_x_len_ = {2 * (max_batch_size + 1), kCPUpinned};
    event_      = Event::create();
}

void UnifiedAttentionLayer::Initialize(TensorMap& args)
{
    h_q_len_ = args.at("h_q_len").buffer();
    h_k_len_ = args.at("h_k_len").buffer();

    const int bsz = h_q_len_.size();

    d_cu_q_len_ = d_cu_x_len_.data();
    h_cu_q_len_ = h_cu_x_len_.data();
    d_cu_k_len_ = d_cu_q_len_ + bsz + 1;
    h_cu_k_len_ = h_cu_q_len_ + bsz + 1;

    h_cu_q_len_[0] = h_cu_k_len_[0] = 0;

    std::inclusive_scan(h_q_len_.data(), h_q_len_.data() + bsz, h_cu_q_len_ + 1);
    std::inclusive_scan(h_k_len_.data(), h_k_len_.data() + bsz, h_cu_k_len_ + 1);

    Copy(h_cu_x_len_.slice(0, 2 * bsz + 2), d_cu_x_len_.slice(0, 2 * bsz + 2));

    event_.Record(core::Context::stream());

    decode_num_ = *args.at("decode_num").data<int>();
    prefil_num_ = *args.at("prefil_num").data<int>();

    finished_  = args.at("finished").buffer();
    rope_base_ = args.at("rope_base").buffer();

    cu_block_nums_ = args.at("cu_block_nums").buffer();
    kv_block_ptrs_ = args.at("kv_block_ptrs").buffer();

    if (engine_param_.attn_cp_size > 1) {
        cp_ML_ = args.at("cp_ML").borrow();
    }

    // rotary embedding, add offest when forward
    if (rope_param_.type == RopeType::kDynamic) {
        rope_param_.base = const_cast<float*>(rope_base_.data());
    }
    else if (rope_param_.type == RopeType::kMrope && !isTuning()) {
        auto& position_ids               = args.at("mrope_position_ids");
        rope_param_.mrope.stride         = position_ids.shape(1);
        rope_param_.mrope.position_ids   = position_ids.data<int>();
        rope_param_.mrope.position_delta = args.at("mrope_position_delta").data<int>();
        rope_param_.mrope.length         = args.at("mrope_position_length").data<int>();
    }
}

void UnifiedAttentionLayer::Finalize()
{
    event_.Sync();
}

void UnifiedAttentionLayer::Forward(ForwardParam p)
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
    const int token_num = p.input.shape(0);

    if (token_num == 0) {
        return;
    }

    const int layer_id = p.layer_id;

    const auto& weights = *p.weights;

    // [L, 2, H, s, D]
    const size_t layer_offset = layer_id * 2 * local_kv_head_num_ * param_.cache_block_seq_len * size_per_head_;

    Tensor qkv;

    if (weights.qkv.output_dim) {
        // [token_num, hidden_dim] -> [token_num, local_q_kv_head_num, head_dim]
        qkv = linear_.Forward(p.input, weights.qkv);
        sync_check_cuda_error();

        if (model_param_.qk_norm) {
            qk_norm(qkv, weights);
        }
    }
    else {
        qkv = forward_mla(p.input, weights);
    }

    TM_DEBUG_TENSOR(qkv, Concat("qkv", layer_id), 3);

    auto invoke = [&](auto t) -> Tensor {
        using T = decltype(t);
        return core_attention<T>(qkv, p, weights);
    };

    Tensor attn = [&]() -> Tensor { TM_DISPATCH_PRIMARY_DTYPES_RET(qkv.dtype(), invoke); }();

    TM_DEBUG_TENSOR(attn, Concat("attn", layer_id), 3);

    //////////////////////////////////////////////
    /// output gemm <Bs,HD> -> <Bs,HD>
    (void)linear_.Forward(attn, weights.output, p.output);
    sync_check_cuda_error();
}

template<class T>
Tensor UnifiedAttentionLayer::core_attention(Tensor& qkv, const ForwardParam& p, const WeightType& weights)
{
    const auto device = qkv.device();
    const auto dtype  = qkv.dtype();

    const int batch_size = decode_num_ + prefil_num_;
    const int q_count    = qkv.shape(0);
    const int k_count    = h_cu_k_len_[batch_size] - h_cu_k_len_[decode_num_];
    const int layer_id   = p.layer_id;

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    Tensor attn{{q_count, (int)local_head_num_ * (int)size_per_head_}, dtype, device};
    Tensor tmp_kv{{(int)local_kv_head_num_, 2, k_count + MAX_CTA_S, (int)size_per_head_}, dtype, device};

    auto stream_ptr = streams_.data();

    auto CreateParams = [&](int offset, int batch_size, int max_kv_splits, cudaStream_t stream) {
        AttentionParams<T> params{};

        // Batch offset for `out` and `q` are computed inside the kernel
        params.out = (T*)attn.raw_data();

        params.q      = (T*)qkv.raw_data();
        params.k      = params.q + local_head_num_ * size_per_head_;
        params.v      = params.k + local_kv_head_num_ * size_per_head_;
        params.stride = (local_head_num_ + 2 * local_kv_head_num_) * size_per_head_;

        if (weights.qkv.bias) {
            params.q_bias = (T*)weights.qkv.bias.data_or<T>(nullptr);
            params.k_bias = params.q_bias + local_head_num_ * size_per_head_;
            params.v_bias = params.k_bias + local_kv_head_num_ * size_per_head_;
        }

        params.token_num  = h_cu_q_len_[offset + batch_size] - h_cu_q_len_[offset];
        params.batch_size = batch_size;
        /// TODO: maximum on buffer slice
        params.max_q_len = *std::max_element(h_q_len_.data() + offset, h_q_len_.data() + offset + batch_size);
        params.max_k_len = *std::max_element(h_k_len_.data() + offset, h_k_len_.data() + offset + batch_size);

        // Decoding use only
        params.block_iter_params = BlockIteratorParams{(char**)kv_block_ptrs_.data(),  //
                                                       cu_block_nums_.data() + offset,
                                                       layer_id,
                                                       (int)param_.cache_block_seq_len};

        // Prefilling use only
        const int sum_k_len       = h_cu_k_len_[offset + prefil_num_] - h_cu_k_len_[offset];
        params.linear_iter_params = LinearIteratorParams{tmp_kv.raw_data(),  //
                                                         int(2 * sum_k_len * size_per_head_),
                                                         int(sum_k_len * size_per_head_)};

        params.finished = finished_.data() + offset;
        params.cu_q_len = d_cu_q_len_ + offset;
        params.cu_k_len = d_cu_k_len_ + offset;

        params.num_heads     = local_head_num_;
        params.num_kv_heads  = local_kv_head_num_;
        params.size_per_head = size_per_head_;

        double scaling = 1.;
        if (param_.softmax_scale) {  // model predefined softmax scale
            scaling *= param_.softmax_scale;
        }
        else {  // default value
            scaling /= std::sqrt((float)params.size_per_head);
        }
        params.inv_sqrt_dh = scaling * std::log2(std::exp(1.));

        params.sinks       = weights.sinks.data_or((T*)nullptr);
        params.scale_sinks = scaling;

        params.window_size = weights.window_size;
        if (!params.window_size) {
            params.window_size = 256 << 20;  // 256 M
        }

        // add offset to rope
        params.rope_param = rope_param_;
        if (rope_param_.type == RopeType::kDynamic) {
            params.rope_param.base += offset;
        }
        else if (rope_param_.type == RopeType::kMrope) {
            params.rope_param.mrope.position_ids += offset * rope_param_.mrope.stride;
            params.rope_param.mrope.position_delta += offset;
            params.rope_param.mrope.length += offset;
        }

        // logn attn
        params.use_logn_attn           = param_.use_logn_attn;
        params.max_position_embeddings = param_.max_position_embeddings;

        // Decoding use only for now
        params.split_cnt   = split_cnt_.data();
        params.partial_L   = partial_L_.data();
        params.partial_M   = partial_M_.data();
        params.partial_O   = partial_O_.data();
        params.locks       = barriers_.data();
        params.max_split_k = std::min(std::max(1, kMaxWorkspaceTokens / params.token_num), max_kv_splits);

        // context parallel
        params.cp_rank = engine_param_.attn_cp_rank;
        params.cp_size = engine_param_.attn_cp_size;
        if (params.cp_size > 1) {
            params.cp_divmod = cutlass::FastDivmod(params.cp_size);

            const int offset_ML = engine_param_.attn_cp_size * offset * local_head_num_ * 2;
            params.cp_ML        = cp_ML_.data() + offset_ML + params.cp_rank * params.token_num * local_head_num_ * 2;
            params.cp_k_ML      = cp_k_ML_.data() + (offset ? kMaxWorkspaceTokens * local_head_num_ * 2 : 0);
            params.cp_q_offset  = offset;

            // postprocess func
            params.cp_fn     = CpPost;
            params.cp_fn_ctx = (void*)&cp_fn_ctx_;

            cp_fn_ctx_.cp_ML      = cp_ML_.data() + offset_ML;
            cp_fn_ctx_.attn_param = (void*)&params;
            cp_fn_ctx_.attn_type  = attn.dtype();
        }

        params.arch   = arch_;
        params.stream = stream;

        params.quant_policy = model_param_.quant_policy;
        return params;
    };

    cudaStream_t pf_stream = stream_;
    cudaStream_t dc_stream = stream_;

    if (decode_num_ && prefil_num_) {
        pf_stream = aux_stream_;
        check_cuda_error(cudaEventRecord(qkv_event_, stream_));
        check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));
    }

    if (prefil_num_ && !isTuning()) {
        const int offset    = decode_num_;
        const int sum_k_len = h_cu_k_len_[offset + prefil_num_] - h_cu_k_len_[offset];
        // We are executing prefill & decoding kernels concurrently, but only have 1 workspace
        // disable split kv for prefill for now
        auto params = CreateParams(offset, prefil_num_, 1, pf_stream);
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

    if (decode_num_ && !isTuning()) {
        auto params = CreateParams(0, decode_num_, kMaxKVSplits, dc_stream);
        if constexpr (sizeof(T) == 2) {
            dispatchDecoding<T>(params);
            sync_check_cuda_error();
        }
    }

    if (decode_num_ && prefil_num_) {
        check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
        check_cuda_error(cudaStreamWaitEvent(stream_, aux_event_));
    }

    if (isTuning()) {
        rng_.set_stream(stream_);
        rng_.GenerateUniform(attn.data<T>(), attn.size(), .02f, -.01f);
    }

    return attn;
}

Tensor UnifiedAttentionLayer::forward_mla(const Tensor& hidden_state, const WeightType& w)
{
    const int q_lora_rank  = w.q_a_proj.output_dim;
    const int kv_lora_rank = w.kv_b_proj.input_dim;
    const int qk_rope_dim  = w.kv_a_proj.output_dim - kv_lora_rank;
    const int qk_nope_dim  = std::max(w.q_b_proj.output_dim, w.q_proj.output_dim) / local_head_num_ - qk_rope_dim;
    const int v_head_dim   = w.kv_b_proj.output_dim / local_head_num_ - qk_nope_dim;

    const auto token_num = hidden_state.shape(0);
    const auto dtype     = hidden_state.dtype();

    Tensor q;

    if (w.q_proj.weight) {
        q = linear_.Forward(hidden_state, w.q_proj);
        sync_check_cuda_error();
    }
    else {
        Tensor q_a = linear_.Forward(hidden_state, w.q_a_proj);
        sync_check_cuda_error();

        invokeRMSNorm(q_a, q_a, w.q_a_layernorm, model_param_.norm_eps, stream_);
        sync_check_cuda_error();

        q = linear_.Forward(q_a, w.q_b_proj);
        sync_check_cuda_error();
    }

    Tensor kv_a_k_pe = linear_.Forward(hidden_state, w.kv_a_proj);
    sync_check_cuda_error();

    auto kv_a = kv_a_k_pe.slice({0, 0}, {-1, kv_lora_rank});
    invokeRMSNorm(kv_a, kv_a, w.kv_a_layernorm, model_param_.norm_eps, stream_);
    sync_check_cuda_error();

    Tensor kv_b = linear_.Forward(kv_a, w.kv_b_proj);
    sync_check_cuda_error();

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    Tensor qkv{{token_num, local_q_kv_head_num, (int)size_per_head_}, dtype, hidden_state.device()};
    MLACopyQKV(dtype,
               qkv.raw_data(),
               q.raw_data(),
               kv_a.raw_data(),
               kv_b.raw_data(),
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

void UnifiedAttentionLayer::qk_norm(Tensor& qkv, const WeightType& weights)
{
    check_cuda_error(cudaEventRecord(qkv_event_, stream_));
    check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));

    TM_CHECK(model_param_.attn_bias == false) << "not implemented";

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

}  // namespace turbomind
