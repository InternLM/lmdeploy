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
#include <functional>
#include <math.h>
#include <numeric>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/macro.h"

#include "src/turbomind/models/llama/llama_rope.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/mla_utils.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

// #include "dbg.h"

namespace turbomind {

struct AttentionData {
    struct Stat {
        int n;
        int q_sum;
        int q_max;
        int k_sum;
        int k_max;
    } decode, prefill;

    Buffer_<void*> block_ptrs;
    Buffer_<int>   block_ptrs_offsets;

    Buffer_<float> rope_base;

    Tensor_<int> mrope_position_ids;
    Buffer_<int> mrope_position_delta;
    Buffer_<int> mrope_length;

    // borrowed from env
    Buffer_<bool> finished;
    Buffer_<int>  q_offsets;
    Buffer_<int>  k_offsets;

    // int dbg_offset;
    // int dbg_size;
};

UnifiedAttentionLayer::~UnifiedAttentionLayer()
{

    check_cuda_error(cudaEventDestroy(aux_event_));
    check_cuda_error(cudaEventDestroy(qkv_event_));
    check_cuda_error(cudaStreamDestroy(aux_stream_));

    aux_event_ = qkv_event_ = {};
    aux_stream_             = {};
}

UnifiedAttentionLayer::UnifiedAttentionLayer(const ModelParam&     model,
                                             const AttentionParam& attn,
                                             const EngineParam&    engine,
                                             int                   tp_size,
                                             const Context&        ctx,
                                             int                   phases,
                                             bool                  init):
    head_num_(model.head_num),
    kv_head_num_(model.kv_head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    local_head_num_(head_num_ / tp_size),
    local_kv_head_num_(model.kv_head_num / tp_size),
    param_(attn),
    model_param_(model),
    engine_param_(engine),
    cp_fn_ctx_(ctx.comm.d_comm, ctx.comm.d_cp_group),
    is_warm_up_{*ctx.is_warm_up},
    context_(ctx),
    linear_(*ctx.linear),
    arch_(getSMVersion())
{
    TM_CHECK_EQ(head_num_ % tp_size, 0) << head_num_ << " " << tp_size;
    TM_CHECK_EQ(head_num_ % kv_head_num_, 0) << head_num_ << " " << kv_head_num_;

    check_cuda_error(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    check_cuda_error(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

    init_rope_kernel_param(param_.rope, rope_param_);

    Allocator alloc            = core::Context::device_alloc();
    ssize_t   workspace_tokens = kMaxWorkspaceTokens;
    if (engine_param_.attn_cp_size > 1) {
        alloc = GetSymmAllocator(ctx.comm.d_comm);
        workspace_tokens += engine_param_.max_forward_token_num;
    }
    // partial_O layout:
    //   w/  cp, decode(q, h, k, 2) + prefill(q, h, 1, 2)
    //   w/o cp, decode(q, h, k, 2)
    partial_O_  = Tensor_<float>({workspace_tokens, local_head_num_, size_per_head_}, kDEVICE);
    partial_ML_ = Tensor_<float>({engine_param_.attn_cp_size, workspace_tokens, local_head_num_, 2}, alloc);
    split_cnt_  = Tensor_<int>({workspace_tokens}, kDEVICE);
    if (init) {
        const int dim = (int)local_head_num_ * (int)size_per_head_;
        tmp_attn_     = Tensor{{engine_param_.max_forward_token_num, dim}, model.data_type, kDEVICE};
    }

    Clear(split_cnt_.buffer());

    const int bsz = engine.max_batch_size;

    if (rope_param_.type == RopeType::kDynamic) {
        rope_base_buf_ = {bsz + 1, kCPUpinned};
    }
    else if (rope_param_.type == RopeType::kMrope) {
        // `mrope_position_ids` is not buffered
        mrope_position_delta_buf_ = {bsz, kCPUpinned};
        mrope_length_buf_         = {bsz, kCPUpinned};
    }
    const int max_blocks = bsz * cdiv(engine.session_len, param_.cache_block_seq_len);
    for (int i = 0; i < phases; ++i) {
        auto& d               = data_.emplace_back(std::make_shared<AttentionData>());
        d->block_ptrs         = {max_blocks + 16, kDEVICE};
        d->block_ptrs_offsets = {bsz + 1, kDEVICE};
        if (rope_param_.type == RopeType::kDynamic) {
            d->rope_base = empty_like(rope_base_buf_, kDEVICE);
        }
        else if (rope_param_.type == RopeType::kMrope) {
            /// TODO: total space for `mrope_position_ids` can be reduced to (max_fwd_tokens, 3)
            d->mrope_position_ids    = {{bsz, engine.session_len, 3}, kDEVICE};
            d->mrope_position_delta  = empty_like(mrope_position_delta_buf_, kDEVICE);
            d->mrope_length          = empty_like(mrope_length_buf_, kDEVICE);
            rope_param_.mrope.stride = d->mrope_position_ids.stride(0);
        }
    }
}

static void init_dynamic_ntk(RequestCache& cache, const RopeParam& rope)
{
    cache.rope_base = rope.base;
    if (auto scaling_factor = rope.factor; scaling_factor > 1.f) {
        const auto max_seq_len = cache.prompt_len;
        const auto max_pos_emb = rope.max_position_embeddings;
        if (max_seq_len > max_pos_emb) {
            scaling_factor = scaling_factor * max_seq_len / max_pos_emb - (scaling_factor - 1);
            cache.rope_base *= powf(scaling_factor, rope.dim / (rope.dim - 2.f));
            // clang-format off
            TM_LOG_INFO("[ProcessInferRequests] %ld rope_scaling_factor: %f, rope_theta = %f",
                        (long)cache.req->id, scaling_factor, cache.rope_base);
            // clang-format on
        }
    }
}

void UnifiedAttentionLayer::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kAdd) {
        Buffer_<RequestCache*> rc = env.at("requests").buffer();
        if (rope_param_.type == RopeType::kDynamic) {
            for (int i = 0; i < rc.size(); ++i) {
                init_dynamic_ntk(*rc[i], param_.rope);
            }
        }
    }
    else if (op == BatchOp::kSetup) {
        Setup(phase, env);
    }
    else if (op == BatchOp::kPrepare) {
        data_.at(phase)->finished  = env.at("finished").buffer().borrow();
        data_.at(phase)->q_offsets = env.at("q_offsets").buffer().borrow();
        data_.at(phase)->k_offsets = env.at("k_offsets").buffer().borrow();

        // This is needed in async mode to clear the `attn` buffer for the finished sequences. Ohterwise random NaNs
        // will crash the MoE router later
        /// TODO: use better solution, this increase memory usage and heterogenous attention layers may still break it
        if (tmp_attn_) {
            auto& d = data_.at(phase);
            Clear(tmp_attn_.slice(0, d->decode.n + d->prefill.q_sum));
            Clear(split_cnt_);
        }
    }
}

void UnifiedAttentionLayer::Setup(int phase, TensorMap& env)
{
    const auto& rc  = env.at("batch").data<BatchData*>()[0]->rc;
    const int   bsz = rc.size();

    auto& d    = *data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    {  /// Upload KV cache ptrs
        const Buffer_<int> offsets = env.at("block_ptrs_offsets").buffer();
        copy(env.at("block_ptrs").buffer(), offsets[bsz], d.block_ptrs);
        copy(offsets, bsz + 1, d.block_ptrs_offsets);
    }

    /// prepare Q/K stats for decode/prefill
    d.decode = d.prefill = {};

    d.decode.n  = std::find_if(rc.begin(), rc.end(), [](auto r) { return r->input_len > 1; }) - rc.begin();
    d.prefill.n = bsz - d.decode.n;

    // d.dbg_offset = d.dbg_size = 0;

    for (int i = 0; i < bsz; ++i) {
        const auto& c = *rc[i];

        // if (c.request->id == 4 && c.input_len > 1) {
        //     d.dbg_offset = d.decode.q_sum + d.prefill.q_sum;
        //     d.dbg_size   = c.input_len;
        // }

        auto& s = i < d.decode.n ? d.decode : d.prefill;
        s.q_sum += c.input_len;
        s.k_sum += c.history_len + c.alpha + c.input_len;
        s.q_max = std::max(s.q_max, c.input_len);
        s.k_max = std::max(s.k_max, c.history_len + c.alpha + c.input_len);
    }

    // auto &D = d.decode, &P = d.prefill;
    // dbg(D.n, D.k_sum, D.k_max, P.n, P.q_sum, P.q_max, P.k_sum, P.k_max);

    /// handling different RoPE types
    if (rope_param_.type == RopeType::kDynamic) {
        for (int i = 0; i < bsz; ++i) {
            rope_base_buf_[i] = rc[i]->rope_base;
        }
        copy(rope_base_buf_, bsz, d.rope_base);
    }
    else if (rope_param_.type == RopeType::kMrope) {
        const auto stride = d.mrope_position_ids.stride(0);
        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *rc[i];
            auto& r = *c.req;
            if (auto pos_ids = r.inputs.try_("mrope_position_ids")) {
                int length                   = pos_ids->shape(0);
                mrope_length_buf_[i]         = length;
                mrope_position_delta_buf_[i] = *r.inputs.at("mrope_position_delta").data<int>();
                if (auto o = Interval{0, length} & Interval{c.history_len + c.alpha, Interval::Size{c.input_len}}) {
                    copy(pos_ids->data<int>() + o.begin() * 3,
                         (int)o.size() * 3,
                         d.mrope_position_ids.data() + i * stride + o.begin() * 3);
                }
            }
            else {
                mrope_length_buf_[i] = mrope_position_delta_buf_[i] = 0;
            }
        }
        copy(mrope_length_buf_, rc.size(), d.mrope_length);
        copy(mrope_position_delta_buf_, rc.size(), d.mrope_position_delta);
    }
}

void UnifiedAttentionLayer::Forward(ForwardParam p)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

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

    auto& d = *data_.at(p.phase);

    // if (d.dbg_size) {
    //     DebugTensor(p.input.slice(d.dbg_offset, d.dbg_size), Concat("attn_in", p.layer_id), 0);
    // }

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

    // if (d.dbg_size) {
    //     DebugTensor(attn.slice(d.dbg_offset, d.dbg_size), Concat("attn_out", p.layer_id), 0);
    // }

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

    auto& d = *data_.at(p.phase);

    const int batch_size = d.decode.n + d.prefill.n;
    const int q_count    = qkv.shape(0);

    TM_CHECK_EQ(d.prefill.q_sum + d.decode.n, q_count);

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    Tensor attn;
    if (tmp_attn_) {
        attn = tmp_attn_.slice(0, q_count);
    }
    else {
        attn = {{q_count, (int)local_head_num_ * (int)size_per_head_}, dtype, device};
    }
    Tensor tmp_kv{{(int)local_kv_head_num_, 2, d.prefill.k_sum + MAX_CTA_S, (int)size_per_head_}, dtype, device};

    auto CreateParams = [&](int offset, AttentionData::Stat stat, int max_kv_splits, cudaStream_t stream) {
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

        params.batch_size = stat.n;

        params.token_num = stat.q_sum;
        params.max_q_len = stat.q_max;
        params.max_k_len = stat.k_max;

        // decode only
        params.block_iter_params = BlockIteratorParams{(char**)d.block_ptrs.data(),  //
                                                       d.block_ptrs_offsets.data() + offset,
                                                       p.layer_id,
                                                       (int)param_.cache_block_seq_len};

        // prefill only
        params.linear_iter_params = LinearIteratorParams{tmp_kv.raw_data(),  //
                                                         int(2 * stat.k_sum * size_per_head_),
                                                         int(stat.k_sum * size_per_head_)};

        params.finished = d.finished.data() + offset;
        params.cu_q_len = d.q_offsets.data() + offset;
        params.cu_k_len = d.k_offsets.data() + offset;

        params.num_heads     = local_head_num_;
        params.num_kv_heads  = local_kv_head_num_;
        params.size_per_head = size_per_head_;
        params.layer_id      = p.layer_id;

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

        params.rope_param = rope_param_;
        if (rope_param_.type == RopeType::kDynamic) {
            params.rope_param.base = d.rope_base.data() + offset;
        }
        else if (rope_param_.type == RopeType::kMrope) {
            params.rope_param.mrope.position_ids   = d.mrope_position_ids.data() + offset * rope_param_.mrope.stride;
            params.rope_param.mrope.position_delta = d.mrope_position_delta.data() + offset;
            params.rope_param.mrope.length         = d.mrope_length.data() + offset;
        }

        // logn attn
        params.use_logn_attn           = param_.use_logn_attn;
        params.max_position_embeddings = param_.max_position_embeddings;

        // Decoding use only for now
        params.split_cnt   = split_cnt_.data();
        params.partial_ML  = partial_ML_.data();
        params.partial_O   = partial_O_.data();
        params.max_split_k = std::min(std::max(1, kMaxWorkspaceTokens / params.token_num), max_kv_splits);

        // context parallel
        params.cp_rank = engine_param_.attn_cp_rank;
        params.cp_size = engine_param_.attn_cp_size;
        if (params.cp_size > 1) {
            params.cp_size = cutlass::FastDivmod(params.cp_size);

            // update ML,O offset if both prefill and decode present
            const int offset_ML_stage =
                engine_param_.attn_cp_size * (offset ? kMaxWorkspaceTokens * local_head_num_ * 2 : 0);
            const int offset_ML_rank = params.cp_rank * params.token_num * local_head_num_ * params.max_split_k * 2;
            const int offset_O       = offset ? kMaxWorkspaceTokens * local_head_num_ * size_per_head_ : 0;

            params.partial_ML = partial_ML_.data() + offset_ML_stage + offset_ML_rank;
            params.partial_O  = partial_O_.data() + offset_O;
            params.offset_q   = offset;

            // postprocess func
            params.cp_fn          = CpPost;
            params.cp_fn_ctx      = (void*)&cp_fn_ctx_;
            cp_fn_ctx_.cp_rank    = params.cp_rank;
            cp_fn_ctx_.count      = params.token_num * local_head_num_ * params.max_split_k * 2;
            cp_fn_ctx_.partial_ML = partial_ML_.data() + offset_ML_stage;
            cp_fn_ctx_.stream     = stream;
        }

        params.arch   = arch_;
        params.stream = stream;

        params.quant_policy = model_param_.quant_policy;
        return params;
    };

    const cudaStream_t stream = core::Context::stream().handle();

    cudaStream_t pf_stream = stream;
    cudaStream_t dc_stream = pf_stream;

    if (d.decode.n && d.prefill.n) {
        pf_stream = aux_stream_;
        check_cuda_error(cudaEventRecord(qkv_event_, stream));
        check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));
    }

    if (d.prefill.n && !is_warm_up_) {
        const int offset = d.decode.n;
        // We are executing prefill & decoding kernels concurrently, but only have 1 workspace
        // disable split kv for prefill for now
        auto params = CreateParams(offset, d.prefill, 1, pf_stream);
        if constexpr (sizeof(T) == 2) {
            invokeProcessKV_v2_(params);
            sync_check_cuda_error();

            /// TODO: skip flattening for `sm_80`
            invokeFlattenKV_v2_(params, d.prefill.k_sum);
            sync_check_cuda_error();

            dispatchAttention(params);
            sync_check_cuda_error();
        }
    }

    if (d.decode.n && !is_warm_up_) {
        auto params = CreateParams(0, d.decode, kMaxKVSplits, dc_stream);
        if constexpr (sizeof(T) == 2) {
            dispatchDecoding<T>(params);
            sync_check_cuda_error();
        }
    }

    if (d.decode.n && d.prefill.n) {
        check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
        check_cuda_error(cudaStreamWaitEvent(stream, aux_event_));
    }

    if (is_warm_up_) {
        rng_.set_stream(stream);
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

    const auto stream = core::Context::stream().handle();

    if (w.q_proj.weight) {
        q = linear_.Forward(hidden_state, w.q_proj);
        sync_check_cuda_error();
    }
    else {
        Tensor q_a = linear_.Forward(hidden_state, w.q_a_proj);
        sync_check_cuda_error();

        invokeRMSNorm(q_a, q_a, w.q_a_layernorm, model_param_.norm_eps, stream);
        sync_check_cuda_error();

        q = linear_.Forward(q_a, w.q_b_proj);
        sync_check_cuda_error();
    }

    Tensor kv_a_k_pe = linear_.Forward(hidden_state, w.kv_a_proj);
    sync_check_cuda_error();

    auto kv_a = kv_a_k_pe.slice({0, 0}, {-1, kv_lora_rank});
    invokeRMSNorm(kv_a, kv_a, w.kv_a_layernorm, model_param_.norm_eps, stream);
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
               stream);
    sync_check_cuda_error();

    return qkv;
}

void UnifiedAttentionLayer::qk_norm(Tensor& qkv, const WeightType& weights)
{
    const auto stream = core::Context::stream().handle();

    check_cuda_error(cudaEventRecord(qkv_event_, stream));
    check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));

    TM_CHECK(model_param_.attn_bias == false) << "not implemented";

    const auto token_num = qkv.shape(0);

    auto qkv3 = qkv.view({token_num, -1, (int)size_per_head_});

    auto q = qkv3.slice({0, 0, 0}, {-1, (int)local_head_num_, -1});
    invokeRMSNormQK(q, weights.q_a_layernorm, model_param_.norm_eps, stream);
    sync_check_cuda_error();

    auto k = qkv3.slice({0, (int)local_head_num_, 0}, {-1, (int)local_kv_head_num_, -1});
    invokeRMSNormQK(k, weights.kv_a_layernorm, model_param_.norm_eps, aux_stream_);
    sync_check_cuda_error();

    check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
    check_cuda_error(cudaStreamWaitEvent(stream, aux_event_));
}

}  // namespace turbomind
