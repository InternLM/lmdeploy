// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/dflash2/dflash2_model.h"

#include <algorithm>
#include <cmath>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/dflash2/dflash2_kernels.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/models/speculative/dflash2/dflash2_weight.h"
#include "src/turbomind/models/speculative/eagle3/target_hidden_projection.h"
#include "src/turbomind/models/speculative/registry.h"

namespace turbomind {

int Dflash2Policy::max_proposals() const
{
    return n_;
}

int Dflash2Policy::token_row_tail() const
{
    return n_ + 1;
}

RoundExtent Dflash2Policy::Extent(const RoundRequest&) const
{
    return {n_ + 1, 0};
}

std::optional<BootstrapExtent> Dflash2Policy::Bootstrap(int prompt_len) const
{
    return BootstrapExtent{prompt_len + n_, prompt_len + 2 * n_ + 1};
}

Dflash2Model::Dflash2Model(const SpeculativeModelArgs& args):
    spec_weights_{*TM_CHECK_NOTNULL(args.draft_weights.get<Dflash2Weight>("spec"))},
    policy_{args.param.spec_num_draft_tokens},
    draft_{args.registry, args.param, args.ctx, args.draft_weights, args.phases},
    projection_{std::make_unique<TargetHiddenProjection>(args.param,
                                                         args.ctx,
                                                         args.phases,
                                                         args.target.weights().num_layer,
                                                         args.target.weights().hidden_units,
                                                         args.target.weights().data_type,
                                                         args.param.spec_tap_layer_ids,
                                                         *spec_weights_.fc)},
    ctx_{args.ctx},
    param_{args.param},
    target_{args.target}
{
    const auto& dw = draft_.weights();
    hidden_      = dw.hidden_units;
    const auto* a0 = dw.layer(0)->attention.get();
    hd_          = a0->head_dim;
    local_nh_    = a0->head_num / a0->tp_size;
    local_nkv_   = a0->kv_head_num / a0->tp_size;
    total_heads_ = local_nh_ + 2 * local_nkv_;
    const auto* f0 = dw.layer(0)->feed_forward.get();
    if (f0->w1 && f0->w1->output_dim) {
        inter_local_ = f0->w1->output_dim;
    }
    else if (f0->w1w3 && f0->w1w3->output_dim) {
        inter_local_ = f0->w1w3->output_dim / 2;
    }
    else {
        inter_local_ = f0->inter_size / (f0->tp_size ? f0->tp_size : 1);
    }
    top_k_       = param_.spec_selector_top_k ? param_.spec_selector_top_k : 16;
    sel_rank_    = param_.spec_selector_rank ? param_.spec_selector_rank : 256;
    block_       = param_.spec_block_size ? param_.spec_block_size : 8;
    mask_token_id_ = param_.spec_mask_token_id;
    window_      = param_.spec_sliding_window ? param_.spec_sliding_window : 2048;
    rope_base_   = 1e7f;
    max_ctx_     = param_.session_len;

    const DataType dtype = dw.data_type;
    ctx_kv_int4_ = param_.spec_ctx_kv_int4 != 0;
    if (ctx_kv_int4_ && hd_ != 128) {
        TM_LOG_WARN("DFlash2 int4 draft context KV requires head_dim 128; falling back to BF16");
        ctx_kv_int4_ = false;
    }
    if (ctx_kv_int4_) {
        ctx_k_data_.resize(dw.num_layer);
        ctx_k_param_.resize(dw.num_layer);
        ctx_v_data_.resize(dw.num_layer);
        ctx_v_param_.resize(dw.num_layer);
        const size_t data_bytes  = (size_t)max_ctx_ * local_nkv_ * (hd_ / 2);
        const size_t param_bytes = (size_t)max_ctx_ * local_nkv_ * 2 * byte_size(dtype);
        for (int l = 0; l < dw.num_layer; ++l) {
            ctx_k_data_[l]  = Tensor{{max_ctx_, local_nkv_, hd_ / 2}, kUint8, kDEVICE};
            ctx_k_param_[l] = Tensor{{max_ctx_, local_nkv_, 2}, dtype, kDEVICE};
            ctx_v_data_[l]  = Tensor{{max_ctx_, local_nkv_, hd_ / 2}, kUint8, kDEVICE};
            ctx_v_param_[l] = Tensor{{max_ctx_, local_nkv_, 2}, dtype, kDEVICE};
            cudaMemsetAsync(ctx_k_data_[l].raw_data(), 0, data_bytes, 0);
            cudaMemsetAsync(ctx_k_param_[l].raw_data(), 0, param_bytes, 0);
            cudaMemsetAsync(ctx_v_data_[l].raw_data(), 0, data_bytes, 0);
            cudaMemsetAsync(ctx_v_param_[l].raw_data(), 0, param_bytes, 0);
        }
    }
    else {
        ctx_k_.resize(dw.num_layer);
        ctx_v_.resize(dw.num_layer);
        const size_t bytes = (size_t)max_ctx_ * local_nkv_ * hd_ * byte_size(dtype);
        for (int l = 0; l < dw.num_layer; ++l) {
            ctx_k_[l] = Tensor{{max_ctx_, local_nkv_, hd_}, dtype, kDEVICE};
            ctx_v_[l] = Tensor{{max_ctx_, local_nkv_, hd_}, dtype, kDEVICE};
            cudaMemsetAsync(ctx_k_[l].raw_data(), 0, bytes, 0);
            cudaMemsetAsync(ctx_v_[l].raw_data(), 0, bytes, 0);
        }
    }
    cudaDeviceSynchronize();
    ctx_len_ = {param_.max_batch_size, kDEVICE};
    cudaMemsetAsync(ctx_len_.data(), 0, param_.max_batch_size * sizeof(int));
}

Dflash2Model::~Dflash2Model() = default;

const SpeculativePolicy& Dflash2Model::policy() const
{
    return policy_;
}

void Dflash2Model::Run(BatchOp, int, TensorMap&) {}

HiddenStateTap* Dflash2Model::Tap(int)
{
    return projection_.get();
}

void Dflash2Model::MaterializeContext(const Tensor& context_hidden, const Buffer_<int>& base_dev,
                                      const Buffer_<int>& committed_dev, int rows, cudaStream_t stream)
{
    const auto& dw  = draft_.weights();
    auto&       lin = *ctx_.linear;
    const float eps = spec_weights_.hidden_norm->norm_eps_;
    const bool  zc  = spec_weights_.hidden_norm->zero_centered_;

    // Positions (base + i) are generated on device; rows past the newly
    // committed delta are skipped inside the scatter kernel.
    Buffer_<int> pos_dev{rows, kDEVICE};
    invokeDflash2CountUp(pos_dev.data(), base_dev.data(), rows, stream);

    Tensor x{{rows, hidden_}, dw.data_type, kDEVICE};
    invokeRMSNorm(x, context_hidden, spec_weights_.hidden_norm->weight, eps, zc, stream);

    const int per_row = local_nkv_ * hd_;
    Tensor qkv2{{rows, total_heads_ * hd_}, dw.data_type, kDEVICE};
    for (int l = 0; l < dw.num_layer; ++l) {
        const auto* a = dw.layer(l)->attention.get();
        lin.Forward(x, *a->w_qkv, qkv2);
        Tensor qkv = qkv2.view({rows, total_heads_, hd_});
        invokeQkRMSNorm(qkv, a->q_norm->weight, a->k_norm->weight, local_nh_, local_nkv_,
                        a->q_norm->norm_eps_, a->q_norm->zero_centered_, stream);
        Tensor krow = qkv.slice({0, local_nh_, 0}, {rows, local_nkv_, hd_});
        Tensor vrow = qkv.slice({0, local_nh_ + local_nkv_, 0}, {rows, local_nkv_, hd_});
        Tensor kc{{rows, local_nkv_, hd_}, dw.data_type, kDEVICE};
        Tensor vc{{rows, local_nkv_, hd_}, dw.data_type, kDEVICE};
        GenericCopy(krow, kc, stream);
        GenericCopy(vrow, vc, stream);
        invokeDflash2Rope(kc.raw_data(), pos_dev.data(), rows, local_nkv_, hd_, rope_base_, 16, stream);
        if (ctx_kv_int4_) {
            invokeDflash2KvScatterInt4(ctx_k_data_[l].raw_data(), ctx_k_param_[l].raw_data(), kc.raw_data(),
                                       base_dev.data(), committed_dev.data(), rows, local_nkv_, hd_, 16, stream);
            invokeDflash2KvScatterInt4(ctx_v_data_[l].raw_data(), ctx_v_param_[l].raw_data(), vc.raw_data(),
                                       base_dev.data(), committed_dev.data(), rows, local_nkv_, hd_, 16, stream);
        }
        else {
            invokeDflash2KvScatter(ctx_k_[l].raw_data(), kc.raw_data(), base_dev.data(), committed_dev.data(), rows,
                                   per_row, 16, stream);
            invokeDflash2KvScatter(ctx_v_[l].raw_data(), vc.raw_data(), base_dev.data(), committed_dev.data(), rows,
                                   per_row, 16, stream);
        }
    }
}

Tensor Dflash2Model::RunBlock(const Tensor& block_embeddings, const Tensor& positions, int rows, int ctx_rows,
                              cudaStream_t stream, TensorMap& env)
{
    const auto&    dw     = draft_.weights();
    auto&          lin    = *ctx_.linear;
    const DataType dtype  = dw.data_type;
    const float    qscale = 1.0f / std::sqrt((float)hd_);
    const int      G      = hidden_ / (param_.spec_conv_group_size ? param_.spec_conv_group_size : 16);
    const int      gs     = param_.spec_conv_group_size ? param_.spec_conv_group_size : 16;
    const int      taps   = param_.spec_conv_kernel_size ? param_.spec_conv_kernel_size : 2;

    // Range-preserving: the residual stream is FP32 (fp16 overflows otherwise).
    Tensor residual{{rows, hidden_}, DataType::kFloat32, kDEVICE};
    cudaMemsetAsync(residual.raw_data(), 0, (size_t)rows * hidden_ * sizeof(float), stream);
    Tensor h{{rows, hidden_}, dtype, kDEVICE};
    Tensor qkv2{{rows, total_heads_ * hd_}, dtype, kDEVICE};
    Tensor attn{{rows, local_nh_, hd_}, dtype, kDEVICE};
    Tensor o{{rows, hidden_}, dtype, kDEVICE};
    Tensor red{{rows, hidden_}, dtype, kDEVICE};
    Tensor convd{{rows, 2 * taps * G}, dtype, kDEVICE};
    Tensor convout{{rows, hidden_}, dtype, kDEVICE};
    Tensor gating{{rows, inter_local_}, dtype, kDEVICE};
    Tensor inter{{rows, inter_local_}, dtype, kDEVICE};
    Tensor down{{rows, hidden_}, dtype, kDEVICE};

    auto allreduce = [&](Tensor& t) {
        if (ctx_.comm.d_comm) {
            ctx_.comm.d_comm->AllReduceSum(
                t.raw_data(), red.raw_data(), (size_t)rows * hidden_, dtype, ctx_.comm.d_tp_group, stream);
            Copy(red, t);
        }
    };
    // side-1 coefficients come from the layer INPUT at prepare, consumed at finish.
    Tensor conv_side1{{rows, taps * G}, dtype, kDEVICE};
    Tensor conv_side0{{rows, taps * G}, dtype, kDEVICE};
    auto   conv = [&](GroupedConvWeight* c, Tensor& y, bool prepare) {
        if (!c) {
            return;
        }
        const void* base;
        if (prepare) {
            lin.Forward(y, *c->kernel_projection, convd);
            // The kernel assumes contiguous [rows, taps*G] rows; the raw slice
            // of the [rows, 2*taps*G] projection is strided, so repack it.
            GenericCopy(convd.slice({0, 0}, {rows, taps * G}), conv_side0, stream);
            GenericCopy(convd.slice({0, taps * G}, {rows, taps * G}), conv_side1, stream);
            base = c->base_kernel.slice({0, 0, 0}, {1, taps, hidden_}).raw_data();
            invokeGroupedConv(convout.raw_data(), y.raw_data(), conv_side0.raw_data(), base, rows, hidden_, G, gs, taps,
                              block_, 16, stream);
        }
        else {
            base = c->base_kernel.slice({1, 0, 0}, {1, taps, hidden_}).raw_data();
            invokeGroupedConv(convout.raw_data(), y.raw_data(), conv_side1.raw_data(), base, rows, hidden_, G, gs,
                              taps, block_, 16, stream);
        }
        Copy(convout, y);
    };
    auto resnorm = [&](const Tensor& x, const NormWeight* norm, float x_scale) {
        invokeDflash2ResidualRMSNorm(h.raw_data(),
                                     residual.data<float>(),
                                     x.raw_data(),
                                     norm->weight.raw_data(),
                                     rows,
                                     hidden_,
                                     norm->norm_eps_,
                                     1,
                                     x_scale,
                                     16,
                                     stream);
    };
    // Range-preserving: sub-layer outputs are stored scaled-down (fp16) and the
    // FP32 residual is restored by the reciprocal scale (as the SM70 path does).
    constexpr float kResidualScale = 256.0f;

    resnorm(block_embeddings, dw.layer(0)->attention_norm.get(), 1.0f);

    for (int l = 0; l < dw.num_layer; ++l) {
        auto*       layer = dw.layer(l);
        const auto* a     = layer->attention.get();
        conv(layer->attention_conv.get(), h, true);
        lin.Forward(h, *a->w_qkv, qkv2);
        Tensor qkv = qkv2.view({rows, total_heads_, hd_});
        invokeQkRMSNorm(qkv, a->q_norm->weight, a->k_norm->weight, local_nh_, local_nkv_, a->q_norm->norm_eps_,
                        a->q_norm->zero_centered_, stream);
        Tensor qv = qkv.slice({0, 0, 0}, {rows, local_nh_, hd_});
        Tensor kv = qkv.slice({0, local_nh_, 0}, {rows, local_nkv_, hd_});
        Tensor vv = qkv.slice({0, local_nh_ + local_nkv_, 0}, {rows, local_nkv_, hd_});
        Tensor q2{{rows, local_nh_, hd_}, dtype, kDEVICE};
        Tensor k2{{rows, local_nkv_, hd_}, dtype, kDEVICE};
        Tensor v2{{rows, local_nkv_, hd_}, dtype, kDEVICE};
        GenericCopy(qv, q2, stream);
        GenericCopy(kv, k2, stream);
        GenericCopy(vv, v2, stream);
        invokeDflash2Rope(q2.raw_data(), positions.data<int>(), rows, local_nh_, hd_, rope_base_, 16, stream);
        invokeDflash2Rope(k2.raw_data(), positions.data<int>(), rows, local_nkv_, hd_, rope_base_, 16, stream);
        if (ctx_kv_int4_) {
            Tensor kctx_data  = ctx_k_data_[l].slice({0, 0, 0}, {ctx_rows, local_nkv_, hd_ / 2});
            Tensor kctx_param = ctx_k_param_[l].slice({0, 0, 0}, {ctx_rows, local_nkv_, 2});
            Tensor vctx_data  = ctx_v_data_[l].slice({0, 0, 0}, {ctx_rows, local_nkv_, hd_ / 2});
            Tensor vctx_param = ctx_v_param_[l].slice({0, 0, 0}, {ctx_rows, local_nkv_, 2});
            invokeDflash2AttentionInt4(attn.raw_data(), q2.raw_data(), kctx_data.raw_data(), kctx_param.raw_data(),
                                       vctx_data.raw_data(), vctx_param.raw_data(), k2.raw_data(), v2.raw_data(),
                                       rows, ctx_rows, local_nh_, local_nkv_, hd_, qscale, window_, 16, stream);
        }
        else {
            Tensor kctx = ctx_k_[l].slice({0, 0, 0}, {ctx_rows, local_nkv_, hd_});
            Tensor vctx = ctx_v_[l].slice({0, 0, 0}, {ctx_rows, local_nkv_, hd_});
            invokeDflash2Attention(attn.raw_data(), q2.raw_data(), kctx.raw_data(), vctx.raw_data(), k2.raw_data(),
                                   v2.raw_data(), rows, ctx_rows, local_nh_, local_nkv_, hd_, qscale, window_, 16,
                                   stream);
        }
        invokeDflash2Scale(attn.raw_data(), (long long)rows * local_nh_ * hd_, 1.0f / kResidualScale, 16, stream);
        lin.Forward(attn.view({rows, local_nh_ * hd_}), *a->wo, o);
        allreduce(o);
        conv(layer->attention_conv.get(), o, false);
        resnorm(o, layer->ffn_norm.get(), kResidualScale);
        // mlp
        conv(layer->mlp_conv.get(), h, true);
        auto* ff = layer->feed_forward.get();
        void* gate_scale_ptr = nullptr;
        long long gate_scale_n = 0;
        if (ff->w1 && ff->w3) {
            lin.Forward(h, *ff->w1, gating);
            lin.Forward(h, *ff->w3, inter);
            Activation(gating, inter, ff->act_type, stream);
            gate_scale_ptr = gating.raw_data();
            gate_scale_n   = (long long)rows * inter_local_;
        }
        else if (ff->w1w3) {
            Tensor gate_up{{rows, 2 * inter_local_}, dtype, kDEVICE};
            lin.Forward(h, *ff->w1w3, gate_up);
            gating = gate_up.slice({0, 0}, {rows, inter_local_});
            inter  = gate_up.slice({0, inter_local_}, {rows, inter_local_});
            Activation(gating, inter, ff->act_type, stream);
            gate_scale_ptr = gate_up.raw_data();
            gate_scale_n   = (long long)rows * 2 * inter_local_;
        }
        // Range-preserving: scale the MLP output (w2 input) down; fp32 residual
        // restores the reciprocal scale (silu(gate)*up overflows fp16 otherwise).
        invokeDflash2Scale(gate_scale_ptr, gate_scale_n, 1.0f / kResidualScale, 16, stream);
        lin.Forward(gating, *ff->w2, down);
        allreduce(down);
        conv(layer->mlp_conv.get(), down, false);
        const NormWeight* next_norm = (l + 1 < dw.num_layer) ? dw.layer(l + 1)->attention_norm.get() : dw.norm.get();
        resnorm(down, next_norm, kResidualScale);
    }
    return h;
}

void Dflash2Model::RunDraft(int phase, const DraftContext& ctx, TensorMap& env)
{
    if (*ctx_.is_warm_up) {
        return;  // no tap data during warm-up
    }
    if (ctx.batch_size != 1) {
        return;  // single-request path only for now
    }
    const auto&    dw    = draft_.weights();
    const auto     stream = core::Context::stream().handle();
    const DataType dtype = dw.data_type;

    // 1. context hidden (fc applied by the tap) for this round's rows
    Tensor context_hidden = projection_->ProjectAndGather(phase, stream);  // [N, H]
    const int rows = context_hidden.shape(0);
    if (rows == 0) {
        return;
    }
    // committed length after CommitAcceptedState; only the newly committed rows
    // (seqlen - already materialized) extend the draft context.
    // NOTE: ctx.sequence_length is the ENTRY length (pre-commit this round);
    // the committed length is entry + accept_len. Both live on the device, so
    // `committed` is computed device-side (one D2H read is still needed below
    // to size the block attention over the context).
    Buffer_<int> base_dev{1, kDEVICE};
    cudaMemcpyAsync(base_dev.data(), ctx_len_.data(), sizeof(int), cudaMemcpyDeviceToDevice, stream);  // base
    Buffer_<int> committed_dev{1, kDEVICE};
    invokeDflash2CommitLen(committed_dev.data(),
                           ctx.sequence_length.data(),
                           ctx.accept_len.data(),
                           ctx.accept_len.data() != nullptr,
                           stream);

    MaterializeContext(context_hidden, base_dev, committed_dev, rows, stream);

    cudaMemcpyAsync(ctx_len_.data(), committed_dev.data(), sizeof(int), cudaMemcpyDeviceToDevice, stream);  // next base

    // One D2H: the block attention loops over `committed` context keys, so it
    // needs this host-side. Everything else stays on the device. The stream is
    // synchronized first so the device-side commit_len / ctx_len_ update (on
    // `stream`) are visible to the host read.
    cudaStreamSynchronize(stream);
    int committed = 0;
    cudaMemcpy(&committed, committed_dev.data(), sizeof(int), cudaMemcpyDeviceToHost);

    // 2. block ids [next, mask x (block-1)] (device-side)
    Buffer_<int> ids{block_, kDEVICE};
    invokeDflash2BuildBlockIds(ids.data(),
                               ctx.request_token_ids_ptrs,
                               committed_dev.data(),
                               block_,
                               mask_token_id_,
                               stream);

    // 3. block embeddings (shared target embeddings)
    Tensor embed_buf{{block_, hidden_}, dtype, kDEVICE};
    Tensor embeds = target_.Embed(ids, embed_buf, env);

    // 4. block positions = [committed .. committed+block) (device-side)
    Buffer_<int> bpos_dev{block_, kDEVICE};
    invokeDflash2CountUp(bpos_dev.data(), committed_dev.data(), block_, stream);

    Tensor block_hidden = RunBlock(embeds, bpos_dev, block_, committed, stream, env);  // [block, H]

    // 5. selector over the mask positions (rows 1..block-1)
    const int L = block_ - 1;
    Tensor    mask_h = block_hidden.slice({1, 0}, {L, hidden_});
    Tensor    sel_h{{L, sel_rank_}, dtype, kDEVICE};
    ctx_.linear->Forward(mask_h, *spec_weights_.selector->hidden_projection, sel_h);

    Tensor        logits = target_.Logits(mask_h, {}, env);  // [L, vocab]
    const int     vocab  = logits.shape(1);
    Buffer_<int>  cand{L * top_k_, kDEVICE};
    Tensor        unary{{L, top_k_}, DataType::kFloat32, kDEVICE};
    invokeDflash2TopK(cand.data(), unary.data<float>(), logits.raw_data(), L, vocab, top_k_, 16, stream);

    Tensor edge{{L, top_k_, top_k_}, DataType::kFloat32, kDEVICE};
    invokeCandidateSelector(edge.data<float>(),
                            sel_h.raw_data(),
                            unary.raw_data(),
                            spec_weights_.selector->predecessor_codebook.raw_data(),
                            spec_weights_.selector->successor_codebook.raw_data(),
                            cand.data(),
                            ids.data(),
                            1,
                            L,
                            top_k_,
                            vocab,
                            sel_rank_,
                            16,
                            stream);

    // 6. greedy chain (device) + store into the request's token ids
    Buffer_<int> props{L, kDEVICE};
    invokeDflash2GreedyChain(props.data(), edge.data<float>(), cand.data(), L, top_k_, stream);
    Buffer_<int> sb{1, kDEVICE};
    Copy(committed_dev, sb);
    invokeDflash2StoreTokens(ctx.request_token_ids_ptrs, props.data(), sb.data(), L, stream);
}

TM_REGISTER_SPECULATIVE_MODEL("dflash2", Dflash2Model);

}  // namespace turbomind
