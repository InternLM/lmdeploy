// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/core/context.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/moe_weight.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

// #include "dbg.h"

namespace turbomind {

MoeFfnLayer::MoeFfnLayer(const EngineParam& engine, const Context& ctx):
    tp_size_(engine.mlp_tp_size),
    max_token_num_(engine.max_forward_token_num * engine.attn_dp_size),
    is_warm_up_(*ctx.is_warm_up),
    linear_(*ctx.linear),
    expert_ffn_(std::make_unique<LlamaFfnLayer>(ctx))
{
}

void MoeFfnLayer::Init(ForwardParam& p)
{
    const int expert_num        = p.weights->num_experts();
    const int experts_per_token = p.weights->experts_per_token;

    h_offsets_ = {expert_num + 1, kCPU};

    const int pad_token_num = (max_token_num_ + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    masks_   = {expert_num * pad_token_num, kDEVICE};
    f2n_     = {experts_per_token * max_token_num_, kDEVICE};
    f2E_     = {experts_per_token * max_token_num_, kDEVICE};
    en2f_    = {experts_per_token * max_token_num_, kDEVICE};
    scales_  = {experts_per_token * max_token_num_, kDEVICE};
    offsets_ = {expert_num + 1, kDEVICE};
    accum_   = {expert_num * kMoeGateMaxTiles, kDEVICE};

    initialized_ = true;
}

Tensor_<float> MoeFfnLayer::Gate(const Tensor& input, const LinearWeight& gate)
{
    auto& w = gate.weight;
    TM_CHECK_EQ(input.shape(1), w.shape(0));
    Tensor_<float> logits{{input.shape(0), w.shape(1)}, kDEVICE};
    linear_.Forward(input, gate, logits);
    sync_check_cuda_error();
    ApplyBias(logits, gate.bias, core::Context::stream().handle());
    sync_check_cuda_error();
    return logits;
}

void MoeFfnLayer::Forward(ForwardParam& p)
{
    if (!initialized_) {
        Init(p);
    }

    const int hidden_dim = p.weights->hidden_dim;
    const int inter_size = p.weights->inter_size;

    const int   tokens = p.input.shape(0);
    const auto& moe    = *p.weights;

    const size_t padded     = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;
    const int    expert_num = moe.num_experts();

    FT_CHECK(expert_num);

    auto logits = Gate(p.input, *moe.gate.get());

    TM_DEBUG_TENSOR(logits, "logits", 2);

    const auto st = core::Context::stream().handle();

    if (p.weights->topk_method == "noaux_tc") {
        // invokeMoeGate_NoAuxTC clears accum and masks internally
        TM_CHECK_EQ(p.weights->n_group, 1);
        TM_CHECK_EQ(p.weights->topk_group, 1);
        const float* correction_bias = nullptr;
        if (moe.score_correction_bias) {
            correction_bias = moe.score_correction_bias.size() > 0 ? moe.score_correction_bias.data<float>() : nullptr;
        }
        invokeMoeGate_NoAuxTC(f2n_.data(),
                              f2E_.data(),
                              en2f_.data(),
                              offsets_.data(),
                              scales_.data(),
                              masks_.data(),
                              accum_.data(),
                              logits.data(),
                              correction_bias,
                              tokens,
                              padded,
                              expert_num,
                              p.weights->experts_per_token,
                              p.weights->norm_topk_prob,
                              p.weights->routed_scale,
                              p.weights->scoring_func == "sigmoid",
                              st);
    }
    else {
        // V2: accum must be cleared by caller; masks cleared internally
        check_cuda_error(cudaMemsetAsync(accum_.data(), 0, sizeof(int) * expert_num * kMoeGateMaxTiles, st));

        bool softmax = true;
        if (p.weights->topk_method == "group_limited_greedy") {
            invokeMoeSoftmaxMaskTopKGroups(
                logits.data(), tokens, expert_num, expert_num / p.weights->n_group, p.weights->topk_group, st);
            sync_check_cuda_error();
            softmax = false;
        }

        /// TODO: fix illegal memory access even if NaN are present in logits
        invokeMoeGate_V2(f2n_.data(),
                         f2E_.data(),
                         en2f_.data(),
                         offsets_.data(),
                         scales_.data(),
                         masks_.data(),
                         accum_.data(),
                         logits.data(),
                         tokens,
                         padded,
                         expert_num,
                         p.weights->experts_per_token,
                         softmax,
                         p.weights->norm_topk_prob,
                         p.weights->routed_scale,
                         st);
    }
    sync_check_cuda_error();

    if (is_warm_up_) {
        std::mt19937     g;
        const auto       expert_ids = SampleUniform(tokens, expert_num, p.weights->experts_per_token, g);
        std::vector<int> cnt(expert_num);
        for (const auto& x : expert_ids) {
            ++cnt[x];
        }
        h_offsets_[0] = 0;
        for (int i = 0; i < expert_num; ++i) {
            h_offsets_[i + 1] = h_offsets_[i] + cnt[i];
        }
        check_cuda_error(
            cudaMemcpyAsync(offsets_.data(), h_offsets_.data(), sizeof(int) * (expert_num + 1), cudaMemcpyDefault, st));
    }

    temp_ = Tensor{{p.weights->experts_per_token * tokens, hidden_dim}, p.input.dtype(), p.input.device()};

    if (p.weights->method() == MoeMethod::kNaive) {

        invokeMoeDispatch(temp_, p.input, f2n_.data(), p.weights->experts_per_token, st);
        sync_check_cuda_error();

        check_cuda_error(
            cudaMemcpyAsync(h_offsets_.data(), offsets_.data(), sizeof(int) * (expert_num + 1), cudaMemcpyDefault, st));

        check_cuda_error(cudaStreamSynchronize(st));

        TM_CHECK_EQ(h_offsets_[expert_num], tokens * p.weights->experts_per_token);

        for (int i = 0; i < expert_num; ++i) {
            if (int count = h_offsets_[i + 1] - h_offsets_[i]) {
                auto io = temp_.slice({h_offsets_[i], 0}, {count, -1});
                expert_ffn_->forward({io, io, moe.expert(i), p.layer_id});
            }
        }
    }
    else {

        auto* block = moe.block();

        auto indices = f2n_.slice(0, tokens * p.weights->experts_per_token);
        auto offsets = offsets_.slice(0, expert_num + 1);

        if (block->w1w3 && block->w1w3->weight) {
            // Fused w1w3 path
            Tensor inter = linear_.Forward(p.input, *block->w1w3, indices, offsets_);
            sync_check_cuda_error();

            if (!block->is_fused_silu()) {
                Activation(inter, block->w1w3->bias, f2E_, block->act_type(), st);
                sync_check_cuda_error();
            }

            linear_.Forward(inter.slice({0, 0}, {-1, inter_size}), *block->w2, {}, offsets, temp_);
            sync_check_cuda_error();
        }
        else {
            // Separate w1/w3 path
            Tensor gating = linear_.Forward(p.input, *block->w1, indices, offsets_);
            sync_check_cuda_error();

            Tensor up = linear_.Forward(p.input, *block->w3, indices, offsets_);
            sync_check_cuda_error();

            Activation(gating, up, block->act_type(), st);
            sync_check_cuda_error();

            linear_.Forward(gating, *block->w2, {}, offsets, temp_);
            sync_check_cuda_error();
        }
    }

    if (moe.shared_gate && moe.shared_gate->weight) {
        shared_scales_ = Gate(p.input, *moe.shared_gate.get());
    }
}

void MoeFfnLayer::Combine(ForwardParam& p)
{
    auto& moe = *p.weights;

    const Tensor& block_bias = moe.block() && moe.block()->w2 ? moe.block()->w2->bias : Tensor{};

    invokeMoeCombine(p.output,
                     temp_,
                     block_bias,
                     scales_.data(),
                     en2f_.data(),
                     f2E_.data(),
                     shared_scales_.data_or((float*)nullptr),
                     p.weights->experts_per_token,
                     1.f / tp_size_,
                     p.scale,
                     core::Context::stream().handle());
    sync_check_cuda_error();

    temp_          = {};
    shared_scales_ = {};
}

}  // namespace turbomind
