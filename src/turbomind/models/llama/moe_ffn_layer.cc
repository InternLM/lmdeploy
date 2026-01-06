// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/core/context.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

// #include "dbg.h"

namespace turbomind {

MoeFfnLayer::MoeFfnLayer(const ModelParam& model, const MoeParam& param, const EngineParam& engine, const Context& ctx):
    inter_size_(param.inter_size / engine.mlp_tp_size),
    hidden_dim_(model.hidden_units),
    tp_size_(engine.mlp_tp_size),
    param_(param),
    is_warm_up_{*ctx.is_warm_up},
    linear_(*ctx.linear)
{
    TM_CHECK(!param.expert_num.empty());

    const int max_expert_num = *std::max_element(param.expert_num.begin(), param.expert_num.end());

    if (param_.method == MoeParam::kFused) {
        // pass
    }
    else {
        expert_ffn_ = std::make_unique<LlamaFfnLayer>(model, ctx);
    }

    h_offsets_ = {max_expert_num + 1, kCPUpinned};

    const int max_token_num = engine.max_forward_token_num * engine.attn_dp_size;
    const int pad_token_num = (max_token_num + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    // dbg(inter_size_,
    //     hidden_dim_,
    //     tp_size_,
    //     param_.method,
    //     param.expert_num,
    //     max_expert_num,
    //     max_token_num,
    //     pad_token_num,
    //     param_.experts_per_token);

    masks_   = {max_expert_num * pad_token_num, kDEVICE};
    f2n_     = {param_.experts_per_token * max_token_num, kDEVICE};
    f2E_     = {param_.experts_per_token * max_token_num, kDEVICE};
    en2f_    = {param_.experts_per_token * max_token_num, kDEVICE};
    scales_  = {param_.experts_per_token * max_token_num, kDEVICE};
    offsets_ = {max_expert_num + 1, kDEVICE};
    accum_   = {max_expert_num * kMoeGateMaxTiles, kDEVICE};
}

Tensor_<float> MoeFfnLayer::Gate(const Tensor& input, const LlamaDenseWeight& gate)
{
    auto& weight = gate.weight;
    TM_CHECK_EQ(input.shape(1), weight.shape(0));
    Tensor_<float> logits{{input.shape(0), weight.shape(1)}, kDEVICE};
    linear_.Forward(input, gate, logits);
    sync_check_cuda_error();
    ApplyBias(logits, gate.bias, core::Context::stream().handle());
    sync_check_cuda_error();
    return logits;
}

void MoeFfnLayer::Forward(ForwardParam& p)
{
    const int   tokens = p.input.shape(0);
    const auto& moe    = *p.weights;

    const size_t padded     = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;
    const int    expert_num = moe.experts.size();

    FT_CHECK(expert_num);

    auto logits = Gate(p.input, moe.gate);

    TM_DEBUG_TENSOR(logits, "logits", 2);

    const auto st = core::Context::stream().handle();

    check_cuda_error(cudaMemsetAsync(accum_.data(), 0, sizeof(int) * expert_num * kMoeGateMaxTiles, st));
    check_cuda_error(cudaMemsetAsync(masks_.data(), -1, sizeof(int8_t) * expert_num * padded, st));

    // dump_logits(tokens, layer_id);

    bool softmax = true;
    if (param_.topk_method == "group_limited_greedy") {
        invokeMoeSoftmaxMaskTopKGroups(
            logits.data(), tokens, expert_num, expert_num / param_.n_group, param_.topk_group, st);
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
                     param_.experts_per_token,
                     softmax,
                     param_.norm_topk_prob,
                     param_.routed_scale,
                     st);
    sync_check_cuda_error();

    if (is_warm_up_) {
        std::mt19937     g;
        const auto       expert_ids = SampleUniform(tokens, expert_num, param_.experts_per_token, g);
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

    temp_ = Tensor{{param_.experts_per_token * tokens, hidden_dim_}, p.input.dtype(), p.input.device()};

    if (param_.method == MoeParam::kNaive) {

        invokeMoeDispatch(temp_, p.input, f2n_.data(), param_.experts_per_token, st);
        sync_check_cuda_error();

        check_cuda_error(
            cudaMemcpyAsync(h_offsets_.data(), offsets_.data(), sizeof(int) * (expert_num + 1), cudaMemcpyDefault, st));

        check_cuda_error(cudaStreamSynchronize(st));

        TM_CHECK_EQ(h_offsets_[expert_num], tokens * param_.experts_per_token);

        for (int i = 0; i < expert_num; ++i) {
            if (int count = h_offsets_[i + 1] - h_offsets_[i]) {
                auto io = temp_.slice({h_offsets_[i], 0}, {count, -1});
                expert_ffn_->forward({io, io, moe.experts.at(i).get(), p.layer_id});
            }
        }
    }
    else {

        auto& block = moe.block;

        auto indices = f2n_.slice(0, tokens * param_.experts_per_token);
        auto offsets = offsets_.slice(0, expert_num + 1);

        Tensor inter = linear_.Forward(p.input, block.fused_gating_intermediate, indices, offsets_);
        sync_check_cuda_error();

        if (!block.is_fused_silu) {
            Activation(inter, block.fused_gating_intermediate.bias, f2E_, moe.block.act_type, st);
            sync_check_cuda_error();
        }

        linear_.Forward(inter.slice({0, 0}, {-1, inter_size_}), block.output, {}, offsets, temp_);
        sync_check_cuda_error();
    }

    if (moe.shared_gate.weight) {
        shared_scales_ = Gate(p.input, moe.shared_gate);
    }
}

void MoeFfnLayer::Combine(ForwardParam& p)
{
    auto& moe = *p.weights;

    invokeMoeCombine(p.output,
                     temp_,
                     p.weights->block.output.bias,
                     scales_.data(),
                     en2f_.data(),
                     f2E_.data(),
                     shared_scales_.data_or((float*)nullptr),
                     param_.experts_per_token,
                     1.f / tp_size_,
                     p.scale,
                     core::Context::stream().handle());
    sync_check_cuda_error();

    temp_          = {};
    shared_scales_ = {};
}

}  // namespace turbomind
