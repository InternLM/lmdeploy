// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/kernels/activation_kernels.h"

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

MoeFfnLayer::MoeFfnLayer(const ModelParam& model, const MoeParam& param, const EngineParam& engine, const Context& ctx):
    inter_size_(param.inter_size / engine.mlp_tp_size),
    hidden_dim_(model.hidden_units),
    param_(param),
    stream_(ctx.stream),
    linear_(*ctx.linear)
{
    TM_CHECK(!param.expert_num.empty());

    const int max_expert_num = *std::max_element(param.expert_num.begin(), param.expert_num.end());

    if (param_.method == MoeParam::kFused) {
        context_ =
            std::make_unique<gemm::MoeGemmContext>(max_expert_num, param.experts_per_token, ctx.device_prop, stream_);
    }
    else {
        expert_ffn_ = std::make_unique<LlamaFfnLayer>(model, ctx);
    }

    h_offsets_ = {max_expert_num + 1, kCPUpinned};

    const int max_token_num = engine.max_forward_token_num;
    const int pad_token_num = (max_token_num + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    masks_   = {max_expert_num * pad_token_num, kDEVICE};
    f2n_     = {param_.experts_per_token * max_token_num, kDEVICE};
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
    linear_.forward(input, gate, LlamaLinear::kGemm, logits);
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

    check_cuda_error(cudaMemsetAsync(accum_.data(), 0, sizeof(int) * expert_num * kMoeGateMaxTiles, stream_));
    check_cuda_error(cudaMemsetAsync(masks_.data(), -1, sizeof(int8_t) * expert_num * padded, stream_));

    // dump_logits(tokens, layer_id);

    bool softmax = true;
    if (param_.topk_method == "group_limited_greedy") {
        invokeMoeSoftmaxMaskTopKGroups(
            logits.data(), tokens, expert_num, expert_num / param_.n_group, param_.topk_group, stream_);
        sync_check_cuda_error();
        softmax = false;
    }

    /// TODO: fix illegal memory access even if NaN are present in logits
    invokeMoeGate_V2(f2n_.data(),
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
                     stream_);
    sync_check_cuda_error();

    if (isTuning()) {
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
        check_cuda_error(cudaMemcpyAsync(
            offsets_.data(), h_offsets_.data(), sizeof(int) * (expert_num + 1), cudaMemcpyDefault, stream_));
    }

    temp_ = Tensor{{param_.experts_per_token * tokens, hidden_dim_}, p.input.dtype(), p.input.device()};

    if (param_.method == MoeParam::kNaive) {

        invokeMoeDispatch(temp_, p.input, f2n_.data(), param_.experts_per_token, stream_);
        sync_check_cuda_error();

        check_cuda_error(cudaMemcpyAsync(
            h_offsets_.data(), offsets_.data(), sizeof(int) * (expert_num + 1), cudaMemcpyDefault, stream_));

        check_cuda_error(cudaStreamSynchronize(stream_));

        TM_CHECK_EQ(h_offsets_[expert_num], tokens * param_.experts_per_token);

        for (int i = 0; i < expert_num; ++i) {
            if (int count = h_offsets_[i + 1] - h_offsets_[i]) {
                auto io = temp_.slice({h_offsets_[i], 0}, {count, -1});
                expert_ffn_->forward({io, io, moe.experts.at(i).get(), p.layer_id});
            }
        }
    }
    else {
        context_->update(expert_num, param_.experts_per_token, offsets_.data());

        auto& block = moe.block;

        const int inter_dim = block.is_fused_silu ? inter_size_ : inter_size_ * 2;
        Tensor    inter{{tokens * param_.experts_per_token, inter_dim}, p.input.dtype(), p.input.device()};

        linear_.forward_moe(inter,
                            p.input,
                            f2n_.data(),
                            offsets_.data(),
                            block.fused_gating_intermediate,
                            block.is_fused_silu ? LlamaLinear::kFusedSiluFfn : LlamaLinear::kGemm,
                            context_.get());
        sync_check_cuda_error();

        if (!block.is_fused_silu) {
            invokeGenericActivation_v3<SiluActivation>(inter.slice({0, 0}, {-1, inter_size_}),  //
                                                       inter.slice({0, inter_size_}, {-1, -1}),
                                                       stream_);
            sync_check_cuda_error();
        }

        linear_.forward_moe(temp_,
                            inter.slice({0, 0}, {-1, inter_size_}),
                            nullptr,
                            offsets_.data(),
                            block.output,
                            LlamaLinear::kGemm,
                            context_.get());
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
                     scales_.data(),
                     en2f_.data(),
                     shared_scales_.data_or((float*)nullptr),
                     param_.experts_per_token,
                     p.scale,
                     stream_);
    sync_check_cuda_error();

    temp_          = {};
    shared_scales_ = {};
}

}  // namespace turbomind
