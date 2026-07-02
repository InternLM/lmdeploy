// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>
#include <string>

#include <cuda_runtime.h>

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/moe_weight.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

// #include "dbg.h"

namespace turbomind {

class MoeFfnLayerImpl {
public:
    explicit MoeFfnLayerImpl(const Context& ctx): linear_(*ctx.linear) {}

    virtual ~MoeFfnLayerImpl() = default;

    virtual void Forward(MoeFfnLayer::ForwardParam& p) = 0;

    virtual void Combine(MoeFfnLayer::ForwardParam& p) = 0;

protected:
    Tensor_<float> Gate(const Tensor& input, const LinearWeight& gate);

    LlamaLinear& linear_;
};

class MoeFfnTpImpl final: public MoeFfnLayerImpl {
public:
    MoeFfnTpImpl(const EngineParam& engine, const Context& ctx);

    void Forward(MoeFfnLayer::ForwardParam& p) override;

    void Combine(MoeFfnLayer::ForwardParam& p) override;

private:
    void Init(MoeFfnLayer::ForwardParam& p);

    const int tp_size_;
    const int max_token_num_;
    int&      is_warm_up_;

    std::unique_ptr<LlamaFfnLayer> expert_ffn_;

    bool initialized_ = false;

    Buffer_<int> h_offsets_;

    Buffer_<int>   masks_;
    Buffer_<int>   f2n_;
    Buffer_<int>   f2E_;
    Buffer_<int>   en2f_;
    Buffer_<float> scales_;
    Buffer_<int>   accum_;
    Buffer_<int>   offsets_;

    Tensor         temp_;
    Tensor_<float> shared_scales_;
};

MoeFfnTpImpl::MoeFfnTpImpl(const EngineParam& engine, const Context& ctx):
    MoeFfnLayerImpl(ctx),
    tp_size_(engine.mlp_tp_size),
    max_token_num_(engine.max_forward_token_num * engine.attn_dp_size),
    is_warm_up_(*ctx.is_warm_up),
    expert_ffn_(std::make_unique<LlamaFfnLayer>(ctx))
{
}

void MoeFfnTpImpl::Init(MoeFfnLayer::ForwardParam& p)
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

Tensor_<float> MoeFfnLayerImpl::Gate(const Tensor& input, const LinearWeight& gate)
{
    TM_FUNCTION_SCOPE();

    auto& w = gate.weight;
    TM_CHECK_EQ(input.shape(1), w.shape(0));
    Tensor_<float> logits{{input.shape(0), w.shape(1)}, kDEVICE};
    TM_SCOPE_CALL(linear_.Forward(input, gate, logits));
    ApplyBias(logits, gate.bias, core::Context::stream().handle());
    TM_CUDA_CHECK(cudaGetLastError());
    return logits;
}

void MoeFfnTpImpl::Forward(MoeFfnLayer::ForwardParam& p)
{
    TM_FUNCTION_SCOPE();
    if (!initialized_) {
        Init(p);
    }

    const int   tokens = p.input.shape(0);
    const auto& moe    = *p.weights;

    const auto& block = *TM_CHECK_NOTNULL(moe.block());

    const int hidden_dim = block.hidden_dim;
    const int inter_size = block.inter_size;

    const size_t padded     = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;
    const int    expert_num = moe.num_experts();

    TM_CHECK(expert_num);

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
        TM_CUDA_CHECK(cudaMemsetAsync(accum_.data(), 0, sizeof(int) * expert_num * kMoeGateMaxTiles, st));

        bool softmax = true;
        if (p.weights->topk_method == "group_limited_greedy") {
            invokeMoeSoftmaxMaskTopKGroups(
                logits.data(), tokens, expert_num, expert_num / p.weights->n_group, p.weights->topk_group, st);
            TM_CUDA_CHECK(cudaGetLastError());
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
    TM_CUDA_CHECK(cudaGetLastError());

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
        TM_CUDA_CHECK(
            cudaMemcpyAsync(offsets_.data(), h_offsets_.data(), sizeof(int) * (expert_num + 1), cudaMemcpyDefault, st));
    }

    temp_ = Tensor{{p.weights->experts_per_token * tokens, hidden_dim}, p.input.dtype(), p.input.device()};

    auto indices = f2n_.slice(0, tokens * p.weights->experts_per_token);
    auto offsets = offsets_.slice(0, expert_num + 1);

    if (block.w1w3) {
        // Fused w1w3 path
        Tensor inter;
        TM_SCOPE_CALL(linear_.Forward(p.input, *block.w1w3, indices, offsets_, inter));

        if (!block.is_fused_silu) {
            Activation(inter, block.w1w3->bias, f2E_, block.act_type, st);
            TM_CUDA_CHECK(cudaGetLastError());
        }

        TM_SCOPE_CALL(linear_.Forward(inter.slice({0, 0}, {-1, inter_size}), *block.w2, {}, offsets, temp_));
    }
    else {
        // Separate w1/w3 path
        Tensor gating;
        TM_SCOPE_CALL(linear_.Forward(p.input, *block.w1, indices, offsets_, gating));

        Tensor up;
        TM_SCOPE_CALL(linear_.Forward(p.input, *block.w3, indices, offsets_, up));

        Activation(gating, up, block.act_type, st);
        TM_CUDA_CHECK(cudaGetLastError());

        TM_SCOPE_CALL(linear_.Forward(gating, *block.w2, {}, offsets, temp_));
    }

    if (moe.shared_gate) {
        shared_scales_ = Gate(p.input, *moe.shared_gate);
    }
}

void MoeFfnTpImpl::Combine(MoeFfnLayer::ForwardParam& p)
{
    TM_FUNCTION_SCOPE();
    auto& moe = *p.weights;

    invokeMoeCombine(p.output,
                     temp_,
                     moe.block()->w2->bias,
                     scales_.data(),
                     en2f_.data(),
                     f2E_.data(),
                     shared_scales_.data_or((float*)nullptr),
                     p.weights->experts_per_token,
                     1.f / tp_size_,
                     p.scale,
                     core::Context::stream().handle());
    TM_CUDA_CHECK(cudaGetLastError());

    temp_          = {};
    shared_scales_ = {};
}

class MoeFfnAgRsImpl final: public MoeFfnLayerImpl {
public:
    MoeFfnAgRsImpl(const EngineParam& engine, const Context& ctx):
        MoeFfnLayerImpl(ctx), ep_size_(engine.ep_size), ep_rank_(engine.ep_rank), backend_(engine.all2all_backend)
    {
        TM_CHECK_GT(ep_size_, 1);
        TM_CHECK_EQ(engine.mlp_tp_size, 1);
        TM_CHECK_EQ(backend_, "allgather_reducescatter");
    }

    void Forward(MoeFfnLayer::ForwardParam&) override
    {
        ReportUnsupported();
    }

    void Combine(MoeFfnLayer::ForwardParam&) override
    {
        ReportUnsupported();
    }

private:
    void ReportUnsupported() const
    {
        TM_LOG_FATAL(
            "MoeFfnAgRsImpl is a stub: resolve AllreduceResidualRMSnorm and ag-rs ownership before enabling EP MoE");
    }

    int         ep_size_;
    int         ep_rank_;
    std::string backend_;
};

MoeFfnLayer::MoeFfnLayer(const EngineParam& engine, const Context& ctx)
{
    if (engine.ep_size <= 1) {
        impl_ = std::make_unique<MoeFfnTpImpl>(engine, ctx);
        return;
    }

    if (engine.all2all_backend == "allgather_reducescatter") {
        impl_ = std::make_unique<MoeFfnAgRsImpl>(engine, ctx);
        return;
    }

    TM_LOG_FATAL("Unsupported MoE EP all2all backend: {}", engine.all2all_backend);
}

MoeFfnLayer::~MoeFfnLayer() = default;

void MoeFfnLayer::Forward(ForwardParam& p)
{
    impl_->Forward(p);
}

void MoeFfnLayer::Combine(ForwardParam& p)
{
    impl_->Combine(p);
}

}  // namespace turbomind
