// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>
#include <string>

#include <cuda_runtime.h>

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/gemm/moe_ep_utils.h"
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
    MoeFfnAgRsImpl(const EngineParam& engine, const Context& ctx);

    void Forward(MoeFfnLayer::ForwardParam& p) override;

    void Combine(MoeFfnLayer::ForwardParam& p) override;

private:
    void Init(MoeFfnLayer::ForwardParam& p);

    void ForwardFused(MoeFfnLayer::ForwardParam& p);

    const int                   ep_size_;
    const int                   ep_rank_;
    const int                   max_token_num_;
    const std::string           backend_;
    comm::DeviceCommImpl* const d_comm_;
    Allocator                   symm_allocator_;

    bool initialized_ = false;

    Buffer_<float>   topk_weights_;
    Buffer_<int64_t> topk_idx_;
    Buffer_<int>     f2n_;
    Buffer_<int>     f2E_;
    Buffer_<int>     en2f_;
    Buffer_<int>     offsets_;
    Buffer_<int8_t>  masks_;
    Buffer_<int>     accum_;

    Tensor         temp_;
    Tensor_<float> shared_scales_;
};

MoeFfnAgRsImpl::MoeFfnAgRsImpl(const EngineParam& engine, const Context& ctx):
    MoeFfnLayerImpl(ctx),
    ep_size_(engine.ep_size),
    ep_rank_(engine.ep_rank),
    max_token_num_(engine.max_forward_token_num * engine.attn_dp_size),
    backend_(engine.all2all_backend),
    d_comm_(ctx.comm.d_comm),
    symm_allocator_(GetSymmAllocator(ctx.comm.d_comm))
{
}

void MoeFfnAgRsImpl::Init(MoeFfnLayer::ForwardParam& p)
{
    const auto& moe               = *p.weights;
    const int   local_expert_num  = moe.num_local_experts();
    const int   experts_per_token = moe.experts_per_token;
    const int   pad_token_num     = (max_token_num_ + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    TM_CHECK(p.weights->topk_method != "noaux_tc")
        << "MoeFfnAgRsImpl does not support topk_method=noaux_tc because it requires noaux-specific scoring and "
           "correction-bias handling";

    topk_weights_ = {max_token_num_ * experts_per_token, symm_allocator_};
    topk_idx_     = {max_token_num_ * experts_per_token, symm_allocator_};
    masks_        = {local_expert_num * pad_token_num, kDEVICE};
    f2n_          = {max_token_num_ * experts_per_token, kDEVICE};
    f2E_          = {max_token_num_ * experts_per_token, kDEVICE};
    en2f_         = {max_token_num_ * experts_per_token, kDEVICE};
    offsets_      = {local_expert_num + 1, kDEVICE};
    accum_        = {local_expert_num * kMoeGateMaxTiles, kDEVICE};

    initialized_ = true;
}

void MoeFfnAgRsImpl::Forward(MoeFfnLayer::ForwardParam& p)
{
    TM_FUNCTION_SCOPE();

    if (!initialized_) {
        Init(p);
    }

    const auto& moe               = *p.weights;
    const int   tokens            = p.global_hidden_states.shape(0);
    const int   experts_per_token = moe.experts_per_token;
    const int   expert_num        = moe.num_experts();
    const int   local_expert_num  = moe.num_local_experts();
    const int   hidden_dim        = TM_CHECK_NOTNULL(moe.block())->hidden_dim;
    const auto  st                = core::Context::stream().handle();

    TM_CHECK_LE(tokens, max_token_num_);

    std::vector<size_t> topk_elem_counts(ep_size_);
    size_t              local_topk_offset{};
    for (int rank = 0; rank < ep_size_; ++rank) {
        topk_elem_counts[rank] = TM_CHECK_NOTNULL(p.ep_elem_counts)->at(rank) / hidden_dim * experts_per_token;
        local_topk_offset += rank < ep_rank_ ? topk_elem_counts[rank] : 0;
    }

    auto local_topk_weights = topk_weights_.data() + local_topk_offset;
    auto local_topk_idx     = topk_idx_.data() + local_topk_offset;

    if (auto local_tokens = p.input.shape(0); local_tokens > 0) {
        auto logits = Gate(p.input, *moe.gate.get());
        TM_DEBUG_TENSOR(logits, "logits", 2);

        bool softmax = true;
        if (p.weights->topk_method == "group_limited_greedy") {
            invokeMoeSoftmaxMaskTopKGroups(
                logits.data(), local_tokens, expert_num, expert_num / p.weights->n_group, p.weights->topk_group, st);
            TM_CUDA_CHECK(cudaGetLastError());
            softmax = false;
        }

        invokeMoeGateEp(local_topk_weights,
                        local_topk_idx,
                        logits.data(),
                        local_tokens,
                        expert_num,
                        experts_per_token,
                        softmax,
                        p.weights->norm_topk_prob,
                        p.weights->routed_scale,
                        st);
        TM_CUDA_CHECK(cudaGetLastError());
    }

    d_comm_->GroupStart();
    d_comm_->AllGatherV(
        p.input.raw_data(), p.global_hidden_states.raw_data(), *p.ep_elem_counts, p.input.dtype(), 0, st);
    d_comm_->AllGatherV(local_topk_weights, topk_weights_.data(), topk_elem_counts, kFloat32, 0, st);
    d_comm_->AllGatherV(local_topk_idx, topk_idx_.data(), topk_elem_counts, kInt64, 0, st);
    d_comm_->GroupEnd();
    TM_CUDA_CHECK(cudaGetLastError());

    invokeMoeBuildAgRsRoutingMap(f2n_.data(),
                                 f2E_.data(),
                                 en2f_.data(),
                                 offsets_.data(),
                                 masks_.data(),
                                 accum_.data(),
                                 topk_idx_.data(),
                                 tokens,
                                 experts_per_token,
                                 moe.local_expert_offset(),
                                 local_expert_num,
                                 st);
    TM_CUDA_CHECK(cudaGetLastError());

    temp_ = Tensor{{tokens * experts_per_token, hidden_dim}, p.input.dtype(), p.input.device()};
    ForwardFused(p);

    shared_scales_ = {};
    if (moe.shared_gate && p.input.shape(0) > 0) {
        shared_scales_ = Gate(p.input, *moe.shared_gate);
    }
}

void MoeFfnAgRsImpl::ForwardFused(MoeFfnLayer::ForwardParam& p)
{
    TM_FUNCTION_SCOPE();

    const auto& moe              = *p.weights;
    const auto& block            = *TM_CHECK_NOTNULL(moe.block());
    const int   local_expert_num = moe.num_local_experts();
    const auto  st               = core::Context::stream().handle();

    auto       input            = p.global_hidden_states;
    const int* num_valid_tokens = offsets_.data() + local_expert_num;
    auto       indices          = f2n_.slice(0, temp_.shape(0));
    auto       offsets          = offsets_.slice(0, local_expert_num + 1);
    const bool indices_padded   = true;

    if (block.w1w3) {
        Tensor inter;
        TM_SCOPE_CALL(linear_.Forward(input, *block.w1w3, indices, offsets, inter, indices_padded));

        if (!block.is_fused_silu) {
            Activation(inter, block.w1w3->bias, f2E_, block.act_type, num_valid_tokens, st);
            TM_CUDA_CHECK(cudaGetLastError());
        }

        TM_SCOPE_CALL(linear_.Forward(inter.slice({0, 0}, {-1, block.inter_size}), *block.w2, {}, offsets, temp_));
    }
    else {
        Tensor gating;
        TM_SCOPE_CALL(linear_.Forward(input, *block.w1, indices, offsets, gating, indices_padded));

        Tensor up;
        TM_SCOPE_CALL(linear_.Forward(input, *block.w3, indices, offsets, up, indices_padded));

        Activation(gating, up, block.act_type, num_valid_tokens, st);
        TM_CUDA_CHECK(cudaGetLastError());

        TM_SCOPE_CALL(linear_.Forward(gating, *block.w2, {}, offsets, temp_));
    }
}

void MoeFfnAgRsImpl::Combine(MoeFfnLayer::ForwardParam& p)
{
    TM_FUNCTION_SCOPE();

    const auto& moe               = *p.weights;
    const auto& block             = *TM_CHECK_NOTNULL(moe.block());
    const int   experts_per_token = moe.experts_per_token;
    const auto  st                = core::Context::stream().handle();

    invokeMoeLocalCombineEp(p.global_hidden_states,
                            temp_,
                            block.w2->bias,
                            topk_weights_.data(),
                            en2f_.data(),
                            f2E_.data(),
                            experts_per_token,
                            st);
    TM_CUDA_CHECK(cudaGetLastError());

    d_comm_->ReduceScatterV(
        p.global_hidden_states.raw_data(), p.output.raw_data(), *p.ep_elem_counts, p.output.dtype(), 0, st);
    TM_CUDA_CHECK(cudaGetLastError());

    if (p.shared_output) {
        invokeMoeCombineOutputEp(p.output, p.shared_output, shared_scales_.data_or((float*)nullptr), p.scale, st);
        TM_CUDA_CHECK(cudaGetLastError());
    }

    temp_          = {};
    shared_scales_ = {};
}

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
