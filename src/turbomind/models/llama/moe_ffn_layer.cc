// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/kernels/gemm/moe_ep_utils.h"
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
    ep_size_(engine.ep_size > 0 ? engine.ep_size : 1),
    max_token_num_(engine.max_forward_token_num * engine.attn_dp_size),
    ll_max_tokens_per_rank_(engine.ll_max_tokens_per_rank),
    is_warm_up_(*ctx.is_warm_up),
    linear_(*ctx.linear),
    expert_ffn_(std::make_unique<LlamaFfnLayer>(ctx)),
    d_comm_(ctx.comm.d_comm)
{
}

void MoeFfnLayer::Init(ForwardParam& p)
{
    const int expert_num        = p.weights->num_experts();        // global
    const int local_expert_num  = p.weights->local_num_experts();  // resident on this rank
    const int experts_per_token = p.weights->experts_per_token;

    h_offsets_ = {local_expert_num + 1, kCPU};

    const int pad_token_num = (max_token_num_ + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    masks_   = {local_expert_num * pad_token_num, kDEVICE};
    f2n_     = {experts_per_token * max_token_num_, kDEVICE};
    f2E_     = {experts_per_token * max_token_num_, kDEVICE};
    en2f_    = {experts_per_token * max_token_num_, kDEVICE};
    scales_  = {experts_per_token * max_token_num_, kDEVICE};
    offsets_ = {local_expert_num + 1, kDEVICE};
    accum_   = {local_expert_num * kMoeGateMaxTiles, kDEVICE};

    if (ep_size_ > 1) {
        // TODO: support Glm4MoeForCausalLM Routing under EP
        TM_CHECK_NE(p.weights->topk_method, "noaux_tc") << "This model doesn't support EP";

        ep_mode_      = comm::EpMode::kNull;
        topk_weights_ = {max_token_num_ * experts_per_token, kDEVICE};
        topk_idx_     = {max_token_num_ * experts_per_token, kDEVICE};
        Clear(offsets_);

        const int max_ll_recv_tokens = local_expert_num * ll_max_tokens_per_rank_ * d_comm_->n_ranks(0);
        if (f2n_.size() < max_ll_recv_tokens) {
            f2n_ = {max_ll_recv_tokens, kDEVICE};
            f2E_ = {max_ll_recv_tokens, kDEVICE};
        }
    }

    initialized_ = true;
}

Tensor_<float> MoeFfnLayer::Gate(const Tensor& input, const LinearWeight& gate)
{
    TM_FUNCTION_SCOPE();

    auto& w = gate.weight;
    TM_CHECK_EQ(input.shape(1), w.shape(0));
    Tensor_<float> logits{{input.shape(0), w.shape(1)}, kDEVICE};
    if (input.shape(0) > 0) {
        TM_SCOPE_CALL(linear_.Forward(input, gate, logits));
        ApplyBias(logits, gate.bias, core::Context::stream().handle());
        TM_CUDA_CHECK(cudaGetLastError());
    }
    return logits;
}

void MoeFfnLayer::RouteTP(ForwardParam& p, Tensor_<float>& logits)
{
    const int   tokens     = p.input.shape(0);
    const auto& moe        = *p.weights;
    const int   hidden_dim = TM_CHECK_NOTNULL(moe.block())->hidden_dim;

    const size_t padded     = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;
    const int    expert_num = moe.num_experts();

    TM_CHECK(expert_num);

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

    // input & output
    input_ = p.input;
    temp_  = Tensor{{p.weights->experts_per_token * tokens, hidden_dim}, p.input.dtype(), p.input.device()};
}

void MoeFfnLayer::RouteEP(ForwardParam& p, Tensor_<float>& logits)
{
    TM_CHECK(ep_mode_ == comm::EpMode::kNull);

    const int   tokens     = p.input.shape(0);
    const auto& moe        = *p.weights;
    const int   expert_num = moe.num_experts();  // global
    const int   hidden_dim = TM_CHECK_NOTNULL(moe.block())->hidden_dim;
    const auto  st         = core::Context::stream().handle();

    bool softmax = true;
    if (p.weights->topk_method == "group_limited_greedy") {
        if (tokens > 0) {
            invokeMoeSoftmaxMaskTopKGroups(
                logits.data(), tokens, expert_num, expert_num / p.weights->n_group, p.weights->topk_group, st);
            TM_CUDA_CHECK(cudaGetLastError());
        }
        softmax = false;
    }

    Tensor_<float>   topk_weights{topk_weights_, {tokens, p.weights->experts_per_token}};
    Tensor_<int64_t> topk_idx{topk_idx_, {tokens, p.weights->experts_per_token}};
    invokeMoeGateEp(topk_weights.data_or((float*)nullptr),
                    topk_idx.data_or((int64_t*)nullptr),
                    logits.data_or((float*)nullptr),
                    tokens,
                    expert_num,
                    p.weights->experts_per_token,
                    softmax,
                    p.weights->norm_topk_prob,
                    p.weights->routed_scale,
                    st);
    TM_CUDA_CHECK(cudaGetLastError());

    input_ = empty_like(p.input);
    if (p.input.shape(0) > 0) {
        cudaMemcpyAsync(input_.raw_data(), p.input.raw_data(), p.input.byte_size(), cudaMemcpyDefault, st);
        TM_CUDA_CHECK(cudaGetLastError());
    }

    ep_mode_ =
        p.max_tokens_per_rank <= ll_max_tokens_per_rank_ ? comm::EpMode::kLowLatency : comm::EpMode::kHighThroughput;

    // HT `num_worst_tokens` is the upper bound on distinct tokens received by this rank after dispatch.
    const int num_worst_tokens = ep_mode_ == comm::EpMode::kLowLatency ? ll_max_tokens_per_rank_ * expert_num :
                                                                         p.max_tokens_per_rank * d_comm_->n_ranks(0);
    const int num_worst_flat_tokens =
        ep_mode_ == comm::EpMode::kLowLatency ? num_worst_tokens : num_worst_tokens * p.weights->experts_per_token;
    TM_CHECK_LE(num_worst_flat_tokens, f2n_.size());
    TM_CHECK_LE(num_worst_flat_tokens, f2E_.size());
    TM_CHECK_LE(p.max_tokens_per_rank * d_comm_->n_ranks(0) * p.weights->experts_per_token, en2f_.size());

    const bool use_fp8       = false;
    const bool output_scales = false;
    const bool zero_copy     = ep_mode_ == comm::EpMode::kLowLatency;

    comm::EpDispatchInput dispatch_input{
        ep_mode_, input_, topk_weights, topk_idx, p.ht_buffer, num_worst_tokens, use_fp8, output_scales, zero_copy};
    comm::EpDispatchOutput dispatch_output{{}, {}, {}, f2n_, f2E_, en2f_, offsets_, {}};
    d_comm_->Dispatch(dispatch_input, dispatch_output, 0);
    TM_CUDA_CHECK(cudaGetLastError());

    input_ = dispatch_output.out_x;
    if (dispatch_output.rdma) {
        temp_ = dispatch_output.rdma.view({-1, hidden_dim});
    }
    else {
        temp_ = Tensor{{num_worst_flat_tokens, hidden_dim}, p.input.dtype(), p.input.device()};
    }

    // keep dispatch_output for combine
    dispatch_output_ = std::make_unique<comm::EpDispatchOutput>(dispatch_output);
}

void MoeFfnLayer::SetWarmup(ForwardParam& p)
{
    const int   tokens     = p.input.shape(0);
    const auto& moe        = *p.weights;
    const int   expert_num = moe.local_num_experts();
    const int   hidden_dim = TM_CHECK_NOTNULL(moe.block())->hidden_dim;
    const auto  st         = core::Context::stream().handle();

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
    // use first token
    TM_CUDA_CHECK(cudaMemsetAsync(f2n_.data(), 0, sizeof(int) * expert_ids.size(), st));
    TM_CUDA_CHECK(cudaMemsetAsync(f2E_.data(), 0, sizeof(int) * expert_ids.size(), st));

    // input & output
    input_ = p.input;
    temp_  = Tensor{{p.weights->experts_per_token * tokens, hidden_dim}, p.input.dtype(), p.input.device()};
}

void MoeFfnLayer::Forward(ForwardParam& p)
{
    TM_FUNCTION_SCOPE();
    if (!initialized_) {
        Init(p);
    }

    const auto& moe = *p.weights;

    auto logits = Gate(p.input, *moe.gate.get());
    TM_DEBUG_TENSOR(logits, "logits", 2);

    if (is_warm_up_) {
        SetWarmup(p);
    }
    else if (ep_size_ == 1) {
        RouteTP(p, logits);
    }
    else {
        RouteEP(p, logits);
    }

    if (temp_.shape(0) != 0) {
        ForwardFused(p);
    }

    if (moe.shared_gate) {
        shared_scales_ = Gate(p.input, *moe.shared_gate);
    }
}

void MoeFfnLayer::ForwardFused(ForwardParam& p)
{
    TM_FUNCTION_SCOPE();
    TM_CHECK_GT(temp_.shape(0), 0);

    const auto& moe              = *p.weights;
    const auto  st               = core::Context::stream().handle();
    const auto& block            = *TM_CHECK_NOTNULL(moe.block());
    const int   local_expert_num = moe.local_num_experts();

    const int* num_flat_tok_ptr = (ep_mode_ != comm::EpMode::kNull) ? offsets_.data() + local_expert_num : nullptr;

    auto indices = f2n_.slice(0, temp_.shape(0));
    auto offsets = offsets_.slice(0, local_expert_num + 1);

    std::optional<Tensor> scales;
    if (dispatch_output_ && dispatch_output_->out_x_scales) {
        scales = dispatch_output_->out_x_scales;
    }

    if (block.w1w3) {
        // Fused w1w3 path
        Tensor inter;
        TM_SCOPE_CALL(linear_.Forward(input_, scales, *block.w1w3, indices, offsets_, inter));

        if (!block.is_fused_silu) {
            Activation(inter, block.w1w3->bias, f2E_, block.act_type, num_flat_tok_ptr, st);
            TM_CUDA_CHECK(cudaGetLastError());
        }

        TM_SCOPE_CALL(linear_.Forward(inter.slice({0, 0}, {-1, block.inter_size}), *block.w2, {}, offsets, temp_));
    }
    else {
        // Separate w1/w3 path
        Tensor gating;
        TM_SCOPE_CALL(linear_.Forward(input_, scales, *block.w1, indices, offsets_, gating));

        Tensor up;
        TM_SCOPE_CALL(linear_.Forward(input_, scales, *block.w3, indices, offsets_, up));

        Activation(gating, up, block.act_type, num_flat_tok_ptr, st);
        TM_CUDA_CHECK(cudaGetLastError());

        TM_SCOPE_CALL(linear_.Forward(gating, *block.w2, {}, offsets, temp_));
    }
}

void MoeFfnLayer::Combine(ForwardParam& p)
{
    TM_FUNCTION_SCOPE();
    if (is_warm_up_) {
        // pass
    }
    else if (ep_size_ == 1) {
        CombineTP(p);
    }
    else {
        CombineEP(p);
    }
    TM_CUDA_CHECK(cudaGetLastError());

    input_         = {};
    temp_          = {};
    shared_scales_ = {};

    dispatch_output_.reset();
    ep_mode_ = comm::EpMode::kNull;
}

void MoeFfnLayer::CombineTP(ForwardParam& p)
{
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
}

void MoeFfnLayer::CombineEP(ForwardParam& p)
{
    TM_CHECK(ep_mode_ != comm::EpMode::kNull);

    const auto& moe = *p.weights;
    auto        st  = core::Context::stream().handle();

    // Local reduce
    Tensor input = (input_.dtype() == kFloat8_e4m3 && ep_mode_ == comm::EpMode::kHighThroughput) ?
                       Tensor{input_.layout(), temp_.dtype(), kDEVICE} :
                       input_;
    if (ep_mode_ == comm::EpMode::kHighThroughput) {
        invokeMoeLocalCombineEp(input,
                                temp_,
                                moe.block()->w2->bias,
                                dispatch_output_->out_topk_weights.data_or((float*)nullptr),
                                en2f_.data(),
                                f2E_.data(),
                                p.weights->experts_per_token,
                                dispatch_output_->num_distinct_tokens_ptr,
                                st);
    }
    else {
        const int  local_expert_num = moe.local_num_experts();
        const int* num_flat_tok_ptr = offsets_.data() + local_expert_num;
        invokeMoeAddBias(temp_, moe.block()->w2->bias, f2E_.data(), num_flat_tok_ptr, st);
    }
    TM_CUDA_CHECK(cudaGetLastError());

    // Moe Reduce
    comm::EpCombineInput  combine_input{ep_mode_, input, dispatch_output_->handle};
    comm::EpCombineOutput combine_output{};
    if (ep_mode_ == comm::EpMode::kLowLatency) {
        combine_input.x            = temp_;
        combine_input.topk_idx     = Tensor{topk_idx_, {p.input.shape(0), p.weights->experts_per_token}};
        combine_input.topk_weights = Tensor{topk_weights_, {p.input.shape(0), p.weights->experts_per_token}};
        combine_input.zero_copy    = static_cast<bool>(dispatch_output_->rdma);
    }
    d_comm_->Combine(combine_input, combine_output, 0);
    TM_CUDA_CHECK(cudaGetLastError());

    // Merge shared expert output.
    invokeMoeCombineOutputEp(p.output,  //
                             combine_output.out_x,
                             shared_scales_.data_or((float*)nullptr),
                             p.scale,
                             st);
}

}  // namespace turbomind
