// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/core/context.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/kernels/gemm/moe_ep_utils.h"
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
    inter_size_(param.inter_size / (engine.ep_size > 1 ? 1 : engine.mlp_tp_size)),
    hidden_dim_(model.hidden_units),
    tp_size_(engine.mlp_tp_size),
    ep_size_(engine.ep_size),
    param_(param),
    is_warm_up_{*ctx.is_warm_up},
    linear_(*ctx.linear),
    d_comm_(ctx.comm.d_comm)
{
    TM_CHECK(!param.expert_num.empty());

    const int max_local_expert_num =
        *std::max_element(param.expert_num.begin(), param.expert_num.end()) / engine.ep_size;

    if (param_.method == MoeParam::kFused) {
        // pass
    }
    else {
        expert_ffn_ = std::make_unique<LlamaFfnLayer>(model, ctx);
    }

    h_offsets_ = {max_local_expert_num + 1, kCPUpinned};

    const int max_token_num = engine.max_forward_token_num * engine.attn_dp_size;
    const int pad_token_num = (max_token_num + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    // dbg(inter_size_,
    //     hidden_dim_,
    //     tp_size_,
    //     param_.method,
    //     param.expert_num,
    //     max_local_expert_num,
    //     max_token_num,
    //     pad_token_num,
    //     param_.experts_per_token);

    masks_   = {max_local_expert_num * pad_token_num, kDEVICE};
    f2n_     = {param_.experts_per_token * max_token_num, kDEVICE};
    f2E_     = {param_.experts_per_token * max_token_num, kDEVICE};
    en2f_    = {param_.experts_per_token * max_token_num, kDEVICE};
    scales_  = {param_.experts_per_token * max_token_num, kDEVICE};
    offsets_ = {max_local_expert_num + 1, kDEVICE};
    accum_   = {max_local_expert_num * kMoeGateMaxTiles, kDEVICE};

    if (ep_size_ > 1) {
        // TODO: support Glm4MoeForCausalLM Routing
        TM_CHECK_NE(param_.topk_method, "noaux_tc") << "This model doesn't support EP";

        ep_mode_      = comm::EpMode::kNull;
        topk_weights_ = {max_token_num * param_.experts_per_token, kDEVICE};
        topk_idx_     = {max_token_num * param_.experts_per_token, kDEVICE};
        Clear(offsets_);
    }
}

Tensor_<float> MoeFfnLayer::Gate(const Tensor& input, const LlamaDenseWeight& gate)
{
    auto& weight = gate.weight;
    TM_CHECK_EQ(input.shape(1), weight.shape(0));
    Tensor_<float> logits{{input.shape(0), weight.shape(1)}, kDEVICE};
    if (input.shape(0) > 0) {
        linear_.Forward(input, gate, logits);
        sync_check_cuda_error();
        ApplyBias(logits, gate.bias, core::Context::stream().handle());
        sync_check_cuda_error();
    }
    return logits;
}

void MoeFfnLayer::RouteTP(ForwardParam& p, Tensor_<float>& logits)
{
    const int   tokens = p.input.shape(0);
    const auto& moe    = *p.weights;

    const size_t padded     = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;
    const int    expert_num = moe.experts.size();

    FT_CHECK(expert_num);

    const auto st = core::Context::stream().handle();

    if (param_.topk_method == "noaux_tc") {
        // invokeMoeGate_NoAuxTC clears accum and masks internally
        TM_CHECK_EQ(param_.n_group, 1);
        TM_CHECK_EQ(param_.topk_group, 1);
        const float* correction_bias =
            (moe.score_correction_bias.size() > 0) ? moe.score_correction_bias.data<float>() : nullptr;
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
                              param_.experts_per_token,
                              param_.norm_topk_prob,
                              param_.routed_scale,
                              param_.scoring_func == "sigmoid",
                              st);
    }
    else {
        // V2: accum must be cleared by caller; masks cleared internally
        check_cuda_error(cudaMemsetAsync(accum_.data(), 0, sizeof(int) * expert_num * kMoeGateMaxTiles, st));

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
    }
    sync_check_cuda_error();

    // input & output
    input_ = p.input;
    temp_  = Tensor{{param_.experts_per_token * tokens, hidden_dim_}, p.input.dtype(), p.input.device()};
}

void MoeFfnLayer::RouteEP(ForwardParam& p, Tensor_<float>& logits)
{
    TM_CHECK(ep_mode_ != comm::EpMode::kNull);

    const int   tokens     = p.input.shape(0);
    const auto& moe        = *p.weights;
    const int   expert_num = moe.experts.size() * ep_size_;
    const auto  st         = core::Context::stream().handle();

    bool softmax = true;
    if (param_.topk_method == "group_limited_greedy") {
        if (tokens > 0) {
            invokeMoeSoftmaxMaskTopKGroups(
                logits.data(), tokens, expert_num, expert_num / param_.n_group, param_.topk_group, st);
            sync_check_cuda_error();
        }
        softmax = false;
    }

    Tensor_<float>   topk_weights{topk_weights_, {tokens, param_.experts_per_token}};
    Tensor_<int64_t> topk_idx{topk_idx_, {tokens, param_.experts_per_token}};
    invokeMoeGateEp(topk_weights.data_or((float*)nullptr),
                    topk_idx.data_or((int64_t*)nullptr),
                    logits.data_or((float*)nullptr),
                    tokens,
                    moe.experts.size() * ep_size_,
                    param_.experts_per_token,
                    softmax,
                    param_.norm_topk_prob,
                    param_.routed_scale,
                    core::Context::stream().handle());
    sync_check_cuda_error();

    ep_mode_ = p.max_tokens_per_rank <= param_.ll_max_tokens_per_rank ? comm::EpMode::kLowLatency :
                                                                        comm::EpMode::kHighThroughput;
    comm::EpDispatchInput  dispatch_input{ep_mode_, p.input, topk_weights, topk_idx};
    comm::EpDispatchOutput dispatch_output{{}, {}, f2n_, f2E_, en2f_, offsets_, {}};
    d_comm_->Dispatch(dispatch_input, dispatch_output, 0);
    sync_check_cuda_error();

    input_ = dispatch_output.out_x;
    temp_  = Tensor{{dispatch_output.out_expert_token_num, hidden_dim_}, p.input.dtype(), p.input.device()};

    // keep dispatch_output for combine
    dispatch_output_ = std::make_unique<comm::EpDispatchOutput>(dispatch_output);
}

void MoeFfnLayer::SetWarmup(ForwardParam& p)
{
    const int  tokens     = p.input.shape(0);
    const int  expert_num = p.weights->experts.size();
    const auto st         = core::Context::stream().handle();

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
    check_cuda_error(cudaMemcpyAsync(offsets_.data(),
                                     h_offsets_.data(),
                                     sizeof(int) * (expert_num + 1),
                                     cudaMemcpyDefault,
                                     core::Context::stream().handle()));
    // use first token
    check_cuda_error(cudaMemsetAsync(f2n_.data(), 0, sizeof(int) * expert_ids.size(), st));
    check_cuda_error(cudaMemsetAsync(f2E_.data(), 0, sizeof(int) * expert_ids.size(), st));

    // input & output
    input_ = p.input;
    temp_  = Tensor{{param_.experts_per_token * tokens, hidden_dim_}, p.input.dtype(), p.input.device()};
}

void MoeFfnLayer::Forward(ForwardParam& p)
{
    const int   tokens     = p.input.shape(0);
    const auto& moe        = *p.weights;
    const int   expert_num = moe.experts.size() * ep_size_;

    auto logits = Gate(p.input, moe.gate);
    TM_DEBUG_TENSOR(logits, "logits", 2);
    // dump_logits(tokens, layer_id);

    const auto st = core::Context::stream().handle();

    if (is_warm_up_) {
        SetWarmup(p);
    }
    else if (ep_size_ == 1) {
        RouteTP(p, logits);
    }
    else {
        RouteEP(p, logits);
    }

    if (input_.shape(0) == 0) {
        // pass
    }
    else if (param_.method == MoeParam::kNaive) {
        ForwardNative(p);
    }
    else {
        ForwardFused(p);
    }

    if (moe.shared_gate.weight) {
        shared_scales_ = Gate(p.input, moe.shared_gate);
    }
}

void MoeFfnLayer::ForwardNative(ForwardParam& p)
{
    TM_CHECK_EQ(ep_size_, 1);
    TM_CHECK_GT(input_.shape(0), 0);

    const auto& moe              = *p.weights;
    const auto  st               = core::Context::stream().handle();
    const int   tokens           = input_.shape(0);
    const int   local_expert_num = moe.experts.size();

    invokeMoeDispatch(temp_, input_, f2n_.data(), param_.experts_per_token, st);
    sync_check_cuda_error();

    check_cuda_error(cudaMemcpyAsync(
        h_offsets_.data(), offsets_.data(), sizeof(int) * (local_expert_num + 1), cudaMemcpyDefault, st));

    check_cuda_error(cudaStreamSynchronize(st));

    TM_CHECK_EQ(h_offsets_[local_expert_num], tokens * param_.experts_per_token);

    for (int i = 0; i < local_expert_num; ++i) {
        if (int count = h_offsets_[i + 1] - h_offsets_[i]) {
            auto io = temp_.slice({h_offsets_[i], 0}, {count, -1});
            expert_ffn_->forward({io, io, moe.experts.at(i).get(), p.layer_id});
            sync_check_cuda_error();
        }
    }
}

void MoeFfnLayer::ForwardFused(ForwardParam& p)
{
    TM_CHECK_GT(input_.shape(0), 0);

    const auto& moe              = *p.weights;
    const auto  st               = core::Context::stream().handle();
    const int   tokens           = input_.shape(0);
    const int   local_expert_num = moe.experts.size();

    auto& block = moe.block;

    auto indices = f2n_.slice(0, temp_.shape(0));
    auto offsets = offsets_.slice(0, local_expert_num + 1);

    Tensor inter = linear_.Forward(input_, block.fused_gating_intermediate, indices, offsets);
    sync_check_cuda_error();

    if (!block.is_fused_silu) {
        Activation(inter, block.fused_gating_intermediate.bias, f2E_, moe.block.act_type, st);
        sync_check_cuda_error();
    }

    linear_.Forward(inter.slice({0, 0}, {-1, inter_size_}), block.output, {}, offsets, temp_);
    sync_check_cuda_error();
}

void MoeFfnLayer::Combine(ForwardParam& p)
{
    if (is_warm_up_) {
        // pass
    }
    else if (ep_size_ == 1) {
        CombineTP(p);
    }
    else {
        CombineEP(p);
    }
    sync_check_cuda_error();

    input_         = {};
    temp_          = {};
    shared_scales_ = {};

    dispatch_output_.reset();
    ep_mode_ = comm::EpMode::kNull;
}

void MoeFfnLayer::CombineTP(ForwardParam& p)
{
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
}

void MoeFfnLayer::CombineEP(ForwardParam& p)
{
    TM_CHECK(ep_mode_ != comm::EpMode::kNull);
    auto st = core::Context::stream().handle();
    // Local reduce
    if (ep_mode_ == comm::EpMode::kHighThroughput) {
        invokeMoeLocalCombineEp(input_,
                                temp_,
                                p.weights->block.output.bias,
                                dispatch_output_->out_topk_weights.data_or((float*)nullptr),
                                en2f_.data(),
                                f2E_.data(),
                                param_.experts_per_token,
                                st);
    }
    else {
        invokeMoeAddBias(temp_, p.weights->block.output.bias, f2E_.data(), st);
    }
    sync_check_cuda_error();

    // Moe Reduce
    comm::EpCombineInput  combine_input{ep_mode_, input_, dispatch_output_->handle};
    comm::EpCombineOutput combine_output{};
    if (ep_mode_ == comm::EpMode::kLowLatency) {
        combine_input.x            = temp_;
        combine_input.topk_idx     = Tensor{topk_idx_, {p.input.shape(0), param_.experts_per_token}};
        combine_input.topk_weights = Tensor{topk_weights_, {p.input.shape(0), param_.experts_per_token}};
    }
    d_comm_->Combine(combine_input, combine_output, 0);
    sync_check_cuda_error();

    // Merge shared expert output.
    invokeMoeCombineOutputEp(p.output,  //
                             combine_output.out_x,
                             shared_scales_.data_or((float*)nullptr),
                             p.scale,
                             st);
}

}  // namespace turbomind
