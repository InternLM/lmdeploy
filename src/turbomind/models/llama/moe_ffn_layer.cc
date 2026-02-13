// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_type.h"
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
    ep_size_(engine.ep_size),
    ep_rank_(engine.ep_rank),
    inter_size_(param.inter_size / (engine.ep_size > 1 ? 1 : engine.mlp_tp_size)),
    hidden_dim_(model.hidden_units),
    tp_size_(engine.mlp_tp_size),
    param_(param),
    is_warm_up_{*ctx.is_warm_up},
    linear_(*ctx.linear)
{
    TM_CHECK(!param.expert_num.empty());

    const int max_local_expert_num = *std::max_element(param.expert_num.begin(), param.expert_num.end()) / ep_size_;

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
        comm::DeviceCommImpl* comm = TM_CHECK_NOTNULL(ctx.comm.d_comm);
        d_comm_                    = TM_CHECK_NOTNULL(dynamic_cast<comm::CudaIpcCommImpl*>(comm));

        topk_scales_       = {max_token_num * ep_size_ * param_.experts_per_token, kDEVICE};
        topk_experts_      = {max_token_num * ep_size_ * param_.experts_per_token, kDEVICE};
        token_idx_in_rank_ = {ep_size_ * (max_token_num + 2), kDEVICE};

        auto symm_alloc = GetSymmAllocator(ctx.comm.d_comm);
        symm_meta_      = {2 * ep_size_ * ep_size_, symm_alloc};
        symm_hidden_    = {byte_size(model.data_type, max_token_num * hidden_dim_), symm_alloc};
        symm_scales_    = {param_.experts_per_token * max_token_num, symm_alloc};
        symm_masks_     = {max_local_expert_num * max_token_num, symm_alloc};
    }
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

    const size_t padded           = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;
    const int    local_expert_num = moe.experts.size();
    const int    expert_num       = moe.experts.size() * ep_size_;

    FT_CHECK(expert_num);

    auto logits = (tokens > 0) ? Gate(p.input, moe.gate) : Tensor_<float>{};
    TM_DEBUG_TENSOR(logits, "logits", 2);

    const auto st = core::Context::stream().handle();

    check_cuda_error(cudaMemsetAsync(accum_.data(), 0, sizeof(int) * local_expert_num * kMoeGateMaxTiles, st));
    check_cuda_error(cudaMemsetAsync(masks_.data(), -1, sizeof(int8_t) * local_expert_num * padded, st));

    // dump_logits(tokens, layer_id);

    bool softmax = true;
    if (param_.topk_method == "group_limited_greedy") {
        if (tokens > 0) {
            invokeMoeSoftmaxMaskTopKGroups(
                logits.data(), tokens, expert_num, expert_num / param_.n_group, param_.topk_group, st);
        }
        sync_check_cuda_error();
        softmax = false;
    }

    input_ = p.input;

    if (is_warm_up_) {
        std::mt19937     g;
        const auto       expert_ids = SampleUniform(tokens, local_expert_num, param_.experts_per_token, g);
        std::vector<int> cnt(local_expert_num);
        for (const auto& x : expert_ids) {
            ++cnt[x];
        }
        h_offsets_[0] = 0;
        for (int i = 0; i < local_expert_num; ++i) {
            h_offsets_[i + 1] = h_offsets_[i] + cnt[i];
        }
        check_cuda_error(cudaMemcpyAsync(
            offsets_.data(), h_offsets_.data(), sizeof(int) * (local_expert_num + 1), cudaMemcpyDefault, st));

        check_cuda_error(cudaMemsetAsync(f2n_.data(), 0, sizeof(int) * tokens * param_.experts_per_token, st));
        temp_ = Tensor{{tokens * param_.experts_per_token, hidden_dim_}, input_.dtype(), input_.device()};
    }
    else if (ep_size_ == 1) {
        RouteTP(logits, tokens, padded, expert_num, softmax, st);
    }
    else {
        RouteEP(logits, tokens, padded, expert_num, softmax, st);
    }

    if (input_.shape(0) == 0) {}
    else if (param_.method == MoeParam::kFused) {
        ForwardFused(p);
    }
    else {
        ForwardNative(p);
    }

    if (moe.shared_gate.weight) {
        shared_scales_ = (tokens > 0) ? Gate(p.input, moe.shared_gate) : Tensor_<float>{};
    }
}

void MoeFfnLayer::RouteTP(Tensor_<float>& logits, int tokens, int padded, int expert_num, bool softmax, cudaStream_t st)
{
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

    temp_ = Tensor{{tokens * param_.experts_per_token, hidden_dim_}, input_.dtype(), input_.device()};
}

void MoeFfnLayer::RouteEP(Tensor_<float>& logits, int tokens, int padded, int expert_num, bool softmax, cudaStream_t st)
{
    const int local_expert_num = expert_num / ep_size_;

    invokeMoeGate_a2a(topk_scales_.data(),
                      topk_experts_.data(),
                      token_idx_in_rank_.data(),
                      logits.data_or((float*)nullptr),
                      tokens,
                      expert_num,
                      ep_size_,
                      param_.experts_per_token,
                      softmax,
                      param_.norm_topk_prob,
                      param_.routed_scale,
                      st);
    sync_check_cuda_error();
    (volatile int&)(*num_input) = (volatile int&)(*num_flat) = -1;
    d_comm_->AllToAllNotifyDispatch(
        symm_meta_.data(), num_input.mapped(), num_flat.mapped(), token_idx_in_rank_.data(), tokens, 0, st);
    sync_check_cuda_error();
    while ((volatile int&)*num_input == -1 || (volatile int&)*num_flat == -1) {}  // sync

    const int padded_ = round_up((volatile int&)*num_input, kMoeGateVecSize);
    check_cuda_error(cudaMemsetAsync(symm_masks_.data(), -1, sizeof(int8_t) * local_expert_num * padded_, st));
    sync_check_cuda_error();

    d_comm_->AllToAllDispatch(symm_hidden_.data(),
                              symm_scales_.data(),
                              symm_masks_.data(),
                              symm_meta_.data(),
                              input_.raw_data(),
                              topk_scales_.data(),
                              topk_experts_.data(),
                              token_idx_in_rank_.data(),
                              tokens,
                              hidden_dim_,
                              param_.experts_per_token,
                              input_.dtype(),
                              0,
                              st);
    sync_check_cuda_error();

    check_cuda_error(cudaMemsetAsync(en2f_.data(), -1, sizeof(int) * param_.experts_per_token * *num_input, st));

    invokeMoeScan_a2a(f2n_.data(),
                      f2E_.data(),
                      en2f_.data(),
                      offsets_.data(),
                      symm_masks_.data(),
                      accum_.data(),
                      *num_input,
                      padded_,
                      local_expert_num,
                      st);
    sync_check_cuda_error();

    // update context
    input_ = {symm_hidden_.view(input_.dtype()), {*num_input, hidden_dim_}};
    temp_  = {{*num_flat, hidden_dim_}, input_.dtype(), input_.device()};
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

void MoeFfnLayer::Combine(ForwardParam& p)
{
    auto& moe = *p.weights;

    if (is_warm_up_) {}
    else if (ep_size_ > 1) {
        CombineEP(p);
    }
    else {
        CombineTP(p);
    }
    sync_check_cuda_error();

    input_         = {};
    temp_          = {};
    shared_scales_ = {};
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
    // merge experts on the local rank.
    if (input_.shape(0) > 0) {
        invokeMoeCombine_a2a(input_,
                             temp_,
                             p.weights->block.output.bias,
                             symm_scales_.data(),
                             en2f_.data(),
                             f2E_.data(),
                             param_.experts_per_token,
                             core::Context::stream().handle());
        sync_check_cuda_error();
    }

    // merge experts on the remote ranks
    // TODO: support shared expert
    d_comm_->AllToAllCombine(p.output.raw_data(),
                             symm_meta_.data(),
                             input_.raw_data(),
                             token_idx_in_rank_.data(),
                             p.output.shape(0),
                             hidden_dim_,
                             p.output.dtype(),
                             0,
                             core::Context::stream().handle());
}

}  // namespace turbomind
