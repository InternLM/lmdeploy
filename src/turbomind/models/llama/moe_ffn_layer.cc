// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>

#include <cuda_runtime.h>

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/copy/copy.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#ifdef USE_NCCL
#include "src/turbomind/comm/nccl/deepep/moe_a2a_utils.h"
#include "src/turbomind/comm/nccl/deepep/token_dispatcher.h"
#endif

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

    virtual Tensor GetShardFfnInput(Tensor& global_hidden_states, const std::vector<int>& local_token_nums) = 0;

protected:
    Tensor_<float> Gate(const Tensor& input, const LinearWeight& gate);

    LlamaLinear& linear_;
};

Tensor_<float> MoeFfnLayerImpl::Gate(const Tensor& input, const LinearWeight& gate)
{
    TM_FUNCTION_SCOPE();

    TM_CHECK_EQ(input.shape(1), gate.input_dim);
    Tensor_<float> logits{{input.shape(0), gate.output_dim}, kDEVICE};
    TM_SCOPE_CALL(linear_.Forward(input, gate, logits));
    TM_SCOPE_CALL(ApplyBias(logits, gate.bias, core::Context::stream().handle()));
    return logits;
}

class MoeFfnDefaultImpl final: public MoeFfnLayerImpl {
public:
    MoeFfnDefaultImpl(const EngineParam& engine, const Context& ctx, const MoeWeight& weights);

    void Forward(MoeFfnLayer::ForwardParam& p) override;

    void Combine(MoeFfnLayer::ForwardParam& p) override;

    Tensor GetShardFfnInput(Tensor& global_hidden_states, const std::vector<int>& local_token_nums) override;

private:
    void Init(const MoeWeight& weights);

    const int tp_size_;
    const int ep_size_;
    const int ep_rank_;
    const int max_token_num_;
    int&      is_warm_up_;

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

    // When a MOE model enables EP inference and the model includes dense layers or shared experts, an additional stream
    // is used to perform sharding and cleaning of the FFN inputs.
    Stream clear_stream_;
    Event  clear_ready_event_;
    Event  clear_done_event_;
    bool   clear_pending_ = false;
};

MoeFfnDefaultImpl::MoeFfnDefaultImpl(const EngineParam& engine, const Context& ctx, const MoeWeight& weights):
    MoeFfnLayerImpl(ctx),
    tp_size_(engine.mlp_tp_size),
    ep_size_(engine.ep_size),
    ep_rank_(engine.ep_rank),
    max_token_num_(engine.max_forward_token_num * engine.attn_dp_size),
    is_warm_up_(*ctx.is_warm_up)
{
    if (ep_size_ > 1) {
        clear_stream_      = Stream::create();
        clear_ready_event_ = Event::create();
        clear_done_event_  = Event::create();
    }
    Init(weights);
}

void MoeFfnDefaultImpl::Init(const MoeWeight& weights)
{
    const int expert_num        = weights.num_experts();
    const int local_expert_num  = weights.num_local_experts();
    const int experts_per_token = weights.experts_per_token;

    h_offsets_ = {local_expert_num + 1, kCPU};

    const int pad_token_num = (max_token_num_ + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    masks_   = {expert_num * pad_token_num, kDEVICE};
    f2n_     = {experts_per_token * max_token_num_, kDEVICE};
    f2E_     = {experts_per_token * max_token_num_, kDEVICE};
    en2f_    = {experts_per_token * max_token_num_, kDEVICE};
    scales_  = {experts_per_token * max_token_num_, kDEVICE};
    offsets_ = {local_expert_num + 1, kDEVICE};
    accum_   = {expert_num * kMoeGateMaxTiles, kDEVICE};
}

void MoeFfnDefaultImpl::Forward(MoeFfnLayer::ForwardParam& p)
{
    TM_FUNCTION_SCOPE();

    const int   tokens = p.input.shape(0);
    const auto& moe    = *p.weights;

    const auto& block = *TM_CHECK_NOTNULL(moe.block());

    const int hidden_dim = block.hidden_dim;
    const int inter_size = block.inter_size;

    const size_t padded = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    const int expert_num        = moe.num_experts();
    const int local_expert_num  = moe.num_local_experts();
    const int expert_offset     = moe.local_expert_offset();
    const int experts_per_token = moe.experts_per_token;

    TM_CHECK(expert_num);

    auto logits = Gate(p.input, *moe.gate.get());

    TM_DEBUG_TENSOR(logits, "logits", 2);

    const auto st = core::Context::stream().handle();

    // en2f is the skip sentinel for combine: entries not written by routing (invalid
    // tokens, or shrunk batches with stale values) must read as -1 regardless of EP.
    TM_CUDA_CHECK(cudaMemsetAsync(en2f_.data(), -1, sizeof(int) * tokens * experts_per_token, st));

    if (p.weights->topk_method == "noaux_tc") {
        TM_CHECK_EQ(p.weights->n_group, 1);
        TM_CHECK_EQ(p.weights->topk_group, 1);
        const float* correction_bias = nullptr;
        if (moe.score_correction_bias) {
            correction_bias = moe.score_correction_bias.size() > 0 ? moe.score_correction_bias.data<float>() : nullptr;
        }
        TM_SCOPE_CALL(invokeMoeGate_NoAuxTC(f2n_.data(),
                                            f2E_.data(),
                                            en2f_.data(),
                                            offsets_.data(),
                                            scales_.data(),
                                            masks_.data(),
                                            accum_.data(),
                                            logits.data(),
                                            p.token_mask,
                                            correction_bias,
                                            tokens,
                                            padded,
                                            expert_num,
                                            experts_per_token,
                                            expert_offset,
                                            local_expert_num,
                                            p.weights->norm_topk_prob,
                                            p.weights->routed_scale,
                                            p.weights->scoring_func == "sigmoid",
                                            st));
    }
    else {
        // V2: accum must be cleared by caller; masks cleared internally
        TM_CUDA_CHECK(cudaMemsetAsync(accum_.data(), 0, sizeof(int) * expert_num * kMoeGateMaxTiles, st));

        bool softmax = true;
        if (p.weights->topk_method == "group_limited_greedy") {
            TM_SCOPE_CALL(invokeMoeSoftmaxMaskTopKGroups(
                logits.data(), tokens, expert_num, expert_num / p.weights->n_group, p.weights->topk_group, st));
            softmax = false;
        }

        TM_SCOPE_CALL(invokeMoeGate_V2(f2n_.data(),
                                       f2E_.data(),
                                       en2f_.data(),
                                       offsets_.data(),
                                       scales_.data(),
                                       masks_.data(),
                                       accum_.data(),
                                       logits.data(),
                                       p.token_mask,
                                       tokens,
                                       padded,
                                       expert_num,
                                       experts_per_token,
                                       expert_offset,
                                       local_expert_num,
                                       softmax,
                                       p.weights->norm_topk_prob,
                                       p.weights->routed_scale,
                                       st));
    }

    if (is_warm_up_) {
        std::mt19937     g;
        const auto       expert_ids = SampleUniform(tokens, local_expert_num, experts_per_token, g);
        std::vector<int> cnt(local_expert_num);
        for (const auto& x : expert_ids) {
            ++cnt[x];
        }
        h_offsets_[0] = 0;
        for (int i = 0; i < local_expert_num; ++i) {
            h_offsets_[i + 1] = h_offsets_[i] + cnt[i];
        }
        TM_CUDA_CHECK(cudaMemcpyAsync(
            offsets_.data(), h_offsets_.data(), sizeof(int) * (local_expert_num + 1), cudaMemcpyDefault, st));

        if (ep_size_ > 1) {
            const auto entries = static_cast<size_t>(tokens) * experts_per_token;
            TM_CUDA_CHECK(cudaMemsetAsync(f2n_.data(), 0, sizeof(int) * entries, st));
            TM_CUDA_CHECK(cudaMemsetAsync(f2E_.data(), 0, sizeof(int) * entries, st));
            TM_CUDA_CHECK(cudaMemsetAsync(en2f_.data(), -1, sizeof(int) * entries, st));
        }
    }

    temp_ = Tensor{{tokens * experts_per_token, hidden_dim}, p.input.dtype(), p.input.device()};

    // Masked-out tokens compact the routing tables even for ep_size == 1, so the
    // routed count is device-side offsets_[local_expert_num] in all cases.
    const int* num_valid_tokens = offsets_.data() + local_expert_num;

    auto indices = f2n_.slice(0, temp_.shape(0));
    auto offsets = offsets_.slice(0, local_expert_num + 1);

    if (block.w1w3) {
        Tensor inter;
        Tensor inter_scales;
        Tensor unused_in_scales;
        if (block.is_fused_silu && block.w1w3->output_dtype() == kFloat8_e4m3) {
            TM_SCOPE_CALL(
                linear_.Forward(p.input, unused_in_scales, *block.w1w3, indices, offsets, inter, inter_scales));
            TM_SCOPE_CALL(linear_.Forward(
                inter.slice({0, 0}, {-1, inter_size}), inter_scales, *block.w2, {}, offsets, temp_, unused_in_scales));
        }
        else {
            TM_SCOPE_CALL(linear_.Forward(p.input, *block.w1w3, indices, offsets, inter));

            if (!block.is_fused_silu) {
                TM_SCOPE_CALL(Activation(inter, block.w1w3->bias, f2E_, block.act_type, num_valid_tokens, st));
            }

            TM_SCOPE_CALL(linear_.Forward(inter.slice({0, 0}, {-1, inter_size}), *block.w2, {}, offsets, temp_));
        }
    }
    else {
        // Separate w1/w3 path
        Tensor gating;
        TM_SCOPE_CALL(linear_.Forward(p.input, *block.w1, indices, offsets, gating));

        Tensor up;
        TM_SCOPE_CALL(linear_.Forward(p.input, *block.w3, indices, offsets, up));

        TM_SCOPE_CALL(Activation(gating, up, block.act_type, num_valid_tokens, st));

        TM_SCOPE_CALL(linear_.Forward(gating, *block.w2, {}, offsets, temp_));
    }

    if (moe.shared_gate) {
        shared_scales_ = Gate(p.input, *moe.shared_gate);
    }
}

void MoeFfnDefaultImpl::Combine(MoeFfnLayer::ForwardParam& p)
{
    TM_FUNCTION_SCOPE();
    auto& moe = *p.weights;

    if (clear_pending_) {
        core::Context::stream().Wait(clear_done_event_);
        clear_pending_ = false;
    }

    TM_SCOPE_CALL(invokeMoeCombine(p.output,
                                   temp_,
                                   TM_CHECK_NOTNULL(moe.block())->w2->bias,
                                   scales_.data(),
                                   en2f_.data(),
                                   f2E_.data(),
                                   shared_scales_.data_or((float*)nullptr),
                                   moe.experts_per_token,
                                   1.f / tp_size_,
                                   p.scale,
                                   core::Context::stream().handle()));

    temp_          = {};
    shared_scales_ = {};
}

Tensor MoeFfnDefaultImpl::GetShardFfnInput(Tensor& global_hidden_states, const std::vector<int>&)
{
    TM_FUNCTION_SCOPE();
    if (ep_size_ == 1) {
        return global_hidden_states;
    }

    TM_CHECK(!clear_pending_);
    const int token_num         = global_hidden_states.shape(0);
    const int tokens_per_rank   = token_num / ep_size_;
    const int remainder         = token_num % ep_size_;
    const int local_token_num   = tokens_per_rank + (ep_rank_ < remainder);
    const int local_token_begin = ep_rank_ * tokens_per_rank + std::min(ep_rank_, remainder);
    const int local_token_end   = local_token_begin + local_token_num;

    if (local_token_begin > 0 || local_token_end < token_num) {
        auto& stream = core::Context::stream();

        clear_ready_event_.Record(stream);
        clear_stream_.Wait(clear_ready_event_);

        if (local_token_begin > 0) {
            Clear(global_hidden_states.slice(0, local_token_begin), clear_stream_);
        }
        if (local_token_end < token_num) {
            Clear(global_hidden_states.slice(local_token_end, token_num - local_token_end), clear_stream_);
        }

        clear_done_event_.Record(clear_stream_);
        clear_pending_ = true;
    }

    return global_hidden_states.slice(local_token_begin, local_token_num);
}

#ifdef USE_NCCL

class MoeFfnA2AImpl final: public MoeFfnLayerImpl {
public:
    MoeFfnA2AImpl(const EngineParam& engine, const Context& ctx, const MoeWeight& weights);

    void Forward(MoeFfnLayer::ForwardParam& p) override;

    void Combine(MoeFfnLayer::ForwardParam& p) override;

    Tensor GetShardFfnInput(Tensor& global_hidden_states, const std::vector<int>& local_token_nums) override;

private:
    void Init(const MoeWeight& weights);

    void CompileKernels(const MoeWeight& weights);

    MoeA2AInputPartition GetInputPartition(const Tensor&           global_hidden_states,
                                           const std::vector<int>& local_token_nums) const;

    const int ep_size_;
    const int ep_rank_;
    const int mlp_tp_size_;
    const int max_token_num_;
    const int max_token_per_rank_num_;
    int&      is_warm_up_;

    Buffer_<int>   f2n_;
    Buffer_<int>   f2E_;
    Buffer_<int>   en2f_;
    Buffer_<float> scales_;
    Buffer_<int>   offsets_;
    Buffer_<float> topk_weights_;
    Buffer_<int>   topk_indices_;

    Tensor               input_;
    Tensor               output_;
    Tensor               temp_;
    Tensor_<float>       shared_scales_;
    MoeA2AInputPartition partition_{};
    int                  max_token_num_per_rank_{};

    Stream scales_copy_stream_;
    Event  scales_copy_ready_event_;
    Event  scales_copy_done_event_;

    std::unique_ptr<comm::TokenDispatcher> dispatcher_;
};

MoeFfnA2AImpl::MoeFfnA2AImpl(const EngineParam& engine, const Context& ctx, const MoeWeight& weights):
    MoeFfnLayerImpl(ctx),
    ep_size_(engine.ep_size),
    ep_rank_(engine.ep_rank),
    mlp_tp_size_(engine.mlp_tp_size),
    max_token_num_(engine.max_forward_token_num * engine.attn_dp_size),
    max_token_per_rank_num_(engine.mlp_tp_size
                            * cdiv(engine.max_forward_token_num, engine.attn_tp_size * engine.attn_cp_size)),
    is_warm_up_(*ctx.is_warm_up)
{
    TM_CHECK(engine.data_type == kBfloat16)
        << "engine.data_type only support bfloat16 for now, got " << engine.data_type;
    dispatcher_ = std::make_unique<comm::TokenDispatcher>(ctx.comm.h_ep_group);

    scales_copy_stream_      = Stream::create();
    scales_copy_ready_event_ = Event::create();
    scales_copy_done_event_  = Event::create();

    Init(weights);
    CompileKernels(weights);
}

MoeA2AInputPartition MoeFfnA2AImpl::GetInputPartition(const Tensor&           global_hidden_states,
                                                      const std::vector<int>& local_token_nums) const
{
    return GetMoeA2AInputPartition(local_token_nums, ep_size_, ep_rank_, mlp_tp_size_);
}

void MoeFfnA2AImpl::Init(const MoeWeight& weights)
{
    const int experts_per_token = weights.experts_per_token;

    topk_weights_ = {max_token_per_rank_num_ * experts_per_token, kDEVICE};
    topk_indices_ = {max_token_per_rank_num_ * experts_per_token, kDEVICE};

    f2n_    = {experts_per_token * max_token_num_, kDEVICE};
    f2E_    = {experts_per_token * max_token_num_, kDEVICE};
    en2f_   = {experts_per_token * max_token_num_, kDEVICE};
    scales_ = {experts_per_token * max_token_num_, kDEVICE};

    dispatcher_->Init(max_token_per_rank_num_,
                      TM_CHECK_NOTNULL(weights.block())->hidden_dim,
                      experts_per_token,
                      weights.num_local_experts(),
                      false /* use_fp8_dispatch */);
}

void MoeFfnA2AImpl::CompileKernels(const MoeWeight& weights)
{
    TM_CUDA_CHECK(cudaMemsetAsync(
        topk_indices_.data(), -1, sizeof(int) * topk_indices_.size(), core::Context::stream().handle()));

    std::vector<int> tokens;
    for (int i = 1; i < max_token_per_rank_num_; i *= 2) {
        tokens.push_back(i);
    }
    tokens.push_back(max_token_per_rank_num_);

    for (const auto& token_num : tokens) {
        TM_LOG_INFO("Compiling a2a kernels for token_num = {}", token_num);
        Tensor input        = {{token_num, weights.block()->hidden_dim}, kBfloat16, kDEVICE};
        Tensor topk_indices = {topk_indices_, {token_num, weights.experts_per_token}};
        Tensor topk_weights = {topk_weights_, {token_num, weights.experts_per_token}};
        Tensor scales_t, out_moe;
        TM_SCOPE_CALL(dispatcher_->Dispatch(
            input, topk_indices, topk_weights, token_num, output_, scales_t, f2n_, f2E_, en2f_, offsets_));
        TM_SCOPE_CALL(dispatcher_->Combine(output_, out_moe));
    }
}

void MoeFfnA2AImpl::Forward(MoeFfnLayer::ForwardParam& p)
{
    TM_FUNCTION_SCOPE();

    partition_ = GetInputPartition(p.input, p.local_token_num);
    TM_CHECK_LE(partition_.max_tokens_per_rank, max_token_per_rank_num_);
    input_ = p.input.slice(partition_.begin, partition_.size);
    // slice token mask for current rank
    const bool* token_mask = TM_CHECK_NOTNULL(p.token_mask) + partition_.begin;
    // deepep jit treat the max_tokens_per_rank parameter as template parameter, round it to the nearest power of two to
    // reduce the number of functions that need to be compiled.
    max_token_num_per_rank_ = std::min(CeilPowerOfTwo(partition_.max_tokens_per_rank), max_token_per_rank_num_);

    const auto& moe               = *p.weights;
    const auto& block             = *TM_CHECK_NOTNULL(moe.block());
    const int   token_num         = partition_.size;
    const int   hidden_dim        = block.hidden_dim;
    const int   inter_size        = block.inter_size;
    const int   expert_num        = moe.num_experts();
    const int   local_expert_num  = moe.num_local_experts();
    const int   experts_per_token = moe.experts_per_token;
    const auto  st                = core::Context::stream().handle();

    Tensor_<float> logits = token_num ? Gate(input_, *moe.gate.get()) : Tensor_<float>{{0, expert_num}, kDEVICE};

    if (p.weights->topk_method == "noaux_tc") {
        TM_CHECK_EQ(moe.n_group, 1);
        TM_CHECK_EQ(moe.topk_group, 1);
        TM_CHECK(moe.scoring_func == "sigmoid" || moe.scoring_func == "softmax")
            << "unsupported noaux_tc scoring function: " << moe.scoring_func;

        const float* correction_bias = nullptr;
        if (moe.score_correction_bias && moe.score_correction_bias.size() > 0) {
            TM_CHECK_EQ(moe.score_correction_bias.size(), expert_num);
            correction_bias = moe.score_correction_bias.data<float>();
        }

        TM_SCOPE_CALL(invokeMoeGate_NoAuxTC(topk_weights_.data(),
                                            topk_indices_.data(),
                                            logits.data_or((float*)nullptr),
                                            token_mask,
                                            correction_bias,
                                            token_num,
                                            expert_num,
                                            experts_per_token,
                                            moe.norm_topk_prob,
                                            moe.routed_scale,
                                            moe.scoring_func == "sigmoid",
                                            st));
    }
    else {
        bool softmax = true;
        if (p.weights->topk_method == "group_limited_greedy") {
            if (token_num) {
                TM_SCOPE_CALL(invokeMoeSoftmaxMaskTopKGroups(
                    logits.data(), token_num, expert_num, expert_num / p.weights->n_group, p.weights->topk_group, st));
            }
            softmax = false;
        }

        TM_SCOPE_CALL(invokeMoeGate_V2(topk_weights_.data(),
                                       topk_indices_.data(),
                                       logits.data_or((float*)nullptr),
                                       token_mask,
                                       token_num,
                                       expert_num,
                                       experts_per_token,
                                       softmax,
                                       moe.norm_topk_prob,
                                       moe.routed_scale,
                                       st));
    }

    Tensor topk_indices = {topk_indices_, {token_num, experts_per_token}};
    Tensor topk_weights = {topk_weights_, {token_num, experts_per_token}};
    Tensor scales_t;
    TM_SCOPE_CALL(dispatcher_->Dispatch(
        input_, topk_indices, topk_weights, max_token_num_per_rank_, output_, scales_t, f2n_, f2E_, en2f_, offsets_));

    // transpose scales to [experts_per_token, token_num] for later combine
    Tensor scales_src = scales_t.t();
    Tensor scales_dst = {scales_, {experts_per_token, output_.shape(0)}};
    auto&  stream     = core::Context::stream();
    scales_copy_ready_event_.Record(stream);
    scales_copy_stream_.Wait(scales_copy_ready_event_);
    core::GenericCopy(scales_src, scales_dst, scales_copy_stream_.handle());
    scales_copy_done_event_.Record(scales_copy_stream_);

    temp_ = {{output_.shape(0) * p.weights->experts_per_token, hidden_dim}, output_.dtype(), output_.device()};
    const int* num_valid_tokens = offsets_.data() + local_expert_num;

    auto indices = f2n_.slice(0, temp_.shape(0));
    auto offsets = offsets_.slice(0, local_expert_num + 1);

    if (block.w1w3) {
        Tensor inter;
        Tensor inter_scales;
        Tensor unused_in_scales;
        if (block.is_fused_silu && block.w1w3->output_dtype() == kFloat8_e4m3) {
            TM_SCOPE_CALL(
                linear_.Forward(output_, unused_in_scales, *block.w1w3, indices, offsets, inter, inter_scales));
            TM_SCOPE_CALL(linear_.Forward(
                inter.slice({0, 0}, {-1, inter_size}), inter_scales, *block.w2, {}, offsets, temp_, unused_in_scales));
        }
        else {
            TM_SCOPE_CALL(linear_.Forward(output_, *block.w1w3, indices, offsets, inter));
            if (!block.is_fused_silu) {
                TM_SCOPE_CALL(Activation(inter, block.w1w3->bias, f2E_, block.act_type, num_valid_tokens, st));
            }
            TM_SCOPE_CALL(linear_.Forward(inter.slice({0, 0}, {-1, inter_size}), *block.w2, {}, offsets, temp_));
        }
    }
    else {
        // Separate w1/w3 path
        Tensor gating;
        TM_SCOPE_CALL(linear_.Forward(output_, *block.w1, indices, offsets, gating));

        Tensor up;
        TM_SCOPE_CALL(linear_.Forward(output_, *block.w3, indices, offsets, up));

        TM_SCOPE_CALL(Activation(gating, up, block.act_type, num_valid_tokens, st));

        TM_SCOPE_CALL(linear_.Forward(gating, *block.w2, {}, offsets, temp_));
    }

    if (moe.shared_gate && token_num > 0) {
        shared_scales_ = Gate(input_, *moe.shared_gate);
    }
}

void MoeFfnA2AImpl::Combine(MoeFfnLayer::ForwardParam& p)
{
    TM_FUNCTION_SCOPE();
    const auto& moe    = *p.weights;
    const auto& block  = *TM_CHECK_NOTNULL(moe.block());
    auto&       stream = core::Context::stream();

    stream.Wait(scales_copy_done_event_);
    TM_SCOPE_CALL(invokeMoeCombine(output_,
                                   temp_,
                                   block.w2->bias,
                                   scales_.data(),
                                   en2f_.data(),
                                   f2E_.data(),
                                   nullptr,
                                   moe.experts_per_token,
                                   1.f / mlp_tp_size_,
                                   0.f,
                                   stream.handle()));

    Tensor out_moe;
    TM_SCOPE_CALL(dispatcher_->Combine(output_, out_moe));

    TM_SCOPE_CALL(
        invokeMoeA2ASharedCombine(input_, out_moe, shared_scales_.data_or((float*)nullptr), p.scale, stream.handle()));

    input_         = {};
    output_        = {};
    temp_          = {};
    shared_scales_ = {};
    partition_     = {};
}

Tensor MoeFfnA2AImpl::GetShardFfnInput(Tensor& global_hidden_states, const std::vector<int>& local_token_nums)
{
    TM_FUNCTION_SCOPE();

    if (partition_.max_tokens_per_rank != 0) {
        return global_hidden_states.slice(partition_.begin, partition_.size);
    }

    const auto partition = GetInputPartition(global_hidden_states, local_token_nums);
    return global_hidden_states.slice(partition.begin, partition.size);
}

#endif

MoeFfnLayer::MoeFfnLayer(const EngineParam& engine, const Context& ctx, const MoeWeight* weights)
{
    const auto& moe = *TM_CHECK_NOTNULL(weights);

#ifdef BUILD_MULTI_GPU
    if (engine.ep_size > 1 && engine.moe_a2a_backend == "deepep") {
        impl_ = std::make_unique<MoeFfnA2AImpl>(engine, ctx, moe);
        return;
    }
#endif
    if (engine.ep_size == 1 || engine.moe_a2a_backend == "default") {
        impl_ = std::make_unique<MoeFfnDefaultImpl>(engine, ctx, moe);
        return;
    }

    TM_LOG_FATAL("Unsupported config for MoeFfnLayer");
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

Tensor MoeFfnLayer::GetShardFfnInput(Tensor& global_hidden_states, const std::vector<int>& local_token_nums)
{
    return impl_->GetShardFfnInput(global_hidden_states, local_token_nums);
}

}  // namespace turbomind
