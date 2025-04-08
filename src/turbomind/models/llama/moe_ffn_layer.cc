// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/core/typecvt.h"
#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/monotonic.h"
#include "src/turbomind/utils/nvtx_utils.h"
#include "src/turbomind/utils/string_utils.h"
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <iomanip>

namespace turbomind {

void MoeFfnLayer::AllocateBuffer(size_t tokens, size_t padded, size_t expert_num, size_t inter_buf_factor)
{
    char* base = 0;

    auto allocate = [&](void* base) {
        Monotonic alloc{base};
        alloc(&masks_, expert_num * padded);
        alloc(&f2n_, param_.experts_per_token * tokens);
        alloc(&en2f_, param_.experts_per_token * tokens);
        alloc(&scales_, param_.experts_per_token * tokens);
        alloc(&shared_scales_, tokens);
        return (char*)alloc.ptr() - (char*)base;
    };

    const auto workspace_size = allocate(0);

    workspace_ = (char*)allocator_->reMalloc(workspace_, workspace_size);

    allocate(workspace_);
}

void MoeFfnLayer::FreeBuffer()
{
    allocator_->free((void**)&workspace_);

    allocator_->free((void**)&accum_);
    allocator_->free((void**)&offsets_);

    allocator_->free((void**)&h_offsets_, true);
}

core::Tensor_<float> MoeFfnLayer::Gate(const core::Tensor& input, const LlamaDenseWeight& gate)
{
    auto& weight = gate.weight;
    TM_CHECK_EQ(input.shape(1), weight.shape(0));
    const float          alpha = 1.f;
    const float          beta  = 0.f;
    core::Tensor_<float> logits{{input.shape(0), weight.shape(1)}, MEMORY_GPU};
    linear_->forward(input, gate, LlamaLinear::kGemm, logits);
    sync_check_cuda_error();
    return logits;
}

void MoeFfnLayer::Forward(ForwardParam& p)
{
    const int   tokens = p.input.shape(0);
    const auto& moe    = *p.weight;

    const size_t padded     = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;
    const int    expert_num = moe.experts.size();

    FT_CHECK(expert_num);

    const size_t inter_buf_factor = [&] {
        if (param_.method == MoeParam::kNaive) {
            return 0;  // managed by ffn
        }
        else if (moe.block.is_fused_silu) {
            return 1;
        }
        else {
            return 2;
        }
    }();

    AllocateBuffer(tokens, padded, expert_num, inter_buf_factor);

    auto logits = Gate(p.input, moe.gate);

    // if (tensor_para_.rank_ == 0) {
    //     Compare(logits_, tokens * expert_num, Concat("logit", layer_id), compare_mode, stream_);
    // }

    check_cuda_error(cudaMemsetAsync(accum_, 0, sizeof(int) * expert_num * kMoeGateMaxTiles, stream_));
    check_cuda_error(cudaMemsetAsync(masks_, -1, sizeof(int8_t) * expert_num * padded, stream_));

    // dump_logits(tokens, layer_id);

    bool softmax = true;
    if (param_.topk_method == "group_limited_greedy") {
        invokeMoeSoftmaxMaskTopKGroups(
            logits.data(), tokens, expert_num, expert_num / param_.n_group, param_.topk_group, stream_);
        sync_check_cuda_error();
        softmax = false;
    }

    /// TODO: fix illegal memory access even if NaN are present in logits
    invokeMoeGate_V2(f2n_,
                     en2f_,
                     offsets_,
                     scales_,
                     masks_,
                     accum_,
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
        check_cuda_error(
            cudaMemcpyAsync(offsets_, h_offsets_, sizeof(int) * (expert_num + 1), cudaMemcpyDefault, stream_));
    }

    p.temp = core::Tensor{{tokens * param_.experts_per_token, hidden_dim_}, p.input.dtype(), p.input.device()};

    if (param_.method == MoeParam::kNaive) {

        invokeMoeDispatch(p.temp, p.input, f2n_, param_.experts_per_token, stream_);
        sync_check_cuda_error();

        check_cuda_error(
            cudaMemcpyAsync(h_offsets_, offsets_, sizeof(int) * (expert_num + 1), cudaMemcpyDefault, stream_));

        check_cuda_error(cudaStreamSynchronize(stream_));

        if (h_offsets_[expert_num] != tokens * param_.experts_per_token) {
            FT_CHECK_WITH_INFO(0, fmtstr("%d vs %d", h_offsets_[expert_num], tokens * param_.experts_per_token));
        }

        for (int i = 0; i < expert_num; ++i) {
            FT_CHECK(moe.experts[i]->is_fused_silu == false);
            if (int count = h_offsets_[i + 1] - h_offsets_[i]) {
                auto io = p.temp.slice({h_offsets_[i], 0}, {count, -1});
                expert_ffn_->forward({io, io, moe.experts.at(i).get(), p.layer_id});
            }
        }
    }
    else {
        context_->update(expert_num, param_.experts_per_token, offsets_);

        auto& block = moe.block;

        const int    inter_dim = block.is_fused_silu ? inter_dim : inter_dim * 2;
        core::Tensor inter{{tokens * param_.experts_per_token, inter_dim}, p.input.dtype(), p.input.device()};

        linear_->forward_moe(inter,
                             p.input,
                             f2n_,
                             offsets_,
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

        linear_->forward_moe(p.temp,
                             inter.slice({0, 0}, {-1, inter_size_}),
                             nullptr,
                             offsets_,
                             block.output,
                             LlamaLinear::kGemm,
                             context_.get());
        sync_check_cuda_error();
        auto mode1 = kCmpRead;
    }
}

void MoeFfnLayer::Combine(ForwardParam& p)
{
    auto& moe = *p.weight;

    core::Tensor_<float> shared_scales;

    if (moe.shared_gate.weight) {
        shared_scales = Gate(p.input, moe.shared_gate);
    }

    invokeMoeCombine(p.output,
                     p.temp,
                     scales_,
                     en2f_,
                     shared_scales ? shared_scales.data() : nullptr,
                     param_.experts_per_token,
                     p.scale,
                     stream_);
    sync_check_cuda_error();
}

}  // namespace turbomind
