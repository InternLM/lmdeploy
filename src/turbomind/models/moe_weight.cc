// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/moe_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

MoeWeight::MoeWeight(const core::MoeConfig& cfg)
{
    layer_id_         = cfg.layer_id;
    method_           = static_cast<MoeMethod>(cfg.method);
    experts_per_token = cfg.experts_per_token;
    norm_topk_prob    = cfg.norm_topk_prob;
    use_shared_gate   = cfg.shared_gate;
    routed_scale      = static_cast<float>(cfg.routed_scale);
    router_bias       = cfg.router_bias;
    topk_group        = cfg.topk_group;
    topk_method       = cfg.topk_method;
    n_group           = cfg.n_group;
    scoring_func      = cfg.scoring_func;
    router_n_groups   = cfg.router_n_groups;
    hidden_dim        = cfg.hidden_dim;
    inter_size        = cfg.inter_size;
    mlp_bias_         = cfg.mlp_bias;
    data_type_        = cfg.data_type;
    tp_size_          = cfg.tp_size;
    tp_rank_          = cfg.tp_rank;
    act_type_         = static_cast<ActivationType>(cfg.act_type);
    fuse_silu_act_    = cfg.fuse_silu;
    expert_num        = cfg.expert_num;
}

// Adapted from LinkExperts for LinearWeight
static void LinkLinearExperts(std::function<LinearWeight*(int)> experts, int n, LinearWeight& d)
{
    const auto& e0 = *experts(0);

    e0.copy_metadata_to(d);

    d.k_desc.num = d.q_desc.num = n;

    if (e0.bias) {
        d.bias = Tensor{{n, e0.output_dim}, e0.bias.dtype(), kDEVICE};
    }

    std::vector<std::pair<void*, int>> weights;
    std::vector<std::pair<void*, int>> scales;

    for (int i = 0; i < n; ++i) {
        auto& e = *experts(i);
        weights.emplace_back(e.weight.raw_data(), e.k_desc.ld);
        if (e.scales) {
            scales.emplace_back(e.scales.raw_data(), e.q_desc.ld);
        }
        if (e.bias) {
            Copy(e.bias, d.bias.slice(i, 1).squeeze(0));
        }
    }

    auto stream = core::Context::stream().handle();

    if (d.weight_format.dtype == kFloat8_e4m3 && d.input_dtype() == kFloat8_e4m3) {
        auto make_blocked_ptr = [&](const auto& ptrs) {
            return std::shared_ptr<void>{gemm::MakeBlockedPtrs(ptrs, stream), [](auto p) { cudaFree(p); }};
        };
        d.weight         = Tensor{make_blocked_ptr(weights), {n}, e0.weight.dtype(), kDEVICE};
        d.scales         = Tensor{make_blocked_ptr(scales), {n}, e0.scales.dtype(), kDEVICE};
        d.k_desc.offsets = d.q_desc.offsets = (int*)1;
    }
    else {
        auto make_strided_ptr = [&](const auto& ptrs) {
            return std::shared_ptr<void>{gemm::MakeStridedPtrs(ptrs, stream), [](auto p) { cudaFree(p); }};
        };
        d.weight = Tensor{make_strided_ptr(weights), {n}, d.weight_format.dtype, kDEVICE};
        if (e0.scales) {
            d.scales = Tensor{make_strided_ptr(scales), {n}, e0.scales.dtype(), kDEVICE};
        }
        d.k_desc.ld = d.q_desc.ld = 0;
    }
}

FfnWeight* MoeWeight::expert(int i) const
{
    if (!experts) {
        return nullptr;
    }
    return static_cast<FfnWeight*>(experts->child(std::to_string(i)));
}

void MoeWeight::prepare()
{
    // First prepare all children (experts, gate, etc.)
    Module::prepare();

    // Create batched block view for fused MoE path
    if (expert_num > 0 && method() == MoeMethod::kFused) {
        // Derive per-rank inter_size from first expert's weights.
        if (auto* e0 = expert(0)) {
            inter_size = e0->inter_size();
        }

        core::FfnConfig block_cfg;
        block_cfg.hidden_dim = hidden_dim;
        block_cfg.inter_size = inter_size;
        block_cfg.has_bias   = mlp_bias_;
        block_cfg.tp_size    = tp_size_;
        block_cfg.tp_rank    = tp_rank_;
        block_cfg.data_type  = data_type_;
        block_cfg.act_type   = static_cast<int>(act_type_);
        block_cfg.fuse_silu  = fuse_silu_act_;
        block_               = std::make_unique<FfnWeight>(block_cfg);

        // Link each linear in the block to the corresponding expert linears
        auto get_expert_w1w3 = [this](int i) -> LinearWeight* {
            auto* exp = expert(i);
            return exp ? exp->w1w3.get() : nullptr;
        };
        auto get_expert_w1 = [this](int i) -> LinearWeight* {
            auto* exp = expert(i);
            return exp ? exp->w1.get() : nullptr;
        };
        auto get_expert_w3 = [this](int i) -> LinearWeight* {
            auto* exp = expert(i);
            return exp ? exp->w3.get() : nullptr;
        };
        auto get_expert_w2 = [this](int i) -> LinearWeight* {
            auto* exp = expert(i);
            return exp ? exp->w2.get() : nullptr;
        };

        if (get_expert_w1w3(0)) {
            // Fused w1w3 path: experts have a single fused gate+up projection
            block_->add_child("w1w3", std::make_unique<LinearWeight>());
            LinkLinearExperts(get_expert_w1w3, expert_num, *block_->w1w3);
        }
        else {
            // Separate w1/w3 path: link individually
            block_->add_child("w1", std::make_unique<LinearWeight>());
            block_->add_child("w3", std::make_unique<LinearWeight>());
            if (get_expert_w1(0)) {
                LinkLinearExperts(get_expert_w1, expert_num, *block_->w1);
            }
            if (get_expert_w3(0)) {
                LinkLinearExperts(get_expert_w3, expert_num, *block_->w3);
            }
        }

        block_->add_child("w2", std::make_unique<LinearWeight>());
        if (get_expert_w2(0)) {
            LinkLinearExperts(get_expert_w2, expert_num, *block_->w2);
        }

        // Propagate the actual fused-silu state from the first expert to
        // the block.  Each expert's prepare() has already run above, so
        // is_fused_silu() now reflects whether the GEMM epilogue applies
        // SiLU (true for quantized formats, false for trivial bf16/fp16).
        if (auto* e0 = expert(0)) {
            block_->set_fused_silu(e0->is_fused_silu());
        }
    }
}

TM_MODULE_REGISTER(MoeWeight, core::MoeConfig);

TM_MODULE_METHODS(MoeWeight, MOE_WEIGHT_CHILDREN, MOE_WEIGHT_PARAMS)

}  // namespace turbomind
