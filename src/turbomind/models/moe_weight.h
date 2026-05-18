// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/ffn_weight.h"

namespace turbomind {

}  // namespace turbomind

namespace turbomind::core {

struct MoeConfig: ModuleConfig {
    MoeConfig(): ModuleConfig{"MoeWeight"} {}

#define MOE_FIELDS(X)                                                                                                  \
    X(int, expert_num)                                                                                                 \
    X(int, experts_per_token)                                                                                          \
    X(int, act_type)                                                                                                   \
    X(bool, fuse_silu)                                                                                                 \
    X(bool, norm_topk_prob)                                                                                            \
    X(std::string, topk_method)                                                                                        \
    X(std::string, scoring_func)                                                                                       \
    X(int, topk_group)                                                                                                 \
    X(int, n_group)                                                                                                    \
    X(int, router_n_groups)                                                                                            \
    X(double, routed_scale)                                                                                            \
    X(int, ep_size)                                                                                                    \
    X(int, ep_rank)                                                                                                    \
    X(int, ll_max_tokens_per_rank)                                                                                     \
    X(DataType, data_type)

    MOE_FIELDS(TM_MEMBER)
    TM_FOR_EACH(MoeConfig, MOE_FIELDS)

#undef MOE_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class MoeWeight: public core::Module {
public:
    const char* type() const override
    {
        return "MoeWeight";
    }

    MoeWeight() = default;

    MoeWeight(const core::MoeConfig& cfg);

    void prepare() override;
    int  num_experts() const
    {
        return expert_num;
    }
    // Number of experts physically resident on this EP rank.
    int local_num_experts() const
    {
        return ep_size > 1 ? expert_num / ep_size : expert_num;
    }
    // Global index of this rank's first local expert.
    int local_expert_offset() const
    {
        return ep_size > 1 ? ep_rank * (expert_num / ep_size) : 0;
    }

    // --- X-macro child members ---
#define MOE_WEIGHT_CHILDREN(X)                                                                                         \
    X(LinearWeight, gate)                                                                                              \
    X(LinearWeight, shared_gate)                                                                                       \
    X(core::ModuleList, experts)

#define MOE_WEIGHT_PARAMS(X) X(score_correction_bias)

    TM_MODULE_DECLARE(MoeWeight, MOE_WEIGHT_CHILDREN, MOE_WEIGHT_PARAMS)

    // --- Typed accessors ---
    FfnWeight* expert(int i) const;
    FfnWeight* block() const
    {
        return block_.get();
    }

    // --- Config fields (public for runtime access) ---
    int         expert_num{};
    int         experts_per_token{};
    bool        norm_topk_prob{};
    float       routed_scale{};
    int         topk_group{};
    std::string topk_method;
    int         n_group{};
    std::string scoring_func;
    int         router_n_groups{};
    int         ep_size{1};
    int         ep_rank{0};
    int         ll_max_tokens_per_rank{};

private:
    ActivationType act_type_{};
    bool           fuse_silu_act_{};

    DataType data_type_{};

    std::unique_ptr<FfnWeight> block_;
};

}  // namespace turbomind
