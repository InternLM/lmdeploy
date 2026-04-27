// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::core {

struct FfnConfig: ModuleConfig {
    FfnConfig(): ModuleConfig{"FfnWeight"} {}

#define FFN_FIELDS(X)                                                                                                  \
    X(int, hidden_dim)                                                                                                 \
    X(int, inter_size)                                                                                                 \
    X(bool, has_bias)                                                                                                  \
    X(int, tp_size)                                                                                                    \
    X(int, tp_rank)                                                                                                    \
    X(DataType, data_type)                                                                                             \
    X(int, act_type)                                                                                                   \
    X(bool, fuse_silu)                                                                                                 \
    X(bool, fused_moe)

    FFN_FIELDS(TM_MEMBER)
    TM_FOR_EACH(FfnConfig, FFN_FIELDS)

#undef FFN_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class FfnWeight: public core::Module {
public:
    const char* type() const override
    {
        return "FfnWeight";
    }

    FfnWeight() = default;

    FfnWeight(const core::FfnConfig& cfg);

    void prepare() override;

    // --- X-macro child members ---
#define FFN_WEIGHT_CHILDREN(X)                                                                                         \
    X(LinearWeight, w1)                                                                                                \
    X(LinearWeight, w3)                                                                                                \
    X(LinearWeight, w2)                                                                                                \
    X(LinearWeight, w1w3)

#define FFN_WEIGHT_PARAMS(X)

    TM_MODULE_DECLARE(FfnWeight, FFN_WEIGHT_CHILDREN, FFN_WEIGHT_PARAMS)

    int inter_size() const
    {
        return inter_size_;
    }
    ActivationType act_type() const
    {
        return act_type_;
    }
    bool is_fused_silu() const
    {
        return is_fused_silu_;
    }

    /// Override is_fused_silu_ (used by MoE block view after linking experts).
    void set_fused_silu(bool val)
    {
        is_fused_silu_ = val;
    }

private:
    int            hidden_dim_{};
    int            inter_size_{};
    bool           bias_{};
    int            tp_size_{};
    int            tp_rank_{};
    DataType       data_type_{};
    ActivationType act_type_{};
    bool           is_fused_silu_{};
    bool           is_fused_moe_{};
};

}  // namespace turbomind
