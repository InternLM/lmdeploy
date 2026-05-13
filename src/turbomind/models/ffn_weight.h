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
    X(int, act_type)                                                                                                   \
    X(bool, fuse_silu)                                                                                                 \
    X(bool, is_expert)                                                                                                 \
    X(DataType, data_type)                                                                                             \
    X(int, tp_size)                                                                                                    \
    X(int, tp_rank)

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

    int            hidden_dim{};
    int            inter_size{};
    ActivationType act_type{};
    bool           is_fused_silu{};
    int            tp_size{};
    int            tp_rank{};

private:
    bool     is_expert_{};
    DataType data_type_{};
};

}  // namespace turbomind
