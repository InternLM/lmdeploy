// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/attention_weight.h"

namespace turbomind::core {

struct InternVitBlockConfig: ModuleConfig {
    InternVitBlockConfig(): ModuleConfig{"InternVitBlockWeight"} {}

#define INTERNVIT_BLOCK_FIELDS(X)                                                                                      \
    X(DataType, data_type)                                                                                             \
    X(int, hidden_dim)                                                                                                 \
    X(int, head_num)                                                                                                   \
    X(int, intermediate_size)                                                                                          \
    X(float, norm_eps, 1e-6f)

    INTERNVIT_BLOCK_FIELDS(TM_MEMBER)
    TM_FOR_EACH(InternVitBlockConfig, INTERNVIT_BLOCK_FIELDS)

#undef INTERNVIT_BLOCK_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class LinearWeight;

class InternVitBlockWeight: public core::Module {
public:
    const char* type() const override
    {
        return "InternVitBlockWeight";
    }

    InternVitBlockWeight() = default;
    explicit InternVitBlockWeight(const core::InternVitBlockConfig& cfg):
        data_type{cfg.data_type},
        hidden_dim{cfg.hidden_dim},
        head_num{cfg.head_num},
        intermediate_size{cfg.intermediate_size},
        norm_eps{cfg.norm_eps}
    {
    }

    void prepare() override;

#define INTERNVIT_BLOCK_CHILDREN(X)                                                                                    \
    X(core::Module, norm1)                                                                                             \
    X(core::Module, norm2)                                                                                             \
    X(AttentionWeight, attention)                                                                                      \
    X(LinearWeight, mlp_fc1)                                                                                           \
    X(LinearWeight, mlp_fc2)

#define INTERNVIT_BLOCK_PARAMS(X)                                                                                      \
    X(lambda_1)                                                                                                        \
    X(lambda_2)

    TM_MODULE_DECLARE(InternVitBlockWeight, INTERNVIT_BLOCK_CHILDREN, INTERNVIT_BLOCK_PARAMS)

    DataType data_type{};
    int      hidden_dim{};
    int      head_num{};
    int      intermediate_size{};
    float    norm_eps{};
};

}  // namespace turbomind
