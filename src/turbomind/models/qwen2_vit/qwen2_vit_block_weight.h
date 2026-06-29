// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/attention_weight.h"

namespace turbomind::core {

/// Per-block config for the Qwen2/Qwen2.5 vision transformer.
struct Qwen2VitBlockConfig: ModuleConfig {
    Qwen2VitBlockConfig(): ModuleConfig{"Qwen2VitBlockWeight"} {}

#define QWEN2VIT_BLOCK_FIELDS(X)                                                                                       \
    X(DataType, data_type)                                                                                             \
    X(int, hidden_dim)                                                                                                 \
    X(int, head_num)                                                                                                   \
    X(int, intermediate_size)                                                                                          \
    X(float, norm_eps, 1e-6f)

    QWEN2VIT_BLOCK_FIELDS(TM_MEMBER)
    TM_FOR_EACH(Qwen2VitBlockConfig, QWEN2VIT_BLOCK_FIELDS)

#undef QWEN2VIT_BLOCK_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class LinearWeight;

/// One transformer block of the Qwen2/Qwen2.5 ViT.
///
/// Children:
///   - norm1, norm2      LayerNorm or RMSNorm
///   - attention         AttentionWeight (packed Q/K/V + output projection)
///   - mlp_fc1, mlp_fc2  Linear (in: hidden ↔ intermediate)
///   - mlp_gate          Optional Qwen2.5 gated SiLU gate projection
class Qwen2VitBlockWeight: public core::Module {
public:
    const char* type() const override
    {
        return "Qwen2VitBlockWeight";
    }

    Qwen2VitBlockWeight() = default;
    explicit Qwen2VitBlockWeight(const core::Qwen2VitBlockConfig& cfg):
        data_type{cfg.data_type},
        hidden_dim{cfg.hidden_dim},
        head_num{cfg.head_num},
        intermediate_size{cfg.intermediate_size},
        norm_eps{cfg.norm_eps}
    {
    }

#define QWEN2VIT_BLOCK_CHILDREN(X)                                                                                     \
    X(core::Module, norm1)                                                                                             \
    X(core::Module, norm2)                                                                                             \
    X(AttentionWeight, attention)                                                                                      \
    X(LinearWeight, mlp_gate)                                                                                          \
    X(LinearWeight, mlp_fc1)                                                                                           \
    X(LinearWeight, mlp_fc2)

#define QWEN2VIT_BLOCK_PARAMS(X)

    TM_MODULE_DECLARE(Qwen2VitBlockWeight, QWEN2VIT_BLOCK_CHILDREN, QWEN2VIT_BLOCK_PARAMS)

    // --- Public scalars ---
    DataType data_type{};
    int      hidden_dim{};
    int      head_num{};
    int      intermediate_size{};
    float    norm_eps{};
};

}  // namespace turbomind
