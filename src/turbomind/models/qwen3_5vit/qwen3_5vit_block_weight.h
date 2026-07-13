// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/attention_weight.h"

namespace turbomind::core {

/// Per-block config for the Qwen3.5 vision transformer.
///
/// Carries the dimensions the runtime needs (hidden_dim, head_num,
/// intermediate_size). Loading itself is structural — children are
/// committed by the Python builder via ``add_child_raw``.
struct Qwen3_5VitBlockConfig: ModuleConfig {
    Qwen3_5VitBlockConfig(): ModuleConfig{"Qwen3_5VitBlockWeight"} {}

#define QWEN3_5VIT_BLOCK_FIELDS(X)                                                                                     \
    X(DataType, data_type)                                                                                             \
    X(int, hidden_dim)                                                                                                 \
    X(int, head_num)                                                                                                   \
    X(int, intermediate_size)                                                                                          \
    X(float, norm_eps, 1e-6f)

    QWEN3_5VIT_BLOCK_FIELDS(TM_MEMBER)
    TM_FOR_EACH(Qwen3_5VitBlockConfig, QWEN3_5VIT_BLOCK_FIELDS)

#undef QWEN3_5VIT_BLOCK_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class LayerNormWeight;
class LinearWeight;

/// One transformer block of the Qwen3.5 ViT.
///
/// Children:
///   - norm1, norm2      LayerNorm (weight + bias)
///   - attention         AttentionWeight (packed Q/K/V + output projection)
///   - mlp_fc1, mlp_fc2  Linear (in: hidden ↔ intermediate)
class Qwen3_5VitBlockWeight: public core::Module {
public:
    const char* type() const override
    {
        return "Qwen3_5VitBlockWeight";
    }

    Qwen3_5VitBlockWeight() = default;
    explicit Qwen3_5VitBlockWeight(const core::Qwen3_5VitBlockConfig& cfg):
        data_type{cfg.data_type},
        hidden_dim{cfg.hidden_dim},
        head_num{cfg.head_num},
        intermediate_size{cfg.intermediate_size},
        norm_eps{cfg.norm_eps}
    {
    }

#define QWEN3_5VIT_BLOCK_CHILDREN(X)                                                                                   \
    X(LayerNormWeight, norm1)                                                                                          \
    X(LayerNormWeight, norm2)                                                                                          \
    X(AttentionWeight, attention)                                                                                      \
    X(LinearWeight, mlp_fc1)                                                                                           \
    X(LinearWeight, mlp_fc2)

#define QWEN3_5VIT_BLOCK_PARAMS(X)

    TM_MODULE_DECLARE(Qwen3_5VitBlockWeight, QWEN3_5VIT_BLOCK_CHILDREN, QWEN3_5VIT_BLOCK_PARAMS)

    // --- Public scalars ---
    DataType data_type{};
    int      hidden_dim{};
    int      head_num{};
    int      intermediate_size{};
    float    norm_eps{};
};

}  // namespace turbomind
