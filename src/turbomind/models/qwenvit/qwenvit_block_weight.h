// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/attention_weight.h"

namespace turbomind::core {

/// Per-block config for the Qwen2 / Qwen2.5 / Qwen3.5 vision transformer.
struct QwenVitBlockConfig: ModuleConfig {
    QwenVitBlockConfig(): ModuleConfig{"QwenVitBlockWeight"} {}

#define QWENVIT_BLOCK_FIELDS(X)                                                                                        \
    X(DataType, data_type)                                                                                             \
    X(int, hidden_dim)                                                                                                 \
    X(int, head_num)                                                                                                   \
    X(int, intermediate_size)                                                                                          \
    X(float, norm_eps, 1e-6f)

    QWENVIT_BLOCK_FIELDS(TM_MEMBER)
    TM_FOR_EACH(QwenVitBlockConfig, QWENVIT_BLOCK_FIELDS)

#undef QWENVIT_BLOCK_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class LinearWeight;

/// One transformer block of the Qwen ViT (covers Qwen2 / Qwen2.5 / Qwen3.5).
///
/// Children:
///   - norm1, norm2      LayerNorm or RMSNorm (held as core::Module)
///   - attention         AttentionWeight (packed Q/K/V + output projection)
///   - mlp_gate          Optional gated SiLU gate projection (Qwen2.5 only)
///   - mlp_fc1, mlp_fc2  Linear (in: hidden ↔ intermediate)
class QwenVitBlockWeight: public core::Module {
public:
    const char* type() const override
    {
        return "QwenVitBlockWeight";
    }

    QwenVitBlockWeight() = default;
    explicit QwenVitBlockWeight(const core::QwenVitBlockConfig& cfg):
        data_type{cfg.data_type},
        hidden_dim{cfg.hidden_dim},
        head_num{cfg.head_num},
        intermediate_size{cfg.intermediate_size},
        norm_eps{cfg.norm_eps}
    {
    }

#define QWENVIT_BLOCK_CHILDREN(X)                                                                                      \
    X(core::Module, norm1)                                                                                             \
    X(core::Module, norm2)                                                                                             \
    X(AttentionWeight, attention)                                                                                      \
    X(LinearWeight, mlp_gate)                                                                                          \
    X(LinearWeight, mlp_fc1)                                                                                           \
    X(LinearWeight, mlp_fc2)

#define QWENVIT_BLOCK_PARAMS(X)

    TM_MODULE_DECLARE(QwenVitBlockWeight, QWENVIT_BLOCK_CHILDREN, QWENVIT_BLOCK_PARAMS)

    // --- Public scalars ---
    DataType data_type{};
    int      hidden_dim{};
    int      head_num{};
    int      intermediate_size{};
    float    norm_eps{};
};

}  // namespace turbomind
