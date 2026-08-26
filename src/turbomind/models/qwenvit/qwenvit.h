// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/vision_model.h"

namespace turbomind {

class QwenVitWeight;

/// Unified ``VisionModel`` for the Qwen ViT family: Qwen2-VL / Qwen2.5-VL / Qwen3.5.
///
/// A single config-driven implementation. The orthogonal feature toggles that
/// distinguish the families are selected from ``QwenVitConfig``:
///   - window attention      (Qwen2.5):  use_window_attention
///   - learned pos embedding  (Qwen3.5):  num_position_embeddings > 0
///   - gated SiLU MLP         (Qwen2.5):  gated_mlp
///   - tanh-approx GELU MLP   (Qwen3.5):  gelu_tanh
///   - RMSNorm vs LayerNorm   (Qwen2):    norm_type
class QwenVit: public VisionModel {
public:
    QwenVit(const EngineParam& engine, const Context& ctx, const QwenVitWeight& weights, int phases);

    ~QwenVit() override;

    void Run(BatchOp op, int phase, TensorMap& env) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
