// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/vision_model.h"

namespace turbomind {

class Qwen2VitWeight;

/// Concrete ``VisionModel`` for the Qwen2-VL / Qwen2.5-VL ViT encoder.
class Qwen2Vit: public VisionModel {
public:
    Qwen2Vit(const EngineParam& engine, const Context& ctx, const Qwen2VitWeight& weights, int phases);

    ~Qwen2Vit() override;

    void Run(BatchOp op, int phase, TensorMap& env) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
