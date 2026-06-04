// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/vision_model.h"

namespace turbomind {

class Qwen3_5VitWeight;

/// Concrete ``VisionModel`` for the Qwen3.5 ViT encoder.
///
/// This task only stubs the runtime: the phase methods log a debug
/// breadcrumb and return. Follow-up work fills in the actual ViT
/// kernels (patcher → blocks → merger → caching of image embeddings).
class Qwen3_5Vit: public VisionModel {
public:
    Qwen3_5Vit(const EngineParam& engine, const Context& ctx, const Qwen3_5VitWeight& weights, int phases);

    ~Qwen3_5Vit() override;

    void Run(BatchOp op, int phase, TensorMap& env) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
