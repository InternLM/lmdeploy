// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/vision_model.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/models/qwen3_5vit/qwen3_5vit.h"
#include "src/turbomind/models/qwen3_5vit/qwen3_5vit_weight.h"

#include <string_view>

namespace turbomind {

std::unique_ptr<VisionModel> CreateVisionModel(const VisionModelWeight& weights,  //
                                               const EngineParam&       engine,
                                               const Context&           ctx,
                                               int                      phases)
{
    if (std::string_view{weights.type()} == "Qwen3_5VitWeight") {
        return std::make_unique<Qwen3_5Vit>(engine, ctx, static_cast<const Qwen3_5VitWeight&>(weights), phases);
    }

    TM_LOG_FATAL("Unsupported vision model weight type: {}", weights.type());
    return nullptr;
}

}  // namespace turbomind
