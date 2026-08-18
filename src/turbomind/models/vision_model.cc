// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/vision_model.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/models/internvit/internvit.h"
#include "src/turbomind/models/internvit/internvit_weight.h"
#include "src/turbomind/models/qwenvit/qwenvit.h"
#include "src/turbomind/models/qwenvit/qwenvit_weight.h"

#include <string_view>

namespace turbomind {

std::unique_ptr<VisionModel> CreateVisionModel(const VisionModelWeight& weights,  //
                                               const EngineParam&       engine,
                                               const Context&           ctx,
                                               int                      phases)
{
    if (std::string_view{weights.type()} == "QwenVitWeight") {
        return std::make_unique<QwenVit>(engine, ctx, static_cast<const QwenVitWeight&>(weights), phases);
    }
    if (std::string_view{weights.type()} == "InternVitWeight") {
        return std::make_unique<InternVit>(engine, ctx, static_cast<const InternVitWeight&>(weights), phases);
    }

    TM_LOG_FATAL("Unsupported vision model weight type: {}", weights.type());
    return nullptr;
}

}  // namespace turbomind
