// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"

#include <array>
#include <memory>
#include <utility>
#include <vector>

namespace turbomind {

/// Polymorphic peer of ``LanguageModel`` for the vision sub-graph.
///
/// Concrete subclasses (one per VLM family — ``Qwen3_5Vit``,
/// ``InternVit``, …) wire up the per-family C++ runtime. The
/// engine talks to this base via ``Run(BatchOp, phase, env)``,
/// mirroring ``LanguageModel::Run``.
///
/// Lifetime: owned by ``Engine`` as a ``unique_ptr<VisionModel>`` and
/// non-null only when the corresponding ``ModelRoot::vision_model``
/// child was attached during weight loading.
class VisionModel {
public:
    virtual ~VisionModel() = default;

    /// Phase entry point. Called from ``ModelExecutor::Run`` *before*
    /// the language model. Subclasses dispatch on ``op``.
    virtual void Run(BatchOp op, int phase, TensorMap& env) = 0;
};

struct MultiModalData {
    Tensor             data;  // pixel values
    Interval           interval;
    std::array<int, 3> grid_thw;  // qwen3
};

struct MultiModalEmbeddingData {
    Tensor                           data;
    std::vector<std::pair<int, int>> image_embeds_coords;
    std::vector<std::pair<int, int>> input_embeds_coords;

    MultiModalEmbeddingData() = default;

    explicit MultiModalEmbeddingData(Tensor                           data,
                                     std::vector<std::pair<int, int>> image_embeds_coords,
                                     std::vector<std::pair<int, int>> input_embeds_coords):
        data{std::move(data)},
        image_embeds_coords{std::move(image_embeds_coords)},
        input_embeds_coords{std::move(input_embeds_coords)}
    {
    }

    Buffer_<MultiModalEmbeddingData*> buf() const&
    {
        return MakeBuffer(std::make_shared<MultiModalEmbeddingData>(*this));
    }

    Buffer_<MultiModalEmbeddingData*> buf() &&
    {
        return MakeBuffer(std::make_shared<MultiModalEmbeddingData>(std::move(*this)));
    }

private:
    static Buffer_<MultiModalEmbeddingData*> MakeBuffer(std::shared_ptr<MultiModalEmbeddingData> payload)
    {
        auto* raw_ptr = payload.get();
        auto  slot    = std::shared_ptr<MultiModalEmbeddingData*>{
            new MultiModalEmbeddingData*(raw_ptr),
            [payload = std::move(payload)](MultiModalEmbeddingData** p) { delete p; }};

        return {std::static_pointer_cast<void>(slot), 1, kCPU};
    }
};

}  // namespace turbomind
