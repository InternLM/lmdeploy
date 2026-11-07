// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/vision_model_weight.h"

#include <array>
#include <memory>
#include <utility>
#include <vector>

namespace turbomind {

/// Polymorphic peer of ``LanguageModel`` for the vision sub-graph.
///
/// Concrete subclasses (one per VLM family — ``QwenVit``,
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

    /// Phase entry point. Called from the batch-operation fanouts (the
    /// engine's host-op functions and the executor's device steps) *before*
    /// the language model. Subclasses dispatch on ``op``.
    virtual void Run(BatchOp op, int phase, TensorMap& env) = 0;
};

struct MultiModalData {
    Tensor             data;  // pixel values
    Interval           interval;
    std::array<int, 3> grid_thw;  // qwen3
};

struct EmbeddingPatch {
    int row_count;
    int source_row;
    int destination_row;
};

struct MultiModalEmbeddingData {
    Tensor                      data;
    std::vector<EmbeddingPatch> target_patches;
    std::vector<EmbeddingPatch> successor_patches;

    MultiModalEmbeddingData() = default;

    explicit MultiModalEmbeddingData(Tensor                      data,
                                     std::vector<EmbeddingPatch> target_patches,
                                     std::vector<EmbeddingPatch> successor_patches):
        data{std::move(data)},
        target_patches{std::move(target_patches)},
        successor_patches{std::move(successor_patches)}
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

std::unique_ptr<VisionModel> CreateVisionModel(const VisionModelWeight& weights,  //
                                               const EngineParam&       engine,
                                               const Context&           ctx,
                                               int                      phases,
                                               bool                     successor_embeddings);

}  // namespace turbomind
