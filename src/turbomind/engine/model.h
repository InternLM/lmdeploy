// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <memory>
#include <tuple>

#include "src/turbomind/engine/batch.h"
#include "src/turbomind/generation/generation.h"
#include "src/turbomind/models/batch_status.h"
#include "src/turbomind/models/input_processor.h"
#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/output_processor.h"
#include "src/turbomind/models/vision_model.h"

namespace turbomind {

class SpeculativeModel;

// The served model: the target, the optional vision and speculative models,
// and the input, output, generation, and status machinery around them — the
// whole that the engine thread's host operations and the executor's device
// steps drive. It owns its parts; the models are declared before the
// machinery so the machinery is destroyed first, and the speculative
// composition is resolved once from the speculator during construction.
// Constructed once at the
// engine, the composition root, and shared with the executor by reference.
// The generic batch-op fanout lives here: Run drives every module in the
// canonical order recorded once in OrderedComponents, skipping absent
// optional modules. Workflows that differ from the generic fanout — the
// prepare and forward steps — stay with the executor's device bracket.
class Model {
public:
    Model(std::unique_ptr<LanguageModel>    model,
          std::unique_ptr<VisionModel>      vision_model,
          std::unique_ptr<SpeculativeModel> spec_model,
          const EngineParam&                param,
          Context&                          context,
          int                               phases);

    // Defined in model.cc, where SpeculativeModel is complete, so this header
    // never pulls in a speculative header.
    ~Model();

    // Generic batch-op fanout: drives every module in the canonical order.
    void Run(BatchOp op, int phase, TensorMap& env);

    std::unique_ptr<LanguageModel>    target;
    std::unique_ptr<VisionModel>      vision;
    std::unique_ptr<SpeculativeModel> spec;

    BatchStatus     status;
    InputProcessor  input_processor;
    Generation      generation;
    OutputProcessor output_processor;

private:
    // Canonical module order for the generic fanout; absent optional modules
    // (vision, speculative) are skipped by the visitor.
    auto OrderedComponents()
    {
        return std::make_tuple(
            vision.get(), &status, &generation, &input_processor, target.get(), spec.get(), &output_processor);
    }
};

}  // namespace turbomind
