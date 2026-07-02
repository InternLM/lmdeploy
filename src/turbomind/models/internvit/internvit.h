// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/vision_model.h"

#include <memory>

namespace turbomind {

class InternVitWeight;

class InternVit: public VisionModel {
public:
    InternVit(const EngineParam& engine, const Context& ctx, const InternVitWeight& weights, int phases);

    ~InternVit() override;

    void Run(BatchOp op, int phase, TensorMap& env) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
