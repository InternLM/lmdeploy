// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <memory>

#include "src/turbomind/core/core.h"

#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/queue.h"

#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class Model;

// Model executor for auto-regressive language models, optionally
// preceded by a per-batch ViT pass for VLM checkpoints.
class ModelExecutor {
public:
    ~ModelExecutor();

    ModelExecutor();
    ModelExecutor(ModelExecutor&&) noexcept;
    ModelExecutor& operator=(ModelExecutor&&) noexcept;

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    ModelExecutor(Model&                             model,
                  const EngineParam&                 param,
                  Context&                           context,
                  int                                device_id,
                  Queue<std::unique_ptr<BatchData>>& inbound,
                  Queue<std::unique_ptr<BatchData>>& outbound);

    void Start();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
