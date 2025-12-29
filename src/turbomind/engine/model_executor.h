// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>

#include "src/turbomind/core/core.h"

#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/queue.h"
#include "src/turbomind/models/language_model.h"

#include "src/turbomind/models/llama/context.h"

namespace turbomind {

// Model executor for auto-regressive language models
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

    ModelExecutor(LanguageModel&                     model,
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
