
#pragma once

#include <memory>

#include "src/turbomind/engine/gateway.h"

#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

struct ScheduleMetrics;

class Engine {
public:
    ~Engine();

    Engine();
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    Engine(EngineParam        param,
           LanguageModel      model,
           const ModelWeight& weights,
           Context&           ctx,
           Gateway&           gateway,
           int                device_id,
           int                queue_id,
           int                phases);

    void Start();

    std::shared_ptr<ScheduleMetrics> GetScheduleMetrics();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
