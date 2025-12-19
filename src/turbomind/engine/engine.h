
#pragma once

#include <memory>

#include "src/turbomind/engine/gateway.h"

#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

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

    Engine(DataType      dtype,
           EngineParam   param,
           LanguageModel model,
           Context&      ctx,
           Gateway&      gateway,
           int           device_id,
           int           dp_rank,
           int           phases);

    void WarmUp();

    void Start();

    ScheduleMetrics GetScheduleMetrics();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind