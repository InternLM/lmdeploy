#pragma once

#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class ModelWeight;

class LanguageModel {
public:
    ~LanguageModel();

    LanguageModel() = default;

    LanguageModel(LanguageModel&&) noexcept;

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    LanguageModel(const EngineParam& engine, const Context& ctx, const ModelWeight& weights, int phases);

    void Run(BatchOp op, int phase, TensorMap& env);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
