#pragma once

#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class LlamaWeight;

class LanguageModel {
public:
    ~LanguageModel();

    LanguageModel() = default;

    LanguageModel(LanguageModel&&) noexcept;

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    LanguageModel(DataType              dtype,
                  const ModelParam&     model,
                  const EngineParam&    engine,
                  const AttentionParam& attn,
                  const MoeParam&       moe,
                  const Context&        ctx,
                  const LlamaWeight&    weights,
                  int                   phases);

    void Run(BatchOp op, int phase, TensorMap& env);

    const ModelParam&     model_param() const noexcept;
    const AttentionParam& attn_param() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
