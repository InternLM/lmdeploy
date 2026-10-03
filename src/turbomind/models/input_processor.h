#pragma once

#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class InputProcessor {
public:
    ~InputProcessor();

    InputProcessor(const EngineParam& engine,
                   int                hidden_units,
                   DataType           data_type,
                   int                phases,
                   bool               speculative,
                   int                max_verification_positions,
                   bool               successor_embeddings);

    void Run(BatchOp op, int phase, TensorMap& env);

    // Composed mode only: the forward-time target-pass staging step. Gathers
    // the target's input ids from the request token rows and stages the
    // per-request key lengths for the executor's offsets prefix-sum.
    void BuildTargetInputs(int phase, TensorMap& env);

    void PatchEmbedding(int phase, Tensor& embeds, BatchCopy& copy, TensorMap& env);

    void PatchSuccessorEmbedding(int phase, Tensor& embeds, BatchCopy& copy, TensorMap& env);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
