#pragma once

#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class InputProcessor {
public:
    ~InputProcessor();

    InputProcessor(const EngineParam& engine, int hidden_units, DataType data_type, int phases);

    void Run(BatchOp op, int phase, TensorMap& env);

    void PatchEmbedding(int phase, Tensor& embeds, BatchCopy& copy);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
