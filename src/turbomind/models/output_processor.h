#pragma once

#include "src/turbomind/engine/batch.h"

namespace turbomind {

class LanguageModel;

class OutputProcessor {
public:
    ~OutputProcessor();

    OutputProcessor(LanguageModel& model, int tp_rank, int phases);

    void Run(BatchOp op, int phase, TensorMap& env);

    void OutputHiddenStatesAndLogits(int phase, TensorMap& env, int type);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
