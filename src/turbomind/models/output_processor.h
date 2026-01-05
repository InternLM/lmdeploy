#pragma once

#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class OutputProcessor {
public:
    ~OutputProcessor();

    OutputProcessor(const ModelParam&                    model,  //
                    int                                  max_logits_len,
                    int                                  tp_rank,
                    int                                  phases,
                    std::function<Tensor(const Tensor&)> lm_head);

    void Run(BatchOp op, int phase, TensorMap& env);

    void OutputHiddenStatesAndLogits(int phase, TensorMap& env, int type);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
