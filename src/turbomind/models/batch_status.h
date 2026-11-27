// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/state.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"

#include <memory>
#include <vector>

namespace turbomind {

class BatchStatus {
public:
    BatchStatus(int max_batch_size, int phases);

    ~BatchStatus();

    void Run(BatchOp op, int phase, TensorMap& env);

    int VerificationPositions(int phase) const;

    int GeneratingCount(int phase) const;

    Buffer_<int> SequenceLength() const;

private:
    struct Data;

    void Setup(int phase, TensorMap& env);
    void Prepare(int phase, TensorMap& env);
    void Unprep(int phase, TensorMap& env);
    void Fetch(int phase, TensorMap& env);

    const int max_batch_size_;

    Buffer_<bool> false_;
    State         finished_;
    State         sequence_length_;

    Buffer_<int>  sequence_length_buf_;
    Buffer_<int>  readonly_block_num_buf_;
    Buffer_<bool> finished_buf_;

    std::vector<std::unique_ptr<Data>> data_;
};

}  // namespace turbomind
