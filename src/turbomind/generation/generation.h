// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/speculative/speculative_policy.h"

namespace turbomind {

namespace comm {
class HostComm;
}

class TargetVerification;

class Generation {
public:
    ~Generation();

    Generation(DataType              data_type,
               int                   max_batch_size,
               int                   session_len,
               int                   vocab_size,
               int                   vocab_size_padded,
               int                   hidden_units,
               DataType              hidden_dtype,
               const comm::HostComm& tp_group,
               int                   phases,
               const SpeculativePolicy* policy,
               bool                     enable_metrics);

    void Run(BatchOp op, int phase, TensorMap& env);

    // The composed engine's verification component, holding the speculative
    // round's verification steps and buffers. Null in target-only engines.
    TargetVerification* Verification() noexcept;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
