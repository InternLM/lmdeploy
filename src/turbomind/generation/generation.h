

#pragma once

#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"

namespace turbomind {

namespace comm {
class HostComm;
}

struct GenerationData;

class Generation {
public:
    ~Generation();

    Generation(DataType              data_type,  //
               int                   max_batch_size,
               int                   session_len,
               int                   vocab_size,
               int                   vocab_size_padded,
               const comm::HostComm& tp_group,
               int                   phases);

    void Run(BatchOp op, int phase, TensorMap& env);

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
