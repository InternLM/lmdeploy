#pragma once

#include <memory>

#include "src/turbomind/generation/base_param.h"

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/core.h"

namespace turbomind {

class GuidedDecoding: public BaseGenerationParam {
public:
    explicit GuidedDecoding(const BaseGenerationParam& base, const comm::HostComm& tp_group, int phases);

    void Setup(int phase, TensorMap& env);

    void FillMask(int phase, TensorMap& env);

    void ApplyMask(int phase, TensorMap& env);

    void Update(int phase, TensorMap& env);

private:
    comm::HostComm tp_group_;

    struct Data;
    std::vector<std::shared_ptr<Data>> data_;

    Tensor_<int32_t> bitmask_buf_;
    Buffer_<int>     output_ids_buf_;
};

}  // namespace turbomind
