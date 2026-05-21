#pragma once

#include <memory>

#include "src/turbomind/generation/base_param.h"

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/stream.h"
#include "xgrammar/matcher.h"

namespace turbomind {

class GuidedDecoding: public BaseGenerationParam {
public:
    explicit GuidedDecoding(const BaseGenerationParam& base, const comm::HostComm& tp_group, int phases);

    void Setup(int phase, TensorMap& env);

    void FillMask(int phase, TensorMap& env);

    void ApplyMask(int phase, TensorMap& env);

    void ScheduleUpdate(int phase, TensorMap& env);
    void FinishUpdate(int phase);

private:
    comm::HostComm tp_group_;

    struct Data;
    std::vector<std::shared_ptr<Data>> data_;

    xgrammar::BatchGrammarMatcher batch_matcher_;

    Tensor_<int32_t> bitmask_buf_;
    Buffer_<int>     output_ids_buf_;

    core::Stream d2h_stream_;     // secondary stream for D2H copy of output_ids
    core::Event  sampling_done_;  // recorded on main stream after sampling
    core::Event  d2h_done_;       // recorded on d2h_stream_ after copy completes
};

}  // namespace turbomind
