// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/speculative/collect_hidden_states.h"
#include "src/turbomind/models/speculative/hidden_state_tap.h"

namespace turbomind {

class Context;
class EngineParam;

/// Tap for the target's final normalized hidden: captures this rank's owned
/// rows of the last decoder layer's output and reunites them for the draft pass.
class TargetFinalHidden final: public HiddenStateTap {
public:
    TargetFinalHidden(const EngineParam& engine,
                      const Context&     context,
                      int                phases,
                      int                target_layer_count,
                      int                hidden_units,
                      DataType           data_type);

    void Begin(int phase, const std::vector<int>& local_token_nums) override;
    void SeedWarmup(int phase, cudaStream_t stream) override;
    Tensor Gather(int phase, cudaStream_t stream);

    int TapOrdinal(int completed_layer_count) const override
    {
        return completed_layer_count == target_layer_count_ ? 0 : -1;
    }

    void Capture(int,
                 const Tensor&,
                 const Tensor& local_normalized_hidden,
                 cudaStream_t  stream) override;

private:
    int                  target_layer_count_;
    int                  hidden_units_;
    CollectHiddenStates  collect_;
    int                  active_phase_{};
};

}  // namespace turbomind
