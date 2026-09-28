#pragma once

#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/speculative/collect_hidden_states.h"
#include "src/turbomind/models/speculative/eagle3/target_hidden_projection_kernels.h"
#include "src/turbomind/models/speculative/hidden_state_tap.h"

namespace turbomind {

class Context;
class EngineParam;
class LinearWeight;
class LlamaLinear;

/// EAGLE3's tap: captures residuals of the configured target layers, projects
/// the concatenated rows through the eagle fc, and reunites the result across
/// model-TP ranks for the draft pass.
class TargetHiddenProjection: public HiddenStateTap {
public:
    TargetHiddenProjection(const EngineParam&  engine,
                           const Context&      context,
                           int                 phases,
                           int                 target_layer_count,
                           int                 hidden_units,
                           DataType            data_type,
                           std::vector<int>    target_layer_ids,
                           const LinearWeight& projection_weight);

    void Begin(int phase, const std::vector<int>& local_token_nums) override;

    void BeginPhase(int phase);

    void SeedWarmup(int phase, cudaStream_t stream) override;

    int TapOrdinal(int completed_layer_count) const override;

    void Capture(int           tap_ordinal,
                 const Tensor& local_residual,
                 const Tensor&,
                 cudaStream_t  stream) override
    {
        Capture(active_phase_, tap_ordinal, local_residual, stream);
    }

    void Capture(int phase, int tap_ordinal, const Tensor& local_residual, cudaStream_t stream);

    Tensor ProjectLocal(int phase);

    Tensor ProjectAndGather(int phase, cudaStream_t stream);

private:
    struct Data {
        Tensor projected_local;
    };

    DataType              data_type_;
    int                   hidden_units_;
    int                   target_layer_count_;
    LlamaLinear&          linear_;
    const LinearWeight&   projection_weight_;
    std::vector<int>      target_layer_ids_;
    std::vector<int>      tap_ordinal_by_completed_layer_;
    CollectHiddenStates   collect_;
    std::vector<Data>     data_;
    int                   active_phase_{};
};

}  // namespace turbomind
