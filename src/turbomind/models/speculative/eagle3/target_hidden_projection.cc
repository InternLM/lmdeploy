#include "src/turbomind/models/speculative/eagle3/target_hidden_projection.h"

#include <cstddef>
#include <utility>

#include "src/turbomind/core/context.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"

namespace turbomind {

TargetHiddenProjection::TargetHiddenProjection(const EngineParam&  engine,
                                               const Context&      context,
                                               int                 phases,
                                               int                 target_layer_count,
                                               int                 hidden_units,
                                               DataType            data_type,
                                               std::vector<int>    target_layer_ids,
                                               const LinearWeight& projection_weight):
    HiddenStateTap{*context.is_warm_up},
    data_type_{data_type},
    hidden_units_{hidden_units},
    target_layer_count_{target_layer_count},
    linear_{*context.linear},
    projection_weight_{projection_weight},
    target_layer_ids_{std::move(target_layer_ids)},
    tap_ordinal_by_completed_layer_(target_layer_count + 1, -1),
    collect_{engine,
             context,
             phases,
             static_cast<int>(target_layer_ids_.size()) * hidden_units,
             hidden_units,
             data_type},
    data_(phases)
{
    for (int ordinal = 0; ordinal < static_cast<int>(target_layer_ids_.size()); ++ordinal) {
        tap_ordinal_by_completed_layer_[target_layer_ids_[ordinal]] = ordinal;
    }

    for (auto& d : data_) {
        d.projected_local = Tensor{{collect_.capacity(), hidden_units_}, data_type_, kDEVICE};
    }
}

void TargetHiddenProjection::Begin(int phase, const std::vector<int>& local_token_nums)
{
    BeginPhase(phase);
    collect_.Begin(phase, local_token_nums);
}

void TargetHiddenProjection::BeginPhase(int phase)
{
    active_phase_ = phase;
}

void TargetHiddenProjection::SeedWarmup(int phase, cudaStream_t stream)
{
    collect_.SeedWarmup(phase, stream);
}

int TargetHiddenProjection::TapOrdinal(int completed_layer_count) const
{
    return tap_ordinal_by_completed_layer_[completed_layer_count];
}

void TargetHiddenProjection::Capture(int phase, int tap_ordinal, const Tensor& local_residual, cudaStream_t stream)
{
    const comm::OwnedTokenRows& owned    = collect_.owned(phase);
    Tensor&                     captured = collect_.captured(phase);
    const int                   first    = owned.local_begin();
    const int                   rows     = owned.row_count();

    if (rows == 0) {
        return;
    }

    invokeCaptureTargetHiddenRows(local_residual.raw_data(),
                                  captured.raw_data(),
                                  first,
                                  rows,
                                  hidden_units_,
                                  local_residual.stride(0),
                                  captured.stride(0),
                                  tap_ordinal,
                                  byte_size(data_type_) * 8,
                                  stream);
}

Tensor TargetHiddenProjection::ProjectLocal(int phase)
{
    Data&     d           = data_[phase];
    const int tap_count   = static_cast<int>(target_layer_ids_.size());
    const int input_width = tap_count * hidden_units_;
    const int rows        = collect_.owned(phase).row_count();

    Tensor input  = collect_.captured(phase).slice({0, 0}, {rows, input_width});
    Tensor output = d.projected_local.slice({0, 0}, {rows, hidden_units_});

    if (rows == 0) {
        return output;
    }

    TM_SCOPE_CALL(linear_.Forward(input, projection_weight_, output));

    return output;
}

Tensor TargetHiddenProjection::ProjectAndGather(int phase, cudaStream_t stream)
{
    Tensor active = ProjectLocal(phase);
    return collect_.Gather(phase, data_[phase].projected_local, stream);
}

}  // namespace turbomind
