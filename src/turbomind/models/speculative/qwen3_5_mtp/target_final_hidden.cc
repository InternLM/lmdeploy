// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/qwen3_5_mtp/target_final_hidden.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/models/llama/context.h"

namespace turbomind {

TargetFinalHidden::TargetFinalHidden(const EngineParam& engine,
                                     const Context&     context,
                                     int                phases,
                                     int                target_layer_count,
                                     int                hidden_units,
                                     DataType           data_type):
    HiddenStateTap{*context.is_warm_up},
    target_layer_count_{target_layer_count},
    hidden_units_{hidden_units},
    collect_{engine, context, phases, hidden_units, hidden_units, data_type}
{
}

void TargetFinalHidden::Begin(int phase, const std::vector<int>& local_token_nums)
{
    active_phase_ = phase;
    collect_.Begin(phase, local_token_nums);
}

void TargetFinalHidden::Capture(
    int, const Tensor&, const Tensor& local_normalized_hidden, cudaStream_t stream)
{
    const comm::OwnedTokenRows& owned = collect_.owned(active_phase_);
    const int first = owned.local_begin();
    const int rows  = owned.row_count();
    if (rows == 0) {
        return;
    }

    Tensor& captured = collect_.captured(active_phase_);

    const size_t element_bytes = byte_size(captured.dtype());
    const size_t row_bytes     = size_t(hidden_units_) * element_bytes;
    const auto* source = static_cast<const uint8_t*>(local_normalized_hidden.raw_data())
                         + size_t(first) * local_normalized_hidden.stride(0) * element_bytes;
    TM_CUDA_CHECK(cudaMemcpy2DAsync(captured.raw_data(),
                                    captured.stride(0) * element_bytes,
                                    source,
                                    local_normalized_hidden.stride(0) * element_bytes,
                                    row_bytes,
                                    rows,
                                    cudaMemcpyDeviceToDevice,
                                    stream));
}

void TargetFinalHidden::SeedWarmup(int phase, cudaStream_t stream)
{
    collect_.SeedWarmup(phase, stream);
}

Tensor TargetFinalHidden::Gather(int phase, cudaStream_t stream)
{
    return collect_.Gather(phase, collect_.captured(phase), stream);
}

}  // namespace turbomind
