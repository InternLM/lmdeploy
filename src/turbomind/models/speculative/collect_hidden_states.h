// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/token_ownership.h"
#include "src/turbomind/core/core.h"

#include <vector>

namespace turbomind {

class Context;
class EngineParam;

/// Shared machinery of the hidden-state taps: the token rows this rank owns
/// during the target pass, the buffer that captures them, and the padded
/// all-gather that reunites them across model-TP ranks for the draft pass.
/// `capture_width` is the per-row width a tap writes (H, or L * H when several
/// target layers are tapped); `hidden_units` is the width the gather returns.
class CollectHiddenStates {
public:
    CollectHiddenStates(const EngineParam& engine,
                        const Context&     context,
                        int                phases,
                        int                capture_width,
                        int                hidden_units,
                        DataType           data_type);

    /// Records the token rows this rank owns for the round.
    void Begin(int phase, const std::vector<int>& local_token_nums);

    /// Zero-fills captured rows on warmup rounds, where no capture happens.
    void SeedWarmup(int phase, cudaStream_t stream);

    /// Reunites the owned rows of `local_buffer` (a per-phase backing buffer of
    /// capacity rows); returns the full-width active view the draft pass uses.
    Tensor Gather(int phase, const Tensor& local_buffer, cudaStream_t stream);

    Tensor& captured(int phase)
    {
        return data_.at(phase).captured;
    }

    int capacity() const
    {
        return capacity_;
    }

    const comm::OwnedTokenRows& owned(int phase) const
    {
        return data_.at(phase).owned;
    }

private:
    struct Data {
        Tensor               captured;
        Tensor               gathered_padded;
        comm::OwnedTokenRows owned;
        int                  token_num{};
    };

    int        capture_width_;
    int        hidden_units_;
    int        capacity_;
    int        attn_dp_rank_;
    int        model_tp_group_;
    int        model_tp_rank_;
    int        model_tp_size_;
    comm::DeviceCommImpl* d_comm_;
    std::vector<Data>     data_;
};

}  // namespace turbomind
