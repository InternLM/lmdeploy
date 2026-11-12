// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"

#include <cuda_runtime.h>

#include <vector>

namespace turbomind {

class HiddenStateTap {
public:
    explicit HiddenStateTap(const int& is_warm_up):
        is_warm_up_{is_warm_up}
    {
    }

    virtual ~HiddenStateTap() = default;

    /// Arms the tap for a target pass: records the token rows this rank owns
    /// for the round and sizes the captured state. On warm-up rounds, where no
    /// capture happens, zero-fills the captured state instead and returns null
    /// so the decoder runs untapped. Arming is the tap's own business; callers
    /// hold no warm-up knowledge.
    HiddenStateTap* Arm(int phase, const std::vector<int>& local_token_nums, cudaStream_t stream)
    {
        Begin(phase, local_token_nums);
        if (is_warm_up_) {
            SeedWarmup(phase, stream);
            return nullptr;
        }
        return this;
    }

    /// Called before each target forward: records the token rows this rank owns
    /// for the round and sizes the captured state.
    virtual void Begin(int phase, const std::vector<int>& local_token_nums) = 0;

    /// Zero-fills captured state on warmup rounds, where no capture happens.
    virtual void SeedWarmup(int phase, cudaStream_t stream) = 0;

    virtual int TapOrdinal(int completed_layer_count) const = 0;

    virtual void Capture(int           tap_ordinal,
                         const Tensor& local_residual,
                         const Tensor& local_normalized_hidden,
                         cudaStream_t  stream) = 0;

private:
    const int& is_warm_up_;
};

}  // namespace turbomind
