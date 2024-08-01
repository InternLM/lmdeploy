// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/layers/DynamicDecodeBaseLayer.h"
#include "src/turbomind/macro.h"
#include <vector>

namespace turbomind {

template<typename T>
class StopCriteriaLayer: public DynamicDecodeBaseLayer {
public:
    using DynamicDecodeBaseLayer::DynamicDecodeBaseLayer;

    void setup(const size_t batch_size, const size_t beam_width, TensorMap* params) override;

    void forward(TensorMap* output_tensors, TensorMap* input_tensors) override;

    ~StopCriteriaLayer();

private:
    void allocateBuffer() override;

    void freeBuffer() override;

    // host buffer
    int* h_pinned_finished_sum_{};
};

}  // namespace turbomind
