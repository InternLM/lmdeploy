/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
