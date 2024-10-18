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

#include "src/turbomind/kernels/penalty_types.h"
#include "src/turbomind/layers/DynamicDecodeBaseLayer.h"
#include "src/turbomind/macro.h"
#include <vector>

namespace turbomind {

template<typename T>
class LogitsProcessorLayer: public DynamicDecodeBaseLayer {
public:
    using DynamicDecodeBaseLayer::DynamicDecodeBaseLayer;
    using DynamicDecodeBaseLayer::args_;

    void setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args) override;

    void forward(TensorMap* output_tensors, TensorMap* input_tensors) override;

    ~LogitsProcessorLayer();

private:
    void allocateBuffer() override;

    void allocateBuffer(const size_t batch_size);

    void freeBuffer() override;

    // repetition penalty type
    RepetitionPenaltyType repetition_penalty_type_ = RepetitionPenaltyType::None;

    // host buffer
    std::vector<float> repetition_penalty_;
    std::vector<int>   min_lengths_;
    std::vector<float> temperature_;
    std::vector<int>   context_length_;

    // device buffer
    int*   repetition_penalty_workspace_ = nullptr;
    float* repetition_penalty_buf_       = nullptr;
    int*   min_lengths_buf_              = nullptr;
    float* temperature_buf_              = nullptr;
};

}  // namespace turbomind
