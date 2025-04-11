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

#include <vector>

#include "src/turbomind/kernels/penalty_types.h"
#include "src/turbomind/layers/BaseDynamicDecodeLayer.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/Tensor.h"

#include "src/turbomind/engine/request.h"

namespace turbomind {

template<typename T>
class LogitsProcessorLayer: public BaseDynamicDecodeLayer {
public:
    explicit LogitsProcessorLayer(const BaseParam& param);

    void Setup(const std::vector<const Request*>& rs) override;

    void Forward(core::TensorMap& args) override;

private:
    // repetition penalty type
    RepetitionPenaltyType repetition_penalty_type_ = RepetitionPenaltyType::None;

    // host buffer
    core::Buffer_<float> repetition_penalty_;
    core::Buffer_<int>   min_lengths_;
    core::Buffer_<float> temperature_;
    core::Buffer_<int>   bad_words_;
    core::Buffer_<int>   end_ids_;

    std::vector<int> context_length_;
    std::vector<int> prompt_length_;

    // device buffer
    core::Buffer_<float> repetition_penalty_buf_;
    core::Buffer_<int>   min_lengths_buf_;
    core::Buffer_<float> temperature_buf_;
    core::Buffer_<int>   bad_words_buf_;
    core::Buffer_<int>   end_ids_buf_;

    core::Tensor_<int> bad_words_ten_;
    core::Tensor_<int> end_ids_ten_;
};

}  // namespace turbomind
