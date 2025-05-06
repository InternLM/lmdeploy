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

#include "src/turbomind/engine/request.h"

namespace turbomind {

template<typename T>
class LogitsProcessorLayer: public BaseDynamicDecodeLayer {
public:
    explicit LogitsProcessorLayer(const BaseParam& param);

    void Setup(const std::vector<const Request*>& rs, const TensorMap& args) override;

    void Forward(TensorMap& args) override;

private:
    // repetition penalty type
    RepetitionPenaltyType repetition_penalty_type_ = RepetitionPenaltyType::None;

    // host buffer
    Buffer_<float> repetition_penalty_;
    Buffer_<int>   min_lengths_;
    Buffer_<float> temperature_;
    Buffer_<int>   bad_words_;
    Buffer_<int>   end_ids_;

    // device buffer
    Buffer_<float> repetition_penalty_buf_;
    Buffer_<int>   min_lengths_buf_;
    Buffer_<float> temperature_buf_;
    Buffer_<int>   bad_words_buf_;
    Buffer_<int>   end_ids_buf_;

    Tensor_<int> bad_words_ten_;
    Tensor_<int> end_ids_ten_;
};

}  // namespace turbomind
