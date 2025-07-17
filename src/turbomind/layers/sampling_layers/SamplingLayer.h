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

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/layers/BaseDynamicDecodeLayer.h"
#include "src/turbomind/macro.h"

#include "src/turbomind/engine/request.h"

namespace turbomind {

template<typename T>
class SamplingLayer: public BaseDynamicDecodeLayer {
public:
    explicit SamplingLayer(const BaseParam& param);

    void Setup(const std::vector<const Request*>& rs, const TensorMap&) override;

    void Forward(TensorMap& args) override;

private:
    // host buffer
    Buffer_<int>   kept_;
    Buffer_<int>   top_k_;
    Buffer_<float> top_p_;
    Buffer_<float> min_p_;

    int   max_topk_;
    int   min_topk_;
    float min_topp_;
    float max_minp_;

    // device buffer
    Buffer_<int>   top_k_buf_;
    Buffer_<float> top_p_buf_;
    Buffer_<float> min_p_buf_;

    Buffer_<int> kept_buf_;  // kept sample
};

}  // namespace turbomind
