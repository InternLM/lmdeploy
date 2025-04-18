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

#include "src/turbomind/layers/BaseDynamicDecodeLayer.h"

#include "src/turbomind/engine/request.h"

namespace turbomind {

template<typename T>
class StopCriteriaLayer: public BaseDynamicDecodeLayer {
public:
    explicit StopCriteriaLayer(const BaseParam& param);

    void Setup(const std::vector<const Request*>& rs, const TensorMap&) override;

    void Forward(TensorMap& args) override;

private:
    Buffer_<int> stop_words_;
    Buffer_<int> stop_words_buf_;
    Tensor_<int> stop_words_ten_;
};

}  // namespace turbomind
