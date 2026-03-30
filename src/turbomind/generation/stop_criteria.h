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

#include "src/turbomind/core/core.h"

#include "src/turbomind/generation/base_param.h"

namespace turbomind {

struct StopCriteriaData;

class StopCriteria: public BaseGenerationParam {
public:
    explicit StopCriteria(const BaseGenerationParam& base, int phases);

    void Setup(int phase, TensorMap& env);

    void Forward(int phase, TensorMap& env);

private:
    std::vector<std::shared_ptr<StopCriteriaData>> data_;

    Buffer_<int> stop_words_buf_;
    Buffer_<int> max_seq_len_buf_;
};

}  // namespace turbomind
