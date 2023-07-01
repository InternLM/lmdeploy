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

#include "src/turbomind/layers/DenseWeight.h"

namespace turbomind {

template<typename T1, typename T2 = T1>
struct AttentionWeight {
    DenseWeight<T1, T2> query_weight;
    DenseWeight<T1, T2> key_weight;
    DenseWeight<T1, T2> value_weight;
    DenseWeight<T1, T2> attention_output_weight;
    DenseWeight<T1, T2> ia3_key_weight;
    DenseWeight<T1, T2> ia3_value_weight;
};

}  // namespace turbomind
