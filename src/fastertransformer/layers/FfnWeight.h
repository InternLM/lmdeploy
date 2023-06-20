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

#include "DenseWeight.h"

namespace fastertransformer {

template<typename T1, typename T2 = T1>
struct FfnWeight {
    DenseWeight<T1, T2> gating_weight;
    DenseWeight<T1, T2> intermediate_weight;
    DenseWeight<T1, T2> intermediate_weight2;  // for gated activation
    DenseWeight<T1, T2> output_weight;
    DenseWeight<T1, T2> ia3_weight;
};

}  // namespace fastertransformer
