/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/turbomind/layers/attention_layers/AttentionWeight.h"
#include "src/turbomind/utils/ScaleList.h"

namespace turbomind {

template<typename T1, typename T2>
struct AttentionFP8Weight: public AttentionWeight<T1, T2> {
    const float* qk_scale;
    const float* qk_scale_inv;
    float*       qk_h_scale;
    float*       qk_h_scale_inv;
    float*       identity_scale;
    float*       identity_h_scale;
};

}  // namespace turbomind
