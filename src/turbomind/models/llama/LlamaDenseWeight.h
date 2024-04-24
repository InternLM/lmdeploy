/*
 * Copyright (c) OpenMMLab. All rights reserved.
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

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/layers/DenseWeight.h

#pragma once

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

enum class WeightType : int
{
    kFP32,
    kFP16,
    kFP8,  // not supported yet
    kBF16,
    kINT8,
    kINT4
};

inline size_t getBitSize(WeightType type)
{
    switch (type) {
        case WeightType::kFP32:
            return 32;
        case WeightType::kFP16:
            return 16;
        case WeightType::kFP8:
            return 8;
        case WeightType::kBF16:
            return 16;
        case WeightType::kINT8:
            return 8;
        case WeightType::kINT4:
            return 4;
    }
    return 0;
}

enum class LoraPolicy : int
{
    kNull,
    kPlora,
};

inline LoraPolicy getLoraPolicy(const std::string& policy)
{
    if (policy == "plora") {
        return LoraPolicy::kPlora;
    }
    return LoraPolicy::kNull;
}

struct LoraWeight {
    LoraPolicy policy;
    int        r;
    float      scale;
    void*      a;
    void*      b;
};

template<typename T>
struct LlamaDenseWeight {
    size_t     input_dims;
    size_t     output_dims;
    void*      kernel;
    LoraWeight lora;
    WeightType type;
    T*         bias;
    T*         scales_and_zeros;
    int        group_size;
};

template<typename T>
struct LlamaAttentionWeight {
    LlamaDenseWeight<T> qkv;
    LlamaDenseWeight<T> output;
};

template<typename T>
struct LlamaFfnWeight {
    LlamaDenseWeight<T> gating;
    LlamaDenseWeight<T> intermediate;
    LlamaDenseWeight<T> output;
    LlamaDenseWeight<T> fused_gating_intermediate;
};

}  // namespace turbomind
