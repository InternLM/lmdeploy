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

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/DenseWeight.h

#pragma once

#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <cuda_bf16.h>

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

template<class T>
constexpr WeightType get_default_weight_type()
{
    if constexpr (std::is_same_v<T, half>) {
        return WeightType::kFP16;
    }
    else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return WeightType::kBF16;
    }
    else if constexpr (std::is_same_v<T, float>) {
        return WeightType::kFP32;
    }
    else {
        static_assert(sizeof(T) != sizeof(T), "not implemented");
        return {};
    }
}

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
    T*         scales;
    T*         zeros;
    T*         scales_zeros;
    int        group_size;

    gemm::MatrixLayout k_desc;
    gemm::MatrixLayout q_desc;

    size_t kernel_size() const noexcept
    {
        return getBitSize(type) * input_dims * output_dims / 8;
    }

    size_t bias_size() const noexcept
    {
        return sizeof(T) * output_dims;
    }

    size_t scales_size() const noexcept
    {
        return sizeof(T) * input_dims / group_size * output_dims;
    }

    std::pair<size_t, size_t> lora_size() const noexcept
    {
        return {sizeof(T) * input_dims * lora.r, sizeof(T) * lora.r * output_dims};
    }
};

template<typename T>
struct LlamaAttentionWeight {
    LlamaDenseWeight<T> qkv;
    LlamaDenseWeight<T> output;
};

template<typename T>
struct LlamaFfnWeight {

    LlamaFfnWeight() = default;

    LlamaFfnWeight(
        size_t hidden_dim, size_t inter_size, size_t tp, WeightType weight_type, int group_size, bool fuse_silu_act)
    {
        inter_size /= tp;

        this->inter_size = inter_size;

        gating.input_dims  = hidden_dim;
        gating.output_dims = inter_size;
        gating.type        = weight_type;
        gating.group_size  = group_size;

        intermediate.input_dims  = hidden_dim;
        intermediate.output_dims = inter_size;
        intermediate.type        = weight_type;
        intermediate.group_size  = group_size;

        fused_gating_intermediate.input_dims  = hidden_dim;
        fused_gating_intermediate.output_dims = inter_size * 2;
        fused_gating_intermediate.type        = weight_type;
        fused_gating_intermediate.group_size  = group_size;

        is_fused_silu = fuse_silu_act;

        output.input_dims  = inter_size;
        output.output_dims = hidden_dim;
        output.type        = weight_type;
        output.group_size  = group_size;
    }

    LlamaDenseWeight<T> gating;
    LlamaDenseWeight<T> intermediate;
    LlamaDenseWeight<T> output;
    LlamaDenseWeight<T> fused_gating_intermediate;

    int  inter_size{};
    bool is_fused_silu{};
};

template<class T>
struct MoeFfnWeight {

    MoeFfnWeight() = default;

    MoeFfnWeight(size_t     hidden_dim,
                 int        inter_size,
                 int        expert_num,
                 int        method,
                 bool       has_shared_gate,
                 size_t     tp,
                 WeightType weight_type,
                 int        group_size,
                 bool       fuse_silu_act)
    {

        // printf("%d %d %d\n", (int)hidden_dim, (int)inter_size, (int)expert_num);

        if (expert_num == 0) {
            return;
        }

        gate.input_dims  = hidden_dim;
        gate.output_dims = expert_num;
        gate.type        = get_default_weight_type<T>();
        gate.group_size  = group_size;

        experts.resize(expert_num);

        this->method  = method;
        fuse_silu_act = fuse_silu_act && method;

        for (auto& e : experts) {
            // inter size is divided by tp in `FfnWeight`
            e = LlamaFfnWeight<T>{hidden_dim, (size_t)inter_size, tp, weight_type, group_size, fuse_silu_act};
        }

        if (has_shared_gate) {
            shared_gate.input_dims  = hidden_dim;
            shared_gate.output_dims = 1;
            shared_gate.type        = get_default_weight_type<T>();
            gate.group_size         = group_size;
        }
        else {
            shared_gate = {};
        }
    }

    LlamaDenseWeight<T>            gate;
    std::vector<LlamaFfnWeight<T>> experts;

    LlamaDenseWeight<T> shared_gate;

    LlamaFfnWeight<T> block;

    int method{};
};

}  // namespace turbomind
