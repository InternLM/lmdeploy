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
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/weight_type.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"
#include <cuda_bf16.h>

namespace turbomind {

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

    void malloc(cudaStream_t st, bool with_bias = false)
    {
        if (with_bias) {
            deviceMalloc((T**)&bias, output_dims, st);
        }
        const size_t bit_size = getBitSize(type);
        if (bit_size >= 16) {  // fp16, fp32
            deviceMalloc((T**)&kernel, input_dims * output_dims, st);
        }
        else {  // int8, int4
            const int factor = sizeof(float) * 8 / bit_size;
            FT_CHECK(input_dims % factor == 0);
            deviceMalloc((int**)&kernel, input_dims * output_dims / factor, st);
            deviceMalloc((T**)&scales, input_dims / group_size * output_dims, st);
            deviceMalloc((T**)&zeros, input_dims / group_size * output_dims, st);
        }

        if (lora.r > 0) {
            deviceMalloc((T**)&lora.a, input_dims * lora.r, st);
            deviceMalloc((T**)&lora.b, lora.r * output_dims, st);
        }
    }

    void free(cudaStream_t st)
    {
        deviceFree(kernel, st);
        deviceFree(bias, st);
        deviceFree(scales, st);
        deviceFree(zeros, st);
        deviceFree(lora.a, st);
        deviceFree(lora.b, st);
    }
};

template<typename T>
struct LatentAttentionWeight {

    LatentAttentionWeight() = default;

    LatentAttentionWeight(size_t     hidden_dim,
                          size_t     q_lora_rank,
                          size_t     kv_lora_rank,
                          int        head_dim,
                          int        head_num,
                          WeightType weight_type,
                          int        group_size):
        LatentAttentionWeight{}
    {
        if (q_lora_rank) {
            q_a_proj.input_dims  = hidden_dim;
            q_a_proj.output_dims = q_lora_rank;
            q_b_proj.input_dims  = q_lora_rank;
            q_b_proj.output_dims = head_num * head_dim;
            q_b_proj.type = q_a_proj.type = weight_type;
            q_b_proj.group_size = q_a_proj.group_size = group_size;
        }
        else {
            q_proj.input_dims  = hidden_dim;
            q_proj.output_dims = head_num * head_dim;
            q_proj.type        = weight_type;
            q_proj.group_size  = group_size;
        }

        kv_a_proj.input_dims  = hidden_dim;
        kv_a_proj.output_dims = kv_lora_rank;
        kv_b_proj.input_dims  = kv_lora_rank;
        kv_b_proj.output_dims = head_num * head_dim;
        kv_b_proj.type = kv_a_proj.type = weight_type;
        kv_b_proj.group_size = kv_a_proj.group_size = group_size;
    }

    void malloc(cudaStream_t st)
    {
        if (q_proj.output_dims) {
            q_proj.malloc(st);
        }
        else {
            q_a_proj.malloc(st);
            q_b_proj.malloc(st);
            deviceMalloc((T**)q_a_layernorm, q_a_proj.output_dims, st);
        }
        kv_a_proj.malloc(st);
        kv_b_proj.malloc(st);
        deviceMalloc((T**)kv_a_layernorm, kv_a_proj.output_dims, st);
    }

    void free(cudaStream_t st)
    {
        q_proj.free(st);
        q_a_proj.free(st);
        q_b_proj.free(st);
        kv_a_proj.free(st);
        kv_b_proj.free(st);
        deviceFree(q_a_layernorm, st);
        deviceFree(kv_a_layernorm, st);
    }

    LlamaDenseWeight<T> q_proj;

    LlamaDenseWeight<T> q_a_proj;
    LlamaDenseWeight<T> q_b_proj;
    T*                  q_a_layernorm;

    LlamaDenseWeight<T> kv_a_proj;
    LlamaDenseWeight<T> kv_b_proj;
    T*                  kv_a_layernorm;
};

template<typename T>
struct LlamaAttentionWeight {

    LlamaAttentionWeight() = default;

    LlamaAttentionWeight(size_t     hidden_dim,
                         size_t     head_dim,
                         size_t     head_num,
                         size_t     kv_head_num,
                         bool       bias,
                         size_t     tp,
                         WeightType weight_type,
                         int        group_size)
    {
        qkv.input_dims  = hidden_dim;
        qkv.output_dims = (head_num + 2 * kv_head_num) * head_dim / tp;
        qkv.type        = weight_type;
        qkv.group_size  = group_size;

        output.input_dims  = (head_num * head_dim) / tp;
        output.output_dims = hidden_dim;
        output.type        = weight_type;
        output.group_size  = group_size;

        this->bias = bias;
    }

    void malloc(cudaStream_t st)
    {
        qkv.malloc(st, bias);
        output.malloc(st, bias);
    }

    void free(cudaStream_t st)
    {
        qkv.free(st);
        output.free(st);
    }

    LlamaDenseWeight<T> qkv;
    LlamaDenseWeight<T> output;
    bool                bias{};
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

    void malloc(cudaStream_t st)
    {
        gating.malloc(st);
        intermediate.malloc(st);
        output.malloc(st);
    }

    void free(cudaStream_t st)
    {
        gating.free(st);
        intermediate.free(st);
        output.free(st);
        fused_gating_intermediate.free(st);
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

    MoeFfnWeight(int             layer_id,
                 const MoeParam& param,
                 size_t          hidden_dim,
                 WeightType      weight_type,
                 int             group_size,
                 size_t          tp,
                 bool            fuse_silu_act)
    {

        if (param.expert_num.size() <= layer_id) {
            return;
        }

        const int expert_num = param.expert_num[layer_id];

        if (expert_num == 0) {
            return;
        }

        printf("%d %d %d\n", (int)hidden_dim, (int)param.inter_size, (int)expert_num);

        gate.input_dims  = hidden_dim;
        gate.output_dims = expert_num;
        gate.type        = get_default_weight_type<T>();
        gate.group_size  = group_size;

        experts.resize(expert_num);

        method        = param.method;
        fuse_silu_act = fuse_silu_act && method == MoeParam::kFused;

        for (auto& e : experts) {
            // inter size is divided by tp in `FfnWeight`
            e = LlamaFfnWeight<T>{hidden_dim, (size_t)param.inter_size, tp, weight_type, group_size, fuse_silu_act};
        }

        if (param.shared_gate) {
            shared_gate.input_dims  = hidden_dim;
            shared_gate.output_dims = 1;
            shared_gate.type        = get_default_weight_type<T>();
            gate.group_size         = group_size;
        }
        else {
            shared_gate = {};
        }
    }

    void malloc(cudaStream_t st)
    {
        gate.malloc(st);
        if (shared_gate.output_dims) {
            shared_gate.malloc(st);
        }
        for (auto& e : experts) {
            e.malloc(st);
        }
    }

    void free(cudaStream_t st)
    {
        gate.free(st);
        shared_gate.free(st);
        for (auto& e : experts) {
            e.free(st);
        }
        block.free(st);
    }

    LlamaDenseWeight<T>            gate;
    std::vector<LlamaFfnWeight<T>> experts;

    LlamaDenseWeight<T> shared_gate;

    // reference into `experts`
    LlamaFfnWeight<T> block;

    MoeParam::Method method{};
};

}  // namespace turbomind
