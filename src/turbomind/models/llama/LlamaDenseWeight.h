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

#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/tensor.h"

#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/Tensor.h"
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

struct LlamaDenseWeight {

    int input_dim  = 0;
    int output_dim = 0;
    int group_size = 1;

    DataType data_type;
    DataType weight_type;

    core::Tensor weight;
    core::Buffer bias;

    core::Tensor scales;
    core::Tensor zeros;

    core::Tensor scales_zeros;

    LoraWeight lora;

    gemm::MatrixLayout k_desc;
    gemm::MatrixLayout q_desc;

    LlamaDenseWeight(): data_type{}, weight_type{}, lora{}, k_desc{}, q_desc{} {}

    LlamaDenseWeight(int input_dim, int output_dim, DataType data_type, DataType weight_type, int group_size):
        LlamaDenseWeight{}
    {
        this->data_type   = data_type;
        this->weight_type = weight_type;
        this->input_dim   = input_dim;
        this->output_dim  = output_dim;
        this->group_size  = group_size;
    }

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(weight);
    }

    void malloc(bool with_bias = false)
    {
        if (with_bias) {
            bias = core::Buffer{output_dim, data_type, MEMORY_GPU};
        }

        weight = core::Tensor({input_dim, output_dim}, weight_type, MEMORY_GPU);

        if (auto wbits = core::get_byte_size(weight_type, 8); wbits <= 8) {
            TM_CHECK_EQ(input_dim % group_size, 0);
            scales = core::Tensor{{input_dim / group_size, output_dim}, data_type, MEMORY_GPU};
            zeros  = core::Tensor{{input_dim / group_size, output_dim}, data_type, MEMORY_GPU};
        }
    }

    void free()
    {
        bias   = {};
        weight = {};
        scales = {};
        zeros  = {};
    }
};

struct LlamaAttentionWeight {

    LlamaAttentionWeight() = default;

    LlamaAttentionWeight(int      hidden_dim,
                         int      head_dim,
                         int      head_num,
                         int      kv_head_num,
                         MLAParam mla,
                         bool     bias,
                         bool     qk_norm,
                         int      tp,
                         DataType data_type,
                         DataType weight_type,
                         int      group_size)
    {
        this->bias        = bias;
        this->head_dim    = head_dim;
        this->qk_norm     = qk_norm;
        this->data_type   = data_type;
        this->weight_type = weight_type;

        if (mla.kv_lora_rank == 0) {
            qkv = {hidden_dim, (head_num + 2 * kv_head_num) * head_dim / tp, data_type, weight_type, group_size};
        }
        else {
            const int qk_nope_dim = head_dim - mla.qk_rope_dim;
            if (mla.q_lora_rank) {
                q_a_proj = {hidden_dim, mla.q_lora_rank, data_type, weight_type, group_size};
                q_b_proj = {mla.q_lora_rank, head_num * head_dim / tp, data_type, weight_type, group_size};
            }
            else {
                q_proj = {hidden_dim, head_num * head_dim / tp, data_type, weight_type, group_size};
            }
            kv_a_proj = {hidden_dim, mla.kv_lora_rank + mla.qk_rope_dim, data_type, weight_type, group_size};
            kv_b_proj = {
                mla.kv_lora_rank, head_num * (qk_nope_dim + mla.v_head_dim) / tp, data_type, weight_type, group_size};
        }
        output = {(head_num * head_dim) / tp, hidden_dim, data_type, weight_type, group_size};
    }

    void malloc()
    {
        if (qkv.output_dim) {
            qkv.malloc(bias);
            if (qk_norm) {
                q_a_layernorm  = core::Buffer{head_dim, data_type, MEMORY_GPU};
                kv_a_layernorm = core::Buffer{head_dim, data_type, MEMORY_GPU};
            }
        }
        else {  // MLA
            if (q_proj.output_dim) {
                q_proj.malloc();
            }
            else {
                q_a_proj.malloc();
                q_b_proj.malloc();
                q_a_layernorm = core::Buffer{q_b_proj.input_dim, data_type, MEMORY_GPU};
            }
            kv_a_proj.malloc();
            kv_b_proj.malloc();
            kv_a_layernorm = core::Buffer{kv_b_proj.input_dim, data_type, MEMORY_GPU};
        }
        output.malloc(bias);
    }

    void free()
    {
        qkv.free();
        q_proj.free();
        q_a_proj.free();
        q_b_proj.free();
        kv_a_proj.free();
        kv_b_proj.free();
        output.free();
        q_a_layernorm  = {};
        kv_a_layernorm = {};
    }

    int  head_dim{};
    bool bias{};
    bool qk_norm{};

    DataType data_type{};
    DataType weight_type{};

    LlamaDenseWeight qkv;
    LlamaDenseWeight output;

    LlamaDenseWeight q_proj;
    LlamaDenseWeight q_a_proj;
    LlamaDenseWeight q_b_proj;
    LlamaDenseWeight kv_a_proj;
    LlamaDenseWeight kv_b_proj;

    core::Buffer q_a_layernorm;
    core::Buffer kv_a_layernorm;
};

struct LlamaFfnWeight {

    LlamaFfnWeight() = default;

    LlamaFfnWeight(int      hidden_dim,
                   int      inter_size,
                   int      tp,
                   DataType data_type,
                   DataType weight_type,
                   int      group_size,
                   bool     fuse_silu_act)
    {
        TM_CHECK_EQ(inter_size % tp, 0);

        this->inter_size = inter_size;

        gating       = {hidden_dim, inter_size, data_type, weight_type, group_size};
        intermediate = {hidden_dim, inter_size, data_type, weight_type, group_size};

        fused_gating_intermediate = {hidden_dim, inter_size * 2, data_type, weight_type, group_size};
        is_fused_silu             = fuse_silu_act;

        output = {inter_size, hidden_dim, data_type, weight_type, group_size};
    }

    void malloc()
    {
        gating.malloc();
        intermediate.malloc();
        output.malloc();
    }

    void free()
    {
        gating.free();
        intermediate.free();
        output.free();
        fused_gating_intermediate.free();
    }

    LlamaDenseWeight gating;
    LlamaDenseWeight intermediate;
    LlamaDenseWeight output;
    LlamaDenseWeight fused_gating_intermediate;

    int  inter_size{};
    bool is_fused_silu{};
};

struct MoeFfnWeight {

    MoeFfnWeight() = default;

    MoeFfnWeight(int             layer_id,
                 const MoeParam& param,
                 int             hidden_dim,
                 DataType        data_type,
                 DataType        weight_type,
                 int             group_size,
                 int             tp,
                 bool            fuse_silu_act)
    {

        if ((int)param.expert_num.size() <= layer_id) {
            return;
        }

        const int expert_num = param.expert_num[layer_id];

        if (expert_num == 0) {
            return;
        }

        // printf("%d %d %d\n", (int)hidden_dim, (int)param.inter_size, (int)expert_num);

        gate = {hidden_dim, expert_num, data_type, data_type, 1};

        experts.resize(expert_num);

        method        = param.method;
        fuse_silu_act = fuse_silu_act && method == MoeParam::kFused;

        for (auto& e : experts) {
            // inter size is divided by tp in `FfnWeight`
            e = LlamaFfnWeight{hidden_dim, param.inter_size, tp, data_type, weight_type, group_size, fuse_silu_act};
        }

        if (param.shared_gate) {
            shared_gate = {hidden_dim, 1, data_type, data_type, 1};
        }
    }

    void malloc()
    {
        gate.malloc();
        if (shared_gate.output_dim) {
            shared_gate.malloc();
        }
        for (auto& e : experts) {
            e.malloc();
        }
    }

    void free()
    {
        gate.free();
        shared_gate.free();
        for (auto& e : experts) {
            e.free();
        }
        block.free();
    }

    LlamaDenseWeight            gate;
    std::vector<LlamaFfnWeight> experts;

    LlamaDenseWeight shared_gate;

    // reference into `experts`
    LlamaFfnWeight block;

    MoeParam::Method method{};
};

}  // namespace turbomind
