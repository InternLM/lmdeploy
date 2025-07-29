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

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"

#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/llama/llama_params.h"

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

struct LlamaDenseWeight: public core::Module {

    LlamaDenseWeight(): data_type{}, weight_type{}, lora{}, k_desc{}, q_desc{} {}

    void emplace(int input_dim, int output_dim, DataType data_type, bool bias, DataType weight_type, int group_size);

    void prepare(bool fused_moe, bool use_simt);

    void cpu()
    {
        weight       = to_cpu(weight);
        bias         = to_cpu(bias);
        scales       = to_cpu(scales);
        zeros        = to_cpu(zeros);
        scales_zeros = to_cpu(scales_zeros);
    }

    void cuda()
    {
        weight       = to_cuda(weight);
        bias         = to_cuda(bias);
        scales       = to_cuda(scales);
        zeros        = to_cuda(zeros);
        scales_zeros = to_cuda(scales_zeros);
    }

    LlamaDenseWeight& operator=(std::nullptr_t)
    {
        this->~LlamaDenseWeight();
        new (this) LlamaDenseWeight{};
        return *this;
    }

    operator bool() const noexcept
    {
        return static_cast<bool>(weight);
    }

    int input_dim  = 0;
    int output_dim = 0;
    int group_size = 1;

    DataType data_type;
    DataType weight_type;

    Tensor weight;
    Tensor bias;

    Tensor scales;
    Tensor zeros;

    Tensor scales_zeros;

    LoraWeight lora;

    gemm::MatrixLayout k_desc;
    gemm::MatrixLayout q_desc;
};

struct LlamaAttentionWeight: public core::Module {

    LlamaAttentionWeight() = default;

    LlamaAttentionWeight(int      hidden_dim,
                         int      head_dim,
                         int      head_num,
                         int      kv_head_num,
                         MLAParam mla,
                         bool     bias,
                         bool     qk_norm,
                         int      tp_size,
                         int      tp_rank,
                         DataType data_type,
                         DataType weight_type,
                         int      group_size);

    void prepare(bool use_simt);

    void cpu()
    {
        qkv.cpu();
        output.cpu();
        q_proj.cpu();
        q_a_proj.cpu();
        q_b_proj.cpu();
        kv_a_proj.cpu();
        kv_b_proj.cpu();
        q_a_layernorm  = to_cpu(q_a_layernorm);
        kv_a_layernorm = to_cpu(kv_a_layernorm);
    }

    void cuda()
    {
        qkv.cuda();
        output.cuda();
        q_proj.cuda();
        q_a_proj.cuda();
        q_b_proj.cuda();
        kv_a_proj.cuda();
        kv_b_proj.cuda();
        q_a_layernorm  = to_cuda(q_a_layernorm);
        kv_a_layernorm = to_cuda(kv_a_layernorm);
    }

    LlamaDenseWeight qkv;
    LlamaDenseWeight output;

    LlamaDenseWeight q_proj;
    LlamaDenseWeight q_a_proj;
    LlamaDenseWeight q_b_proj;
    LlamaDenseWeight kv_a_proj;
    LlamaDenseWeight kv_b_proj;

    Tensor q_a_layernorm;
    Tensor kv_a_layernorm;
};

struct LlamaFfnWeight: core::Module {

    LlamaFfnWeight() = default;

    LlamaFfnWeight(int      hidden_dim,
                   int      inter_size,
                   int      tp_size,
                   int      tp_rank,
                   DataType data_type,
                   DataType weight_type,
                   int      group_size,
                   bool     fuse_silu_act);

    static constexpr bool fuse_up_and_gate = true;

    void prepare(bool fused_moe, bool use_simt);

    void cpu()
    {
        gating.cpu();
        intermediate.cpu();
        output.cpu();
        fused_gating_intermediate.cpu();
    }

    void cuda()
    {
        gating.cuda();
        intermediate.cuda();
        output.cuda();
        fused_gating_intermediate.cuda();
    }

    LlamaDenseWeight gating;
    LlamaDenseWeight intermediate;
    LlamaDenseWeight output;
    LlamaDenseWeight fused_gating_intermediate;

    int  inter_size{};
    bool is_fused_silu{};
};

struct MoeFfnWeight: core::Module {

    MoeFfnWeight() = default;

    MoeFfnWeight(int             layer_id,
                 const MoeParam& param,
                 int             hidden_dim,
                 DataType        data_type,
                 DataType        weight_type,
                 int             group_size,
                 int             tp_size,
                 int             tp_rank,
                 bool            fuse_silu_act);

    void prepare(bool use_simt);

    void cpu()
    {
        gate.cpu();
        shared_gate.cpu();
        for (auto& expert : experts) {
            expert->cpu();
        }
        block.cpu();
    }

    void cuda()
    {
        gate.cuda();
        shared_gate.cuda();
        for (auto& expert : experts) {
            expert->cuda();
        }
        block.cuda();
    }

    LlamaDenseWeight gate;
    LlamaDenseWeight shared_gate;

    std::vector<std::unique_ptr<LlamaFfnWeight>> experts;

    // reference into `experts`
    LlamaFfnWeight block;

    MoeParam::Method method{};
};

}  // namespace turbomind
