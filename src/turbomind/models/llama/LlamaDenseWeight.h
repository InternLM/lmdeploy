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

#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

using gemm::QuantDesc;
using gemm::MatrixLayout;
using gemm::Epilogue;

struct LlamaDenseWeight: public core::Module {

    LlamaDenseWeight():
        data_type{}, weight_type{}, input_type{}, weight_quant{}, input_quant{}, epilogue{}, k_desc{}, q_desc{}
    {
    }

    void emplace(int input_dim, int output_dim, DataType data_type, bool bias, DataType weight_type, int group_size);

    void preprocess();

    void prepare(bool fused_moe = 0);

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

    Tensor weight;
    Tensor bias;

    Tensor scales;
    Tensor zeros;

    DataType data_type;

    DataType weight_type;
    DataType input_type;

    QuantDesc weight_quant;
    QuantDesc input_quant;

    Epilogue epilogue;

    MatrixLayout k_desc;
    MatrixLayout q_desc;
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
                         int      group_size,
                         int      window_size,
                         bool     sink);

    void prepare();

    LlamaDenseWeight qkv;
    LlamaDenseWeight output;

    Tensor sinks;

    LlamaDenseWeight q_proj;
    LlamaDenseWeight q_a_proj;
    LlamaDenseWeight q_b_proj;
    LlamaDenseWeight kv_a_proj;
    LlamaDenseWeight kv_b_proj;

    Tensor q_a_layernorm;
    Tensor kv_a_layernorm;

    int window_size{};
};

struct LlamaFfnWeight: core::Module {

    LlamaFfnWeight() = default;

    LlamaFfnWeight(int            hidden_dim,
                   int            inter_size,
                   bool           bias,
                   int            tp_size,
                   int            tp_rank,
                   DataType       data_type,
                   DataType       weight_type,
                   int            group_size,
                   ActivationType act_type,
                   bool           fuse_silu_act);

    static constexpr bool fuse_up_and_gate = true;

    void prepare(bool fused_moe);

    LlamaDenseWeight gating;
    LlamaDenseWeight intermediate;
    LlamaDenseWeight output;
    LlamaDenseWeight fused_gating_intermediate;

    ActivationType act_type;

    int  inter_size{};
    bool is_fused_silu{};

    int tp_rank{};
};

struct MoeFfnWeight: core::Module {

    MoeFfnWeight() = default;

    MoeFfnWeight(int             layer_id,
                 const MoeParam& param,
                 int             hidden_dim,
                 bool            mlp_bias,
                 DataType        data_type,
                 DataType        weight_type,
                 int             group_size,
                 int             tp_size,
                 int             tp_rank,
                 ActivationType  act_type,
                 bool            fuse_silu_act);

    void prepare();

    LlamaDenseWeight gate;
    LlamaDenseWeight shared_gate;

    std::vector<std::unique_ptr<LlamaFfnWeight>> experts;

    // reference into `experts`
    LlamaFfnWeight block;

    MoeParam::Method method{};
};

void LinkExperts(std::function<LlamaDenseWeight*(int)> experts, int n, LlamaDenseWeight& d);

}  // namespace turbomind
