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

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h

#pragma once

#include "src/turbomind/core/core.h"

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

struct LlamaDecoderLayerWeight: core::Module {
public:
    LlamaDecoderLayerWeight() = delete;

    LlamaDecoderLayerWeight(DataType           data_type,
                            int                layer_id,
                            const ModelParam&  model,
                            const EngineParam& engine,
                            const MoeParam&    moe_param);

    ~LlamaDecoderLayerWeight();
    LlamaDecoderLayerWeight(const LlamaDecoderLayerWeight&) = delete;
    LlamaDecoderLayerWeight& operator=(const LlamaDecoderLayerWeight&) = delete;

    void prepare(const cudaDeviceProp& prop, cudaStream_t st);

    Tensor self_attn_norm;
    Tensor ffn_norm;

    std::unique_ptr<LlamaAttentionWeight> self_attn_weights;

    std::unique_ptr<LlamaFfnWeight> ffn_weights;
    std::unique_ptr<MoeFfnWeight>   moe_weights;

private:
    int head_num_;
    int kv_head_num_;
    int size_per_head_;
    int hidden_units_;
    int inter_size_;

    DataType data_type_;
    DataType weight_type_;
    DataType expert_weight_type_;

    int  bit_size_;
    bool attn_bias_;
    int  attn_tp_size_;
    int  attn_tp_rank_;
    int  mlp_tp_size_;
    int  mlp_tp_rank_;
};

}  // namespace turbomind
