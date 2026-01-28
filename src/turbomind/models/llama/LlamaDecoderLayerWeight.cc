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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.cc

#include <cstdlib>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

static bool is_fuse_silu_act()
{
    static const bool value = [] {
        const auto str = std::getenv("TM_FUSE_SILU_ACT");
        if (str) {
            try {
                auto v = std::stoi(str) != 0;
                TM_LOG_INFO("TM_FUSE_SILU_ACT=%d", (int)v);
                return v;
            }
            catch (...) {
            }
        }
        // TM_LOG_INFO("TM_FUSE_SILU_ACT=1");
        return true;
    }();
    return value;
}

LlamaDecoderLayerWeight::LlamaDecoderLayerWeight(
    DataType data_type, int layer_id, const ModelParam& model, const EngineParam& engine, const MoeParam& moe_param):
    head_num_(model.head_num),
    kv_head_num_(model.kv_head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    inter_size_(model.inter_size.at(layer_id)),
    data_type_{data_type},
    weight_type_(model.weight_type),
    expert_weight_type_(model.expert_weight_type),
    attn_bias_(model.attn_bias),
    attn_tp_size_(engine.attn_tp_size),
    attn_tp_rank_(engine.attn_tp_rank),
    mlp_tp_size_(engine.mlp_tp_size),
    mlp_tp_rank_(engine.mlp_tp_rank)
{
    self_attn_weights.reset(new LlamaAttentionWeight{hidden_units_,
                                                     size_per_head_,
                                                     head_num_,
                                                     kv_head_num_,
                                                     model.mla,
                                                     attn_bias_,
                                                     model.qk_norm,
                                                     attn_tp_size_,
                                                     attn_tp_rank_,
                                                     data_type_,
                                                     weight_type_,
                                                     model.group_size,
                                                     model.window_size.empty() ? 0 : model.window_size.at(layer_id),
                                                     model.attn_sink});
    register_module("attention", *self_attn_weights);

    if (inter_size_) {
        const bool is_cublas_gemm = byte_size(weight_type_, 8) == 16;
        ffn_weights.reset(new LlamaFfnWeight{
            hidden_units_,
            inter_size_,
            model.mlp_bias,
            mlp_tp_size_,
            mlp_tp_rank_,
            data_type_,
            weight_type_,
            model.group_size,
            model.act_type,
            is_fuse_silu_act() && !is_cublas_gemm,
        });
        register_module("feed_forward", *ffn_weights);
    }

    if (layer_id < moe_param.expert_num.size() && moe_param.expert_num[layer_id]) {
        moe_weights.reset(new MoeFfnWeight{layer_id,
                                           moe_param,
                                           hidden_units_,
                                           model.mlp_bias,
                                           data_type_,
                                           expert_weight_type_,
                                           model.group_size,
                                           mlp_tp_size_,
                                           mlp_tp_rank_,
                                           model.act_type,
                                           is_fuse_silu_act()});
        register_module("moe_ffn", *moe_weights);
    }

    self_attn_norm = Tensor{{hidden_units_}, data_type_, kDEVICE};
    ffn_norm       = Tensor{{hidden_units_}, data_type_, kDEVICE};
    register_parameter("attention_norm.weight", self_attn_norm);
    register_parameter("ffn_norm.weight", ffn_norm);
}

LlamaDecoderLayerWeight::~LlamaDecoderLayerWeight() = default;

void LlamaDecoderLayerWeight::prepare(const cudaDeviceProp& prop, cudaStream_t st)
{
    self_attn_weights->prepare();

    if (ffn_weights) {
        ffn_weights->prepare(false);
    }

    if (moe_weights) {
        moe_weights->prepare();
    }
}

}  // namespace turbomind
