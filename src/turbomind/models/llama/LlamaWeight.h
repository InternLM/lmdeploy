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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGptWeight.h

#pragma once

#include <unordered_map>

#include "src/turbomind/core/context.h"
#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

struct LlamaWeight: core::Module {
    LlamaWeight() = default;

    LlamaWeight(DataType           data_type,
                const ModelParam&  model_param,
                const EngineParam& engine_param,
                const LoraParam&   lora_param,
                const MoeParam&    moe_param);

    ~LlamaWeight();

    LlamaWeight(const LlamaWeight&) = delete;
    LlamaWeight& operator=(const LlamaWeight&) = delete;

    void prepare(const cudaDeviceProp& prop);

    bool is_initialized() const;

    void initialize();

    void release();

    void to_device(const core::Device& device);

    core::ContextGuard context() const;

    std::vector<LlamaDecoderLayerWeight*> decoder_layer_weights;

    LlamaDenseWeight pre_decoder_embedding;
    LlamaDenseWeight post_decoder_embedding;

    Tensor output_norm_weight;

private:
    const ModelParam  model_param_;
    const EngineParam engine_param_;
    const LoraParam   lora_param_;
    const MoeParam    moe_param_;

    int hidden_units_;
    int vocab_size_;
    int vocab_size_padded_;
    int embedding_size_;
    int num_layer_;

    DataType data_type_;
    DataType weight_type_;

    std::unordered_map<std::string, Tensor> pinned_weights_;

    int tp_size_;  // this will follow attn tp param
    int tp_rank_;

    std::vector<int> inter_size_;

    core::Stream    stream_;
    core::Allocator alloca_;
    bool            initialized_{false};
};

}  // namespace turbomind
