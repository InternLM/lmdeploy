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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGptWeight.cc

#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"
#include <cuda_runtime.h>

namespace turbomind {

LlamaWeight::LlamaWeight(DataType           data_type,
                         const ModelParam&  model,
                         const EngineParam& engine_param,
                         const LoraParam&   lora_param,
                         const MoeParam&    moe_param):
    hidden_units_(model.hidden_units),
    inter_size_(model.inter_size),
    vocab_size_(model.vocab_size),
    vocab_size_padded_(model.vocab_size),
    embedding_size_(model.embedding_size),
    num_layer_(model.layer_num),
    data_type_{data_type},
    weight_type_{model.weight_type},
    tp_size_(engine_param.attn_tp_size),
    tp_rank_(engine_param.attn_tp_rank)
{
    if (vocab_size_padded_ % tp_size_ != 0) {
        vocab_size_padded_ = (vocab_size_ + tp_size_ - 1) / tp_size_ * tp_size_;
        TM_LOG_WARNING("pad vocab size from %d to %d", vocab_size_, vocab_size_padded_);
    }
    if (embedding_size_ % tp_size_ != 0) {
        embedding_size_ = (embedding_size_ + tp_size_ - 1) / tp_size_ * tp_size_;
        TM_LOG_WARNING("pad embed size from %d to %d", embedding_size_, embedding_size_);
    }
    FT_CHECK(hidden_units_ % tp_size_ == 0);

    stream_ = core::Stream::create();
    alloca_ = core::Allocator{stream_, false};

    core::ContextGuard guard = context();

    decoder_layer_weights.reserve(num_layer_);
    for (unsigned l = 0; l < num_layer_; ++l) {
        decoder_layer_weights.emplace_back(
            new LlamaDecoderLayerWeight(data_type, l, model, engine_param, lora_param, moe_param));
        decoder_layer_weights.back()->malloc();
    }

    TM_CHECK_EQ(vocab_size_padded_ % tp_size_, 0);
    TM_CHECK_EQ(hidden_units_ % tp_size_, 0);

    pre_decoder_embedding  = core::Tensor{{embedding_size_, hidden_units_ / tp_size_}, data_type_, MEMORY_GPU};
    post_decoder_embedding = core::Tensor{{hidden_units_, vocab_size_padded_ / tp_size_}, data_type_, MEMORY_GPU};

    output_norm_weight = core::Buffer{hidden_units_, data_type_, MEMORY_GPU};
}

LlamaWeight::~LlamaWeight()
{
    core::ContextGuard guard = context();

    pre_decoder_embedding  = {};
    post_decoder_embedding = {};
    output_norm_weight     = {};

    for (auto& p : decoder_layer_weights) {
        p->free();
        delete p;
    }

    decoder_layer_weights.clear();

    // Wait for deallocations
    core::Context::stream().Sync();
}

core::ContextGuard LlamaWeight::context() const
{
    return core::ContextGuard{stream_, alloca_};
}

core::TensorMap LlamaWeight::getParams()
{
    core::TensorMap output;

    output.emplace("tok_embeddings." + std::to_string(tp_rank_) + ".weight", pre_decoder_embedding);
    output.emplace("output." + std::to_string(tp_rank_) + ".weight", post_decoder_embedding);

    output.emplace("norm.weight", output_norm_weight);

    // transformer layers
    for (size_t i = 0; i < num_layer_; i++) {
        std::string     prefix = fmtstr("layers.%d", i);
        core::TensorMap layer  = decoder_layer_weights[i]->getParams(prefix);
        for (auto& kv : layer) {
            output.insert(std::move(kv));
        }
    }

    return output;
}

void LlamaWeight::prepare(const cudaDeviceProp& prop)
{
    core::ContextGuard guard = context();

    const auto workspace_size = [&] {
        size_t size{};
        for (const auto& layer : decoder_layer_weights) {
            size = std::max(size, layer->workspace_size());
        }
        return size;
    }();

    TM_LOG_INFO("[LlamaWeight<T>::prepare] workspace size: %d", workspace_size);

    // Wait for the weights to be filled externally
    check_cuda_error(cudaDeviceSynchronize());

    auto stream = core::Context::stream().handle();

    core::Buffer_<char> workspace;

    if (workspace_size) {
        workspace = core::Buffer_<char>(workspace_size, MEMORY_GPU);
    }
    for (auto& layer : decoder_layer_weights) {
        layer->prepare(workspace.data(), workspace_size, prop, stream);
    }
}

}  // namespace turbomind
