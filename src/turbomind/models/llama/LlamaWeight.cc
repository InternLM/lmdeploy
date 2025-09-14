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

#include <cuda_runtime.h>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

LlamaWeight::LlamaWeight(DataType           data_type,
                         const ModelParam&  model,
                         const EngineParam& engine_param,
                         const LoraParam&   lora_param,
                         const MoeParam&    moe_param):
    model_param_{model},
    engine_param_{engine_param},
    lora_param_{lora_param},
    moe_param_{moe_param},
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
    TM_CHECK_EQ(vocab_size_padded_ % tp_size_, 0);
    TM_CHECK_EQ(hidden_units_ % tp_size_, 0);

    stream_ = core::Stream::create();
    alloca_ = core::Allocator{stream_, false};

    initialize();
}

LlamaWeight::~LlamaWeight()
{
    release();
}

bool LlamaWeight::is_initialized() const
{
    return initialized_;
}

void LlamaWeight::initialize()
{
    core::ContextGuard guard = context();

    pre_decoder_embedding.emplace(embedding_size_, hidden_units_ / tp_size_, data_type_, false, data_type_, 1);
    post_decoder_embedding.emplace(hidden_units_, vocab_size_padded_ / tp_size_, data_type_, false, data_type_, 1);
    register_module("tok_embeddings", pre_decoder_embedding, tp_rank_);
    register_module("output", post_decoder_embedding, tp_rank_);

    /// Lower VRAM pressure on consumer grade GPUs
    /// TODO: Support token embeds on pinned host memory
    pre_decoder_embedding.weight  = empty_like(pre_decoder_embedding.weight, kCPU);
    post_decoder_embedding.weight = empty_like(post_decoder_embedding.weight, kCPU);

    decoder_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; ++i) {
        decoder_layer_weights.emplace_back(
            new LlamaDecoderLayerWeight(data_type_, i, model_param_, engine_param_, lora_param_, moe_param_));
        register_module("layers", *decoder_layer_weights.back(), i);
    }

    output_norm_weight = Tensor{{hidden_units_}, data_type_, kDEVICE};
    register_parameter("norm.weight", output_norm_weight);
    initialized_ = true;
}

void LlamaWeight::release()
{
    core::ContextGuard guard = context();

    pre_decoder_embedding  = {};
    post_decoder_embedding = {};
    output_norm_weight     = {};

    for (auto& p : decoder_layer_weights) {
        delete p;
    }

    decoder_layer_weights.clear();

    // Wait for deallocations
    core::Context::stream().Sync();

    // release memory back to os
    core::Context::device_alloc()->trim(0);
    initialized_ = false;
}

void LlamaWeight::to_device(const core::Device& device)
{
    core::ContextGuard guard = context();

    auto tensor_ptr_map = get_parameters();
    for (auto& [name, tensor_ptr] : tensor_ptr_map) {
        *tensor_ptr = core::to_device(*tensor_ptr, device);
    }
    core::Context::stream().Sync();
    if (device.type == kCPU) {
        core::Context::device_alloc()->trim(0);
    }
}

core::ContextGuard LlamaWeight::context() const
{
    return core::ContextGuard{stream_, alloca_};
}

void LlamaWeight::prepare(const cudaDeviceProp& prop)
{
    core::ContextGuard guard = context();

    // Wait for the weights to be filled externally
    check_cuda_error(cudaDeviceSynchronize());

    auto to_device = [](Tensor& x) {
        auto tmp = x;
        x        = empty_like(tmp, kDEVICE);
        Copy(tmp, x);
    };

    to_device(pre_decoder_embedding.weight);
    to_device(post_decoder_embedding.weight);

    auto stream = core::Context::stream().handle();

    post_decoder_embedding.prepare(false, false);

    for (auto& layer : decoder_layer_weights) {
        layer->prepare(prop, stream);
    }

    // Block until processing is done
    check_cuda_error(cudaStreamSynchronize(stream));
}

}  // namespace turbomind
