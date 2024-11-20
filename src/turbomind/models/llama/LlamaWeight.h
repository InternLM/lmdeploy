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

#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

template<typename T>
struct LlamaWeight {
    LlamaWeight() = default;

    LlamaWeight(const ModelParam& model_param,
                const LoraParam&  lora_param,
                const MoeParam&   moe_param,
                size_t            tp_size,
                size_t            tp_rank);

    ~LlamaWeight();

    LlamaWeight(const LlamaWeight& other) = delete;
    LlamaWeight& operator=(const LlamaWeight& other) = delete;

    void loadModel(std::string dir_path);

    TensorMap getParams();

    void prepare(const cudaDeviceProp& prop);

    std::vector<LlamaDecoderLayerWeight<T>*> decoder_layer_weights;

    T* pre_decoder_embedding_table{};
    T* output_norm_weight{};
    T* post_decoder_embedding_kernel{};

private:
    size_t     hidden_units_;
    size_t     vocab_size_;
    size_t     vocab_size_padded_;
    size_t     embedding_size_;
    size_t     num_layer_;
    WeightType weight_type_;
    size_t     tensor_para_size_;
    size_t     tensor_para_rank_;

    std::vector<int> inter_size_;

    cudaStream_t stream_;
};

}  // namespace turbomind
