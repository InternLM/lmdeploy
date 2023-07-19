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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGptWeight.h

#pragma once

#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<typename T>
struct LlamaWeight {
    LlamaWeight() = default;
    LlamaWeight(size_t     head_num,
                size_t     kv_head_num,
                size_t     size_per_head,
                size_t     inter_size,
                size_t     vocab_size,
                size_t     num_layer,
                WeightType weight_type,
                bool       attn_bias,
                size_t     tensor_para_size,
                size_t     tensor_para_rank,
                int        prefix_cache_len);

    ~LlamaWeight();

    LlamaWeight(const LlamaWeight& other)            = delete;
    LlamaWeight& operator=(const LlamaWeight& other) = delete;

    void loadModel(std::string dir_path);

    std::vector<LlamaDecoderLayerWeight<T>*> decoder_layer_weights;
    const T*                                 pre_decoder_embedding_table{};
    const T*                                 output_norm_weight{};
    const T*                                 post_decoder_embedding_kernel{};

    size_t prefix_cache_len_;
    int*   prefix_cache_token{};
    T*     prefix_cache_key{};
    T*     prefix_cache_value{};

private:
    void mallocWeights();

    size_t     hidden_units_;
    size_t     inter_size_;
    size_t     vocab_size_;
    size_t     num_layer_;
    WeightType weight_type_;
    size_t     tensor_para_size_;
    size_t     tensor_para_rank_;
};

}  // namespace turbomind
