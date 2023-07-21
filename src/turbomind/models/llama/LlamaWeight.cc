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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGptWeight.cc

#include "src/turbomind/models/llama/LlamaWeight.h"

namespace turbomind {

template<typename T>
LlamaWeight<T>::LlamaWeight(size_t     head_num,
                            size_t     kv_head_num,
                            size_t     size_per_head,
                            size_t     inter_size,
                            size_t     vocab_size,
                            size_t     num_layer,
                            WeightType weight_type,
                            bool       attn_bias,
                            size_t     tensor_para_size,
                            size_t     tensor_para_rank,
                            int        prefix_cache_len):
    hidden_units_(head_num * size_per_head),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    weight_type_(weight_type),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    prefix_cache_len_(prefix_cache_len)
{
    decoder_layer_weights.reserve(num_layer_);
    for (unsigned l = 0; l < num_layer_; ++l) {
        decoder_layer_weights.push_back(new LlamaDecoderLayerWeight<T>(head_num,
                                                                       kv_head_num,
                                                                       size_per_head,
                                                                       inter_size_,
                                                                       weight_type_,
                                                                       attn_bias,
                                                                       tensor_para_size_,
                                                                       tensor_para_rank_));
    }

    mallocWeights();
}

template<typename T>
LlamaWeight<T>::~LlamaWeight()
{
    cudaFree((void*)pre_decoder_embedding_table);
    cudaFree((void*)output_norm_weight);
    cudaFree((void*)post_decoder_embedding_kernel);

    if (prefix_cache_key) {
        cudaFree((void*)prefix_cache_key);
        cudaFree((void*)prefix_cache_token);
    }

    pre_decoder_embedding_table   = nullptr;
    post_decoder_embedding_kernel = nullptr;

    prefix_cache_token = nullptr;
    prefix_cache_key   = nullptr;
    prefix_cache_value = nullptr;
}

template<typename T>
void LlamaWeight<T>::mallocWeights()
{
    deviceMalloc((T**)&pre_decoder_embedding_table, vocab_size_ * hidden_units_);
    deviceMalloc((T**)&output_norm_weight, hidden_units_);
    deviceMalloc((T**)&post_decoder_embedding_kernel, hidden_units_ * vocab_size_);

    if (prefix_cache_len_) {
        size_t cache_size = num_layer_ * prefix_cache_len_ * hidden_units_ / tensor_para_size_;
        deviceMalloc((T**)&prefix_cache_key, cache_size * 2);
        prefix_cache_value = prefix_cache_key + cache_size;
        deviceMalloc((int**)&prefix_cache_token, prefix_cache_len_);
    }
}

template<typename T>
void LlamaWeight<T>::loadModel(std::string dir_path)
{
    FtCudaDataType model_file_type = FtCudaDataType::FP16;
    dir_path += '/';

    loadWeightFromBin((T*)pre_decoder_embedding_table,
                      {vocab_size_ * hidden_units_},
                      dir_path + "tok_embeddings.weight",
                      model_file_type);

    loadWeightFromBin((T*)output_norm_weight, {hidden_units_}, dir_path + "norm.weight", model_file_type);

    loadWeightFromBin(
        (T*)post_decoder_embedding_kernel, {hidden_units_ * vocab_size_}, dir_path + "output.weight", model_file_type);

    if (prefix_cache_len_) {
        loadWeightFromBin((float*)prefix_cache_token, {prefix_cache_len_}, dir_path + "prefix_cache.token");
        loadWeightFromBin((T*)prefix_cache_key,
                          {num_layer_ * prefix_cache_len_, hidden_units_ / tensor_para_size_},
                          dir_path + "prefix_cache." + std::to_string(tensor_para_rank_) + ".key",
                          model_file_type);
        loadWeightFromBin((T*)prefix_cache_value,
                          {num_layer_ * prefix_cache_len_, hidden_units_ / tensor_para_size_},
                          dir_path + "prefix_cache." + std::to_string(tensor_para_rank_) + ".value",
                          model_file_type);
    }

    for (unsigned layer = 0; layer < num_layer_; ++layer) {
        decoder_layer_weights[layer]->loadModel(dir_path + "layers." + std::to_string(layer), model_file_type);
    }
}

template struct LlamaWeight<float>;
template struct LlamaWeight<half>;

}  // namespace turbomind
