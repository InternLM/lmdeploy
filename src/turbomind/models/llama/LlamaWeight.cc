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
#include "src/turbomind/utils/memory_utils.h"
#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
LlamaWeight<T>::LlamaWeight(size_t     head_num,
                            size_t     kv_head_num,
                            size_t     size_per_head,
                            size_t     hidden_units,
                            size_t     inter_size,
                            size_t     vocab_size,
                            size_t     num_layer,
                            bool       attn_bias,
                            WeightType weight_type,
                            int        group_size,
                            LoraParam  lora_param,
                            size_t     tensor_para_size,
                            size_t     tensor_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    vocab_size_padded_(vocab_size),
    num_layer_(num_layer),
    weight_type_(weight_type),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank)
{
    if (vocab_size_padded_ % tensor_para_size_ != 0) {
        vocab_size_padded_ = (vocab_size_padded_ + tensor_para_size_ - 1) / tensor_para_size_ * tensor_para_size_;
        TM_LOG_WARNING("pad vocab size from %d to %d", vocab_size_, vocab_size_padded_);
    }

    FT_CHECK(hidden_units_ % tensor_para_size_ == 0);

    decoder_layer_weights.reserve(num_layer_);
    for (unsigned l = 0; l < num_layer_; ++l) {
        decoder_layer_weights.push_back(new LlamaDecoderLayerWeight<T>(l,
                                                                       head_num,
                                                                       kv_head_num,
                                                                       size_per_head,
                                                                       hidden_units_,
                                                                       inter_size_,
                                                                       weight_type_,
                                                                       group_size,
                                                                       lora_param,
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

    pre_decoder_embedding_table   = nullptr;
    output_norm_weight            = nullptr;
    post_decoder_embedding_kernel = nullptr;

    for (auto& p : decoder_layer_weights) {
        delete p;
    }
}

template<typename T>
void LlamaWeight<T>::mallocWeights()
{
    FT_CHECK(vocab_size_padded_ % tensor_para_size_ == 0);
    deviceMalloc((T**)&pre_decoder_embedding_table, vocab_size_padded_ * hidden_units_ / tensor_para_size_);
    deviceMalloc((T**)&output_norm_weight, hidden_units_);
    deviceMalloc((T**)&post_decoder_embedding_kernel, hidden_units_ * vocab_size_padded_ / tensor_para_size_);
}

template<typename T>
void LlamaWeight<T>::loadModel(std::string dir_path)
{
    FtCudaDataType model_file_type = FtCudaDataType::FP16;
    if (weight_type_ == WeightType::kBF16) {
        model_file_type = FtCudaDataType::BF16;
    }
    dir_path += '/';

    loadWeightFromBin((T*)pre_decoder_embedding_table,
                      {vocab_size_padded_ * hidden_units_ / tensor_para_size_},
                      dir_path + "tok_embeddings." + std::to_string(tensor_para_rank_) + ".weight",
                      model_file_type);

    loadWeightFromBin((T*)output_norm_weight, {hidden_units_}, dir_path + "norm.weight", model_file_type);

    loadWeightFromBin((T*)post_decoder_embedding_kernel,
                      {hidden_units_ * vocab_size_padded_ / tensor_para_size_},
                      dir_path + "output." + std::to_string(tensor_para_rank_) + ".weight",
                      model_file_type);

    for (unsigned layer = 0; layer < num_layer_; ++layer) {
        decoder_layer_weights[layer]->loadModel(dir_path + "layers." + std::to_string(layer), model_file_type);
    }
}

template<typename T>
TensorMap LlamaWeight<T>::getParams()
{
    TensorMap output;

    output.insert("tok_embeddings." + std::to_string(tensor_para_rank_) + ".weight",
                  Tensor{MEMORY_GPU,
                         getTensorType<T>(),
                         {vocab_size_padded_ * hidden_units_ / tensor_para_size_ * sizeof(T)},
                         pre_decoder_embedding_table});

    output.insert("norm.weight",
                  Tensor{MEMORY_GPU, getTensorType<T>(), {hidden_units_ * sizeof(T)}, output_norm_weight});

    output.insert("output." + std::to_string(tensor_para_rank_) + ".weight",
                  Tensor{MEMORY_GPU,
                         getTensorType<T>(),
                         {hidden_units_ * vocab_size_padded_ * sizeof(T) / tensor_para_size_},
                         post_decoder_embedding_kernel});

    // transformer layers
    for (size_t i = 0; i < num_layer_; i++) {
        std::string prefix = fmtstr("layers.%d", i);
        TensorMap   layeri = decoder_layer_weights[i]->getParams(prefix);
        for (auto [name, tensor] : layeri) {
            output.insert(name, tensor);
        }
    }

    return output;
}

template<typename T>
void LlamaWeight<T>::prepare(const cudaDeviceProp& prop)
{
    const auto workspace_size = decoder_layer_weights[0]->workspace_size();
    char*      workspace{};

    TM_LOG_INFO("[LlamaWeight<T>::prepare] workspace size: %d\n", workspace_size);

    if (workspace_size) {
        deviceMalloc((char**)&workspace, workspace_size);
    }
    for (auto& layer : decoder_layer_weights) {
        layer->prepare(workspace, workspace_size, prop);
    }
    deviceFree(workspace);
}

#ifdef ENABLE_FP32
template struct LlamaWeight<float>;
#endif
template struct LlamaWeight<half>;
#ifdef ENABLE_BF16
template struct LlamaWeight<__nv_bfloat16>;
#endif

}  // namespace turbomind
