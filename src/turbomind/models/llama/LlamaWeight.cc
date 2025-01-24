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
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"
#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
LlamaWeight<T>::LlamaWeight(
    const ModelParam& model, const LoraParam& lora_param, const MoeParam& moe_param, size_t tp_size, size_t tp_rank):
    hidden_units_(model.hidden_units),
    inter_size_(model.inter_size),
    vocab_size_(model.vocab_size),
    vocab_size_padded_(model.vocab_size),
    embedding_size_(model.embedding_size),
    num_layer_(model.layer_num),
    weight_type_(model.weight_type),
    tensor_para_size_(tp_size),
    tensor_para_rank_(tp_rank)
{
    if (vocab_size_padded_ % tensor_para_size_ != 0) {
        vocab_size_padded_ = (vocab_size_ + tensor_para_size_ - 1) / tensor_para_size_ * tensor_para_size_;
        TM_LOG_WARNING("pad vocab size from %d to %d", vocab_size_, vocab_size_padded_);
    }
    if (embedding_size_ % tensor_para_size_ != 0) {
        embedding_size_ = (embedding_size_ + tensor_para_size_ - 1) / tensor_para_size_ * tensor_para_size_;
        TM_LOG_WARNING("pad embed size from %d to %d", embedding_size_, embedding_size_);
    }
    FT_CHECK(hidden_units_ % tensor_para_size_ == 0);

    check_cuda_error(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

    decoder_layer_weights.reserve(num_layer_);
    for (unsigned l = 0; l < num_layer_; ++l) {
        decoder_layer_weights.emplace_back(
            new LlamaDecoderLayerWeight<T>(l, model, lora_param, moe_param, tp_size, tp_rank));
        decoder_layer_weights.back()->malloc(stream_);
    }

    FT_CHECK(vocab_size_padded_ % tensor_para_size_ == 0);
    deviceMalloc((T**)&pre_decoder_embedding_table, embedding_size_ * hidden_units_ / tensor_para_size_, stream_);
    deviceMalloc((T**)&output_norm_weight, hidden_units_, stream_);
    deviceMalloc((T**)&post_decoder_embedding_kernel, hidden_units_ * vocab_size_padded_ / tensor_para_size_, stream_);

    // Wait for allocations
    check_cuda_error(cudaStreamSynchronize(stream_));
}

template<typename T>
LlamaWeight<T>::~LlamaWeight()
{
    deviceFree(pre_decoder_embedding_table, stream_);
    deviceFree(output_norm_weight, stream_);
    deviceFree(post_decoder_embedding_kernel, stream_);

    for (auto& p : decoder_layer_weights) {
        p->free(stream_);
        delete p;
    }

    decoder_layer_weights.clear();

    // Wait for deallocations
    check_cuda_error(cudaStreamSynchronize(stream_));
    check_cuda_error(cudaStreamDestroy(stream_));
    stream_ = {};
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
                      {embedding_size_ * hidden_units_ / tensor_para_size_},
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
                         {embedding_size_ * hidden_units_ / tensor_para_size_ * sizeof(T)},
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
    const auto workspace_size = [&] {
        size_t size{};
        for (const auto& layer : decoder_layer_weights) {
            size = std::max(size, layer->workspace_size());
        }
        return size;
    }();

    char* workspace{};

    TM_LOG_INFO("[LlamaWeight<T>::prepare] workspace size: %d\n", workspace_size);

    // Wait for the weights to be filled externally
    check_cuda_error(cudaDeviceSynchronize());

    if (workspace_size) {
        deviceMalloc((char**)&workspace, workspace_size, stream_);
    }
    for (auto& layer : decoder_layer_weights) {
        layer->prepare(workspace, workspace_size, prop, stream_);
    }

    deviceFree(workspace, stream_);

    check_cuda_error(cudaStreamSynchronize(stream_));
}

#ifdef ENABLE_FP32
template struct LlamaWeight<float>;
#endif
template struct LlamaWeight<half>;
#ifdef ENABLE_BF16
template struct LlamaWeight<__nv_bfloat16>;
#endif

}  // namespace turbomind
