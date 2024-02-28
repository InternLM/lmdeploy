// Copyright (c) OpenMMLab. All rights reserved.
// Yineng Zhang <me@zhyncs.com>
// Zhiwei Bao <zwbao@foxmail.com>

#include "src/turbomind/models/medusa_plugin/medusa_weight.h"
#include "src/turbomind/utils/memory_utils.h"
#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>

namespace turbomind {

template<typename T>
MedusaWeight<T>::MedusaWeight(
    size_t medusa_num_heads, size_t medusa_num_layers, size_t hidden_size, size_t vocab_size, WeightType weight_type):
    medusa_num_heads_(medusa_num_heads),
    medusa_num_layers_(medusa_num_layers),
    hidden_size_(hidden_size),
    vocab_size_(vocab_size),
    weight_type_(weight_type)
{
    heads_weights_.resize(medusa_num_heads_);
    std::fill_n(heads_weights_.begin(),
                medusa_num_heads_,
                LlamaDenseWeight<T>{hidden_size_, vocab_size_, nullptr, weight_type_, nullptr, nullptr, 0});

    resblocks_weights_.resize(medusa_num_heads_);
    std::fill_n(resblocks_weights_.begin(), medusa_num_heads_, std::vector<LlamaDenseWeight<T>>(medusa_num_layers_));
    std::for_each(resblocks_weights_.begin(), resblocks_weights_.end(), [this](auto& resblock_weights) {
        std::for_each(resblock_weights.begin(), resblock_weights.end(), [this](auto& resblock_weight) {
            resblock_weight.input_dims  = hidden_size_;
            resblock_weight.output_dims = hidden_size_;
            resblock_weight.type        = weight_type_;
        });
    });

    malloc_weight();
}

template<typename T>
MedusaWeight<T>::~MedusaWeight()
{
    free_weight();
}

template<typename T>
void MedusaWeight<T>::malloc_weight(LlamaDenseWeight<T>* weight, bool bias)
{
    if (bias) {
        deviceMalloc((T**)&weight->bias, weight->output_dims);
    }
    const size_t bit_size = getBitSize(weight->type);
    if (bit_size >= 16) {
        deviceMalloc((T**)&weight->kernel, weight->input_dims * weight->output_dims);
    }
}

template<typename T>
void MedusaWeight<T>::malloc_weight()
{
    std::for_each(heads_weights_.begin(), heads_weights_.end(), [this](auto& head_weights) {
        malloc_weight(&head_weights, false);
    });
    std::for_each(resblocks_weights_.begin(), resblocks_weights_.end(), [this](auto& resblock_weights) {
        std::for_each(resblock_weights.begin(), resblock_weights.end(), [this](auto& resblock_weight) {
            malloc_weight(&resblock_weight, true);
        });
    });
}

template<typename T>
void MedusaWeight<T>::free_weight(LlamaDenseWeight<T>* weight)
{
    cudaFree(weight->kernel);
    cudaFree(weight->bias);
    cudaFree(weight->scales_and_zeros);

    weight->kernel           = nullptr;
    weight->bias             = nullptr;
    weight->scales_and_zeros = nullptr;
}

template<typename T>
void MedusaWeight<T>::free_weight()
{
    std::for_each(
        heads_weights_.begin(), heads_weights_.end(), [this](auto& head_weights) { free_weight(&head_weights); });
    std::for_each(resblocks_weights_.begin(), resblocks_weights_.end(), [this](auto& resblock_weights) {
        std::for_each(resblock_weights.begin(), resblock_weights.end(), [this](auto& resblock_weight) {
            free_weight(&resblock_weight);
        });
    });
}

template<typename T>
void MedusaWeight<T>::load_weight(LlamaDenseWeight<T>* weight, const std::string& path, FtCudaDataType model_file_type)
{
    // TODO support quant
    const size_t bit_size = getBitSize(weight->type);
    if (bit_size >= 16) {
        loadWeightFromBin((T*)weight->kernel, {weight->input_dims, weight->output_dims}, path, model_file_type);
    }
}

template<typename T>
void MedusaWeight<T>::load_bias(LlamaDenseWeight<T>* weight, const std::string& path, FtCudaDataType model_file_type)
{
    // TODO support quant
    const size_t bit_size = getBitSize(weight->type);
    if (bit_size >= 16) {
        loadWeightFromBin((T*)weight->bias, {weight->output_dims}, path, model_file_type);
    }
}

template<typename T>
void MedusaWeight<T>::load_model(const std::string& dir_path, FtCudaDataType model_file_type)
{
    auto ends_with = [](std::string& text, const std::string& suffix) noexcept {
        return suffix.empty()
               || (text.size() >= suffix.size()
                   && std::memcmp(text.data() + (text.size() - suffix.size()), suffix.data(), suffix.size()) == 0);
    };
    std::string weight_path = dir_path;
    if (!ends_with(weight_path, "/")) {
        weight_path.append("/");
    }
    std::string prefix = "medusa.";
    // TODO support TP
    std::string rank = "0.";
    weight_path.append(prefix);
    for (int i = 0; i < medusa_num_heads_; i++) {
        for (int j = 0; j < medusa_num_layers_; j++) {
            std::stringstream ss;
            ss << weight_path << i << "." << j << "."
               << "linear." << rank;
            std::string common_prefix = ss.str();

            load_weight(&resblocks_weights_[i][j], common_prefix + "weight", model_file_type);
            load_bias(&resblocks_weights_[i][j], common_prefix + "bias", model_file_type);
        }

        std::stringstream ss;
        ss << weight_path << i << "." << medusa_num_layers_ << "." << rank << "weight";
        load_weight(&heads_weights_[i], ss.str(), model_file_type);
    }
}

template<typename T>
const std::vector<LlamaDenseWeight<T>>& MedusaWeight<T>::get_heads_weights() const
{
    return heads_weights_;
}

template<typename T>
const std::vector<std::vector<LlamaDenseWeight<T>>>& MedusaWeight<T>::get_resblocks_weights() const
{
    return resblocks_weights_;
}

template struct MedusaWeight<float>;
template struct MedusaWeight<half>;
#ifdef ENABLE_BF16
template struct MedusaWeight<__nv_bfloat16>;
#endif

}  // namespace turbomind
