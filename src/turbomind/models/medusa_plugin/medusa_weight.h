// Copyright (c) OpenMMLab. All rights reserved.
// Yineng Zhang <me@zhyncs.com>
// Zhiwei Bao <zwbao@foxmail.com>

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <vector>

namespace turbomind {

template<typename T>
class MedusaWeight {
public:
    MedusaWeight(size_t     medusa_num_heads,
                 size_t     medusa_num_layers,
                 size_t     hidden_size,
                 size_t     vocab_size,
                 WeightType weight_type,
                 size_t     tensor_para_size,
                 size_t     tensor_para_rank);
    ~MedusaWeight();
    MedusaWeight(const MedusaWeight&) = delete;
    MedusaWeight& operator=(const MedusaWeight&) = delete;

    const std::vector<LlamaDenseWeight<T>>&              get_heads_weights() const;
    const std::vector<std::vector<LlamaDenseWeight<T>>>& get_resblocks_weights() const;

    void load_model(const std::string& dir_path, FtCudaDataType model_file_type);

private:
    void malloc_weight(LlamaDenseWeight<T>* weight, bool bias);
    void free_weight(LlamaDenseWeight<T>* weight);
    void malloc_weight();
    void free_weight();
    void load_weight(LlamaDenseWeight<T>* weight, const std::string& path, FtCudaDataType model_file_type);
    void load_bias(LlamaDenseWeight<T>* weight, const std::string& path, FtCudaDataType model_file_type);

private:
    size_t     medusa_num_heads_;
    size_t     medusa_num_layers_;
    size_t     hidden_size_;
    size_t     vocab_size_;
    WeightType weight_type_;

    size_t tensor_para_size_;
    size_t tensor_para_rank_;

    std::vector<LlamaDenseWeight<T>>              heads_weights_;
    std::vector<std::vector<LlamaDenseWeight<T>>> resblocks_weights_;
};

}  // namespace turbomind
