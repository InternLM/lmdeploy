// Copyright (c) OpenMMLab. All rights reserved.
// Yineng Zhang <me@zhyncs.com>
// Zhiwei Bao <zwbao@foxmail.com>

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/nccl_utils.h"

namespace turbomind {

template<typename T>
class ResBlock {
public:
    ResBlock(size_t in_size, cudaStream_t stream, cublasMMWrapper* cublas_wrapper, NcclParam tensor_para):
        in_size_(in_size), stream_(stream), tensor_para_(tensor_para)
    {
        linear_ = std::make_unique<LlamaLinear<T>>(cublas_wrapper, stream);
    }
    ~ResBlock()               = default;
    ResBlock(const ResBlock&) = delete;
    ResBlock& operator=(const ResBlock&) = delete;

    void forward(TensorMap* output_tensors, const TensorMap* input_tensors, const LlamaDenseWeight<T>& weight);
    void forward(T* resblock_output, const T* resblock_input, size_t batch_size, const LlamaDenseWeight<T>& weight);

private:
    size_t in_size_;

    cudaStream_t                    stream_;
    std::unique_ptr<LlamaLinear<T>> linear_;
    NcclParam                       tensor_para_;
};
}  // namespace turbomind
