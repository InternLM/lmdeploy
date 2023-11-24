// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

class BatchedCopy {
public:
    template<class T, std::enable_if_t<alignof(T) <= alignof(uint32_t), int> = 0>
    T* Add(const T* src, int size, T* dst)
    {
        src_.push_back((void*)src);
        dst_.push_back((void*)dst);
        size_.push_back(sizeof(T) * size);
        return dst + size;
    }

    void Submit(cudaStream_t stream)
    {
        invokeBatchedCopy(src_.data(), dst_.data(), size_.data(), size_.size(), stream);
        sync_check_cuda_error();

        src_.clear();
        dst_.clear();
        size_.clear();
    }

private:
    std::vector<void*> src_;
    std::vector<void*> dst_;
    std::vector<int>   size_;
};

}  // namespace turbomind