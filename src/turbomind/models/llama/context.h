// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include <cuda_runtime.h>
#include <memory>

namespace turbomind {

template<class T>
struct Context {
    cudaStream_t                                    stream;
    std::unique_ptr<Allocator<AllocatorType::CUDA>> allocator;
    std::unique_ptr<Allocator<AllocatorType::CUDA>> peer_allocator;
    std::unique_ptr<cublasAlgoMap>                  cublas_algo_map;
    std::unique_ptr<std::mutex>                     cublas_wrapper_mutex;
    std::unique_ptr<cublasMMWrapper>                cublas_wrapper;
    std::unique_ptr<LlamaLinear<T>>                 linear;
    cudaDeviceProp                                  cuda_device_prop;
};

}  // namespace turbomind
