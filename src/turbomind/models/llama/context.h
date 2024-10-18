// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <memory>

namespace turbomind {

template<class T>
struct Context {
    cudaStream_t                                    stream;
    std::unique_ptr<Allocator<AllocatorType::CUDA>> allocator;
    std::unique_ptr<Allocator<AllocatorType::CUDA>> peer_allocator;
    cublasHandle_t                                  cublas_handle;
    cublasLtHandle_t                                cublasLt_handle;
    std::unique_ptr<cublasAlgoMap>                  cublas_algo_map;
    std::unique_ptr<std::mutex>                     cublas_wrapper_mutex;
    std::unique_ptr<cublasMMWrapper>                cublas_wrapper;
    std::unique_ptr<LlamaLinear<T>>                 linear;
    cudaDeviceProp                                  cuda_device_prop;

    Context(int device_id)
    {
        check_cuda_error(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        allocator = std::make_unique<Allocator<AllocatorType::CUDA>>(device_id, false);
        allocator->setStream(stream);

        peer_allocator = std::make_unique<Allocator<AllocatorType::CUDA>>(device_id, true);
        peer_allocator->setStream(stream);

        cublasCreate(&cublas_handle);
        cublasLtCreate(&cublasLt_handle);
        cublasSetStream(cublas_handle, stream);

        if (0) {
            cublasSetWorkspace(cublas_handle, nullptr, 0);
            cublasSetMathMode(cublas_handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        }

        cublas_algo_map      = std::make_unique<cublasAlgoMap>("gemm_config.in");
        cublas_wrapper_mutex = std::make_unique<std::mutex>();
        cublas_wrapper       = std::make_unique<cublasMMWrapper>(
            cublas_handle, cublasLt_handle, stream, cublas_algo_map.get(), cublas_wrapper_mutex.get(), allocator.get());
        linear = std::make_unique<LlamaLinear<T>>(cublas_wrapper.get(), stream);

        check_cuda_error(cudaGetDeviceProperties(&cuda_device_prop, device_id));

        if (std::is_same<T, half>::value) {
            cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
#ifdef ENABLE_FP32
        else if (std::is_same<T, float>::value) {
            cublas_wrapper->setFP32GemmConfig();
        }
#endif
#ifdef ENABLE_BF16
        else if (std::is_same<T, __nv_bfloat16>::value) {
            cublas_wrapper->setBF16GemmConfig();
        }
#endif
    }

    ~Context()
    {
        linear.reset();
        cublas_wrapper.reset();
        cublas_algo_map.reset();

        cublasDestroy(cublas_handle);
        cublas_handle = {};

        cublasLtDestroy(cublasLt_handle);
        cublasLt_handle = {};

        peer_allocator.reset();
        allocator.reset();

        cudaStreamDestroy(stream);
        stream = {};
    }
};

}  // namespace turbomind
