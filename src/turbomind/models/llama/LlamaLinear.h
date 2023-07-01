// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T>
class LlamaLinear {
public:
    LlamaLinear(cublasMMWrapper* cublas_wrapper, cudaStream_t stream): cublas_wrapper_(cublas_wrapper), stream_(stream)
    {
    }

    void forward(T* output_data, const T* input_data, int batch_size, const LlamaDenseWeight<T>& weight)
    {
        switch (weight.type) {
            case WeightType::kFP16:
            case WeightType::kFP32:
                forwardFp(output_data, input_data, batch_size, weight);
                break;
            case WeightType::kINT4:
                forwardInt4(output_data, input_data, batch_size, weight);
                break;
            default:
                FT_CHECK(0);
        }
    }

private:
    void forwardFp(T* output_data, const T* input_data, int batch_size, const LlamaDenseWeight<T>& weight)
    {
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              weight.output_dims,
                              batch_size,
                              weight.input_dims,
                              (const T*)weight.kernel,
                              weight.output_dims,
                              input_data,
                              weight.input_dims,
                              output_data,
                              weight.output_dims);
        sync_check_cuda_error();
    }

    void forwardInt4(T* output_data, const T* input_data, int batch_size, const LlamaDenseWeight<T>& weight)
    {
        FT_CHECK_WITH_INFO(0, "Not implemented");
    }

private:
    cublasMMWrapper* cublas_wrapper_;
    cudaStream_t     stream_{};
};

}  // namespace turbomind
