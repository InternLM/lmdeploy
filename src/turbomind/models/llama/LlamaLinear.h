// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm_s_f16/gemm_s4_f16.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_decoder_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include <type_traits>

namespace turbomind {

template<typename T>
class LlamaLinear {
public:
    enum Type
    {
        kGemm,
        kFusedSiluFfn
    };

    LlamaLinear(cublasMMWrapper* cublas_wrapper, cudaStream_t stream): cublas_wrapper_(cublas_wrapper), stream_(stream)
    {
    }

    void forward(T*                         output_data,
                 const T*                   input_data,
                 int                        batch_size,
                 const LlamaDenseWeight<T>& weight,
                 Type                       type      = kGemm,
                 int*                       lora_mask = nullptr)
    {
        switch (weight.type) {
            case WeightType::kFP16:
            case WeightType::kFP32:
            case WeightType::kBF16:
                forwardFp(output_data, input_data, batch_size, weight, type, lora_mask);
                break;
            case WeightType::kINT4:
                forwardInt4(output_data, input_data, batch_size, weight, type);
                break;
            default:
                FT_CHECK(0);
        }
    }

private:
    void forwardFp(T*                         output_data,
                   const T*                   input_data,
                   int                        batch_size,
                   const LlamaDenseWeight<T>& weight,
                   Type                       type,
                   int*                       lora_mask)
    {
        FT_CHECK(type == kGemm);
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

        if (lora_mask != nullptr && weight.lora_r > 0) {
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  weight.lora_r,                                      // m
                                  batch_size,                                         // n
                                  weight.input_dims,                                  // k
                                  (const T*)weight.lora_a,                            // A
                                  weight.lora_r,                                      // lda
                                  input_data,                                         // B
                                  weight.input_dims,                                  // ldb
                                  output_data + batch_size * weight.output_dims * 2,  // C
                                  weight.lora_r);                                     // ldc

            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  weight.output_dims,                                 // m
                                  batch_size,                                         // n
                                  weight.lora_r,                                      // k
                                  (const T*)weight.lora_b,                            // A
                                  weight.output_dims,                                 // lda
                                  output_data + batch_size * weight.output_dims * 2,  // B
                                  weight.lora_r,                                      // ldb
                                  output_data + batch_size * weight.output_dims,      // C
                                  weight.output_dims);                                // ldc

            invokeMaskAddTwoLinearOutput2(output_data,
                                          output_data + batch_size * weight.output_dims,
                                          weight.lora_scale,
                                          lora_mask,
                                          batch_size,
                                          weight.output_dims,
                                          stream_);
            sync_check_cuda_error();
        }
    }

    void forwardInt4(T* output_data, const T* input_data, int batch_size, const LlamaDenseWeight<T>& weight, Type type)
    {
        if constexpr (std::is_same_v<T, half>) {
            gemm_s4_f16_.Run(output_data,
                             (const uint*)weight.kernel,
                             input_data,
                             (const half2*)weight.scales_and_zeros,
                             weight.output_dims,
                             batch_size,
                             weight.input_dims,
                             weight.group_size,
                             type == kFusedSiluFfn ? GemmS4F16::kFusedSiluFfn : GemmS4F16::kGemm,
                             -1,
                             stream_);
            sync_check_cuda_error();
        }
        else {
            FT_CHECK_WITH_INFO(0, "Not implemented");
        }
    }

private:
    cublasMMWrapper* cublas_wrapper_;
    cudaStream_t     stream_{};
    GemmS4F16        gemm_s4_f16_;
};

}  // namespace turbomind
