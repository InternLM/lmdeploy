// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm_s_f16/gemm_s4_f16.h"
#include "src/turbomind/kernels/marlin_qqq_gemm/marlin_qqq_gemm.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_decoder_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/allocator.h"
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
        kFusedSiluFfn,
        kFusedAdd
    };

    LlamaLinear(cublasMMWrapper* cublas_wrapper, cudaStream_t stream, IAllocator* allocator):
        cublas_wrapper_(cublas_wrapper), stream_(stream), gemm_s4_s8_(allocator)
    {
    }

    void forward(T*                         output_data,
                 const T*                   input_data,
                 int8_t*                    quant_input_data,
                 float*                     quant_scale,
                 int                        batch_size,
                 const LlamaDenseWeight<T>& weight,
                 Type                       type      = kGemm,
                 int*                       lora_mask = nullptr)
    {
        if (lora_mask != nullptr && weight.lora.r > 0) {
            FT_CHECK(type == kGemm);
            // output = lora(x) * scale
            // output = mask(output)
            // output = x*W + output
            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  weight.lora.r,                                  // m
                                  batch_size,                                     // n
                                  weight.input_dims,                              // k
                                  (const T*)weight.lora.a,                        // A
                                  weight.lora.r,                                  // lda
                                  input_data,                                     // B
                                  weight.input_dims,                              // ldb
                                  output_data + batch_size * weight.output_dims,  // C
                                  weight.lora.r);                                 // ldc

            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  weight.output_dims,                             // m
                                  batch_size,                                     // n
                                  weight.lora.r,                                  // k
                                  (const T*)weight.lora.b,                        // A
                                  weight.output_dims,                             // lda
                                  output_data + batch_size * weight.output_dims,  // B
                                  weight.lora.r,                                  // ldb
                                  output_data,                                    // C
                                  weight.output_dims,                             // ldc
                                  weight.lora.scale,                              // alpha
                                  0.0f);                                          // beta

            invokeMask(output_data, lora_mask, batch_size, weight.output_dims, stream_);
            type = kFusedAdd;
        }
        switch (weight.quantization) {
            case QuantMethod::QNone:
                forwardFp(output_data, input_data, batch_size, weight, type);
                break;
            case QuantMethod::AWQ:
                forwardInt4(output_data, input_data, batch_size, weight, type);
                break;
            case QuantMethod::QQQ:
                forwardQQQ(output_data, quant_input_data, quant_scale, batch_size, weight, type);
                break;
            default:
                FT_CHECK(0);
        }
    }

    void setQQQBuffer(int* reduce_buf, int* workspace_buf)
    {
        gemm_s4_s8_.setBuffer(reduce_buf, workspace_buf);
    }

    std::pair<int*, int*> getQQQBuffer()
    {
        return gemm_s4_s8_.getBuffer();
    }

private:
    void forwardFp(T* output_data, const T* input_data, int batch_size, const LlamaDenseWeight<T>& weight, Type type)
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
                              weight.output_dims,
                              1.0f,
                              type == kFusedAdd ? 1.0f : 0.0f);
        sync_check_cuda_error();
    }

    void forwardInt4(T* output_data, const T* input_data, int batch_size, const LlamaDenseWeight<T>& weight, Type type)
    {
        GemmS4F16::Type gemm_type = GemmS4F16::kGemm;
        if (type == kFusedAdd)
            gemm_type = GemmS4F16::kFusedAdd;
        if (type == kFusedSiluFfn)
            gemm_type = GemmS4F16::kFusedSiluFfn;
        if constexpr (std::is_same_v<T, half>) {
            gemm_s4_f16_.Run(output_data,
                             (const uint*)weight.kernel,
                             input_data,
                             (const half2*)weight.scales_and_zeros,
                             weight.output_dims,
                             batch_size,
                             weight.input_dims,
                             weight.group_size,
                             gemm_type,
                             -1,
                             stream_);
            sync_check_cuda_error();
        }
        else {
            FT_CHECK_WITH_INFO(0, "Not implemented");
        }
    }

    // w4a8
    void forwardQQQ(T*                         output_data,
                    const int8_t*              input_data,
                    const float*               act_scale,
                    int                        batch_size,
                    const LlamaDenseWeight<T>& weight,
                    Type                       type)
    {
        // qqq only supports kGemm
        FT_CHECK(type == kGemm);
        if constexpr (std::is_same_v<T, half>) {
            gemm_s4_s8_.Run(output_data,
                            input_data,
                            (const uint*)weight.kernel,
                            act_scale,
                            (const float*)weight.scales_channel,
                            (const half*)weight.scales_and_zeros,
                            batch_size,
                            weight.output_dims,
                            weight.input_dims,
                            weight.group_size,
                            stream_);
            sync_check_cuda_error();
        }
        else {
            FT_CHECK_WITH_INFO(0, "Not implemented");
        }
    }

private:
    cublasMMWrapper*          cublas_wrapper_;
    cudaStream_t              stream_{};
    GemmS4F16                 gemm_s4_f16_;
    marlin_qqq::MarlinQQQGemm gemm_s4_s8_;
};

}  // namespace turbomind
