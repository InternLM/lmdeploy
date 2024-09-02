// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/marlin_qqq_gemm/marlin_qqq_gemm.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_decoder_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/allocator.h"
#include <fstream>

namespace turbomind {

template<class T>
struct LlamaLinear<T>::Impl {

    Impl(cublasMMWrapper* cublas_wrapper, cudaStream_t stream, IAllocator* allocator):
        cublas_wrapper_(cublas_wrapper), stream_(stream), gemm_s4_s8_(allocator)
    {
        workspace_ = {};

        workspace_.barriers_size = gemm::Gemm::kBarriersSize;
        workspace_.partials_size = gemm::Gemm::kPartialsSize;
        cudaMallocAsync(&workspace_.barriers, workspace_.barriers_size, stream_);
        cudaMallocAsync(&workspace_.partials, workspace_.partials_size, stream_);
        cudaMemsetAsync(workspace_.barriers, 0, workspace_.barriers_size, stream_);
    }

    ~Impl()
    {
        cudaFreeAsync(workspace_.barriers, stream_);
        cudaFreeAsync(workspace_.partials, stream_);
        workspace_ = {};
    }

    void forward(T*                         output_data,
                 Pitched                    input_data,
                 int8_t*                    quant_input_data,
                 float*                     quant_scale,
                 int                        batch_size,
                 const LlamaDenseWeight<T>& weight,
                 Type                       type      = kGemm,
                 int*                       lora_mask = nullptr)
    {
        if (input_data.pitch == 0) {
            input_data.pitch = weight.input_dims;
        }
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
                                  input_data.ptr,                                 // B
                                  input_data.pitch,                               // ldb
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
            sync_check_cuda_error();

            type = kFusedAdd;
        }
        switch (weight.quantization) {
            case QuantMethod::QNone:
                return forwardFp(output_data, input_data, batch_size, weight, type);
            case QuantMethod::AWQ:
            case QuantMethod::GPTQ:
                return forwardInt4(output_data, input_data, batch_size, weight, type);
            case QuantMethod::QQQ:
                return forwardQQQ(output_data, quant_input_data, quant_scale, batch_size, weight, type);
            default:
                FT_CHECK(0);
        }
    }

    void forwardFp(T* output_data, Pitched input_data, int batch_size, const LlamaDenseWeight<T>& weight, Type type)
    {
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              weight.output_dims,
                              batch_size,
                              weight.input_dims,
                              (const T*)weight.kernel,
                              weight.output_dims,
                              input_data.ptr,
                              input_data.pitch,
                              output_data,
                              weight.output_dims,
                              1.0f,
                              type == kFusedAdd ? 1.0f : 0.0f);
        sync_check_cuda_error();
    }

    void forwardInt4(T* output_data, Pitched input_data, int batch_size, const LlamaDenseWeight<T>& weight, Type type)
    {
        using namespace gemm;

        const Operation operation{dispatch_policy_,
                                  type == kFusedSiluFfn ? Epilogue::kGatedSilu : Epilogue::kNone,
                                  {QuantType::kNone},
                                  {QuantType::kDefault, weight.group_size},
                                  0,
                                  nullptr};

        const MatrixLayout a_desc{
            get_data_type_v<T>,
            kRowMajor,
            batch_size,
            (int)weight.input_dims,
            input_data.pitch,
        };

        const MatrixLayout c_desc{
            get_data_type_v<T>,
            kRowMajor,
            batch_size,
            (int)weight.output_dims,
            type == kFusedSiluFfn ? (int)weight.output_dims / 2 : (int)weight.output_dims,
        };

        auto ec = gemm_.Run(operation,
                            1.f,
                            input_data.ptr,
                            a_desc,
                            nullptr,
                            {},
                            weight.kernel,
                            weight.k_desc,
                            weight.scales_zeros,
                            weight.q_desc,
                            type == kFusedAdd ? 1.0f : 0.0f,
                            output_data,
                            c_desc,
                            output_data,
                            c_desc,
                            workspace_,
                            stream_);

        if (ec) {
            TM_LOG_ERROR("%s: %d", __PRETTY_FUNCTION__, ec);
            // std::abort();
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
                            (const half*)weight.scales_zeros,
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

    cublasMMWrapper*          cublas_wrapper_;
    gemm::Gemm                gemm_;
    gemm::DispatchPolicy      dispatch_policy_{gemm::DispatchPolicy::kDefault};
    marlin_qqq::MarlinQQQGemm gemm_s4_s8_;
    cudaStream_t              stream_{};

    gemm::Workspace workspace_;
};

template<class T>
LlamaLinear<T>::LlamaLinear(cublasMMWrapper* cublas_wrapper, cudaStream_t stream, IAllocator* allocator):
    impl_{std::make_shared<Impl>(cublas_wrapper, stream, allocator)}
{
}

template<class T>
void LlamaLinear<T>::forward(T*                         output_data,
                             Pitched                    input_data,
                             int8_t*                    quant_input_data,
                             float*                     quant_scale,
                             int                        batch_size,
                             const LlamaDenseWeight<T>& weight,
                             Type                       type,
                             int*                       lora_mask)
{
    impl_->forward(output_data, input_data, quant_input_data, quant_scale, batch_size, weight, type, lora_mask);
}

template<class T>
void LlamaLinear<T>::set_measure(bool measure)
{
    impl_->dispatch_policy_ = measure ? gemm::DispatchPolicy::kMeasure : gemm::DispatchPolicy::kReuse;
}

template<class T>
int LlamaLinear<T>::Export(std::ostream& os)
{
    if (os) {
        return impl_->gemm_.Export(os);
    }
    return 0;
}

template<class T>
int LlamaLinear<T>::Import(std::istream& is)
{
    auto n_records = 0;
    if (is) {
        n_records = impl_->gemm_.Import(is);
    }
    if (n_records) {
        impl_->dispatch_policy_ = gemm::DispatchPolicy::kReuse;
    };
    return n_records;
}

template<class T>
std::pair<int*, int*> LlamaLinear<T>::getQQQBuffer()
{
    return impl_->gemm_s4_s8_.getBuffer();
}

template<class T>
void LlamaLinear<T>::setQQQBuffer(int* reduce_buf, int* workspace_buf)
{
    impl_->gemm_s4_s8_.setBuffer(reduce_buf, workspace_buf);
}

template<class T>
std::vector<int> LlamaLinear<T>::GetTuningSeq() const
{
    return impl_->gemm_.GetTuningSeq();
}

#ifdef ENABLE_FP32
template class LlamaLinear<float>;
#endif
template class LlamaLinear<half>;
#ifdef ENABLE_BF16
template class LlamaLinear<__nv_bfloat16>;
#endif

}  // namespace turbomind
