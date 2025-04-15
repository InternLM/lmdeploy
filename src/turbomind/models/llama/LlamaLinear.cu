// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_decoder_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/core/cuda_data_type.h"

namespace turbomind {

struct LlamaLinear::Impl {

    explicit Impl(cudaStream_t stream): stream_(stream)
    {
        workspace_ = {};

        workspace_.barriers_size = gemm::Gemm::kBarriersSize;
        workspace_.partials_size = gemm::Gemm::kPartialsSize;

        check_cuda_error(cudaMallocAsync(&workspace_.barriers, workspace_.barriers_size, stream_));
        check_cuda_error(cudaMallocAsync(&workspace_.partials, workspace_.partials_size, stream_));
        check_cuda_error(cudaMemsetAsync(workspace_.barriers, 0, workspace_.barriers_size, stream_));

        check_cuda_error(cublasCreate(&cublas_));
        check_cuda_error(cublasSetStream(cublas_, stream_));
        check_cuda_error(cublasSetWorkspace(cublas_, workspace_.partials, workspace_.partials_size));

        if (0) {
            check_cuda_error(cublasSetMathMode(cublas_, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        }
    }

    ~Impl()
    {
        cublasDestroy(cublas_);
        cudaFreeAsync(workspace_.barriers, stream_);
        cudaFreeAsync(workspace_.partials, stream_);
        workspace_ = {};
    }

    void forward(core::Tensor& output, const core::Tensor& input, const LlamaDenseWeight& dense, Type type)
    {
        switch (dense.weight_type) {
            case kF16:
            case kF32:
            case kBF16:
                return forwardFp(output, input, dense.weight);
            case kU4:
                return forwardInt4(output, input, dense, type);
            default:
                TM_CHECK(0) << "not implemented for weight type: " << dense.weight_type;
        }
    }

    void forwardFp(core::Ref<core::Tensor> output_, const core::Tensor& input, const core::Tensor& weight)
    {
        auto& output = output_.get();
        TM_CHECK_EQ(weight.ndim(), 2);
        TM_CHECK_EQ(input.ndim(), 2);
        TM_CHECK_EQ(output.ndim(), 2);

        int m, n, k;
        std::tie(k, m) = weight.shapes(0, 1);
        n              = input.shape(0);

        TM_CHECK_EQ(input.shape(1), k);
        TM_CHECK_EQ(output.shape(0), n);
        TM_CHECK_EQ(output.shape(1), m);

        // [k, m]
        cublasOperation_t transa = weight.stride(1) == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
        // [n, k]
        cublasOperation_t transb = input.stride(1) == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

        const float alpha = 1.f;
        const float beta  = 0.f;

        check_cuda_error(cublasGemmEx(cublas_,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      &alpha,
                                      weight.raw_data(),
                                      to_cuda_dtype(weight.dtype()),
                                      weight.stride(0) * weight.stride(1),  // one of these is 1
                                      input.raw_data(),
                                      to_cuda_dtype(input.dtype()),
                                      input.stride(0) * input.stride(1),  // one of these is 1
                                      &beta,
                                      output.raw_data(),
                                      to_cuda_dtype(output.dtype()),
                                      output.stride(0) * output.stride(1),  // one of these is 1
                                      CUDA_R_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    void forwardInt4(core::Tensor& output, const core::Tensor& input, const LlamaDenseWeight& dense, Type type)
    {
        TM_CHECK_EQ(output.ndim(), 2);  // A [m, k]
        TM_CHECK_EQ(input.ndim(), 2);   // C [m, n]

        TM_CHECK_EQ(input.stride(1), 1) << "input must be row-major";
        TM_CHECK_EQ(output.stride(1), 1) << "output must be row-major";

        TM_CHECK_EQ(output.shape(0), input.shape(0));
        TM_CHECK_EQ(input.shape(1), dense.input_dim);
        // TM_CHECK_EQ(output.shape(1), dense.output_dim);

        using namespace gemm;

        const Operation operation{dispatch_policy_,
                                  type == kFusedSiluFfn ? Epilogue::kGatedSilu : Epilogue::kNone,
                                  {QuantType::kNone},
                                  {QuantType::kDefault, dense.group_size},
                                  0,
                                  {},
                                  nullptr};

        const MatrixLayout a_desc{
            input.dtype(),
            kRowMajor,
            (int)input.shape(0),
            dense.input_dim,
            (int)input.stride(0),
        };

        const MatrixLayout c_desc{
            output.dtype(),  //
            kRowMajor,
            (int)output.shape(0),
            dense.output_dim,
            (int)output.stride(0),
            // type == kFusedSiluFfn ? (int)weight.output_dim / 2 : (int)weight.output_dim,
        };

        auto ec = gemm_.Run(operation,
                            1.f,
                            input.raw_data(),
                            a_desc,
                            nullptr,
                            {},
                            dense.weight.raw_data(),
                            dense.k_desc,
                            dense.scales_zeros.raw_data(),
                            dense.q_desc,
                            type == kFusedAdd ? 1.0f : 0.0f,
                            output.raw_data(),
                            c_desc,
                            output.raw_data(),
                            c_desc,
                            workspace_,
                            stream_);

        if (ec) {
            TM_LOG_ERROR("%s: %d", __PRETTY_FUNCTION__, ec);
        }
    }

    void forward_moe(core::Tensor&           output,
                     const core::Tensor&     input,
                     const int*              indexes,
                     const int*              offsets,
                     const LlamaDenseWeight& dense,
                     Type                    type,
                     gemm::Context*          context)
    {
        using namespace gemm;

        QuantDesc quant_b{};
        if (dense.k_desc.type == kU4) {
            quant_b.type       = QuantType::kDefault;
            quant_b.group_size = dense.group_size;
        }

        const Operation operation{dispatch_policy_,
                                  type == kFusedSiluFfn ? Epilogue::kGatedSilu : Epilogue::kNone,
                                  {QuantType::kNone},
                                  quant_b,
                                  0,
                                  context,
                                  nullptr};

        MatrixLayout a_desc{
            input.dtype(),
            kRowMajor,
            (int)output.shape(0),  // batch size
            dense.input_dim,       // k
            (int)input.stride(0),
        };

        a_desc.offsets = (int*)offsets;
        a_desc.idxs    = (int*)indexes;

        // std::cout << "m" << batch_size << "n" << weight.output_dims << "k" << weight.input_dims << " "
        //           << input_data.pitch << "\n";

        MatrixLayout c_desc{
            output.dtype(),  //
            kRowMajor,
            (int)output.shape(0),  // batch size
            dense.output_dim,
            (int)output.stride(0),
            // type == kFusedSiluFfn ? (int)weight.output_dims / 2 : (int)weight.output_dims,
        };

        c_desc.offsets = (int*)offsets;

        a_desc.num = c_desc.num = dense.k_desc.num;

        auto ec = gemm_.Run(operation,
                            1.f,
                            input.raw_data(),
                            a_desc,
                            nullptr,
                            {},
                            dense.weight.raw_data(),
                            dense.k_desc,
                            dense.scales_zeros.buffer().unsafe_data(),
                            dense.q_desc,
                            type == kFusedAdd ? 1.0f : 0.0f,
                            output.raw_data(),
                            c_desc,
                            output.raw_data(),
                            c_desc,
                            workspace_,
                            stream_);

        if (ec) {
            TM_LOG_ERROR("%s: %d", __PRETTY_FUNCTION__, ec);
        }
    }

    // cublasMMWrapper*     cublas_wrapper_;
    cublasHandle_t       cublas_;
    gemm::Gemm           gemm_;
    gemm::DispatchPolicy dispatch_policy_{gemm::DispatchPolicy::kDefault};
    cudaStream_t         stream_{};

    gemm::Workspace workspace_;
};

LlamaLinear::LlamaLinear(cudaStream_t stream): impl_{std::make_shared<Impl>(stream)} {}

core::Tensor LlamaLinear::forward(const core::Tensor&         input,  //
                                  const LlamaDenseWeight&     dense,
                                  Type                        type,
                                  std::optional<core::Tensor> output)
{
    core::ssize_t output_dim = type == kFusedSiluFfn ? dense.output_dim / 2 : dense.output_dim;

    core::Tensor in = input.view({-1, input.shape(-1)});
    core::Tensor out;

    if (output) {
        out = output->view({in.shape(0), output_dim});
    }
    else {
        out = core::Tensor({in.shape(0), output_dim}, input.dtype(), input.device());
    }

    impl_->forward(out, in, dense, type);

    auto shape   = input.shape();
    shape.back() = out.shape(-1);

    return out.view(shape);
}

void LlamaLinear::forward_moe(core::Tensor&           output,
                              const core::Tensor&     input,
                              const int*              indexes,
                              const int*              offsets,
                              const LlamaDenseWeight& dense,
                              Type                    type,
                              gemm::Context*          context)
{
    return impl_->forward_moe(output, input, indexes, offsets, dense, type, context);
}

void LlamaLinear::set_measure(bool measure)
{
    impl_->dispatch_policy_ = measure ? gemm::DispatchPolicy::kMeasure : gemm::DispatchPolicy::kReuse;
}

int LlamaLinear::Export(std::ostream& os)
{
    if (os) {
        return impl_->gemm_.Export(os);
    }
    return 0;
}

int LlamaLinear::Import(std::istream& is)
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

std::vector<int> LlamaLinear::GetTuningSeq() const
{
    return impl_->gemm_.GetTuningSeq();
}

}  // namespace turbomind
