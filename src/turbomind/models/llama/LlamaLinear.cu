// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/cuda_data_type.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/scope.h"

#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/kernels/quantization.h"

#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

using namespace gemm;

struct LlamaLinear::Impl {

    explicit Impl()
    {
        workspace_ = {};

        workspace_.barriers_size   = gemm::Gemm::kBarriersSize;
        workspace_.partials_size   = gemm::Gemm::kPartialsSize;
        workspace_.tensormaps_size = 8192 * 128;  // maximum 4096 tensor maps

        auto st = core::Context::stream().handle();

        TM_CUDA_CHECK(cudaMallocAsync(&workspace_.barriers, workspace_.barriers_size, st));
        TM_CUDA_CHECK(cudaMallocAsync(&workspace_.partials, workspace_.partials_size, st));
        TM_CUDA_CHECK(cudaMallocAsync(&workspace_.tensormaps, workspace_.partials_size, st));
        TM_CUDA_CHECK(cudaMemsetAsync(workspace_.barriers, 0, workspace_.barriers_size, st));
        TM_CUDA_CHECK(cudaMallocAsync(&workspace_.flags, sizeof(int), st));

        core::Context::stream().Sync();
    }

    ~Impl()
    {
        auto st = core::Context::stream().handle();

        cudaFreeAsync(workspace_.barriers, st);
        cudaFreeAsync(workspace_.partials, st);
        cudaFreeAsync(workspace_.tensormaps, st);
        cudaFreeAsync(workspace_.flags, st);
        workspace_ = {};
    }

    std::tuple<Tensor, MatrixLayout, Tensor, MatrixLayout> GetOperandB(const LinearWeight& weight)
    {
        const Tensor& B      = weight.weight;
        const Tensor& V      = weight.scales;
        MatrixLayout  desc_B = weight.k_desc;
        MatrixLayout  desc_V = weight.q_desc;
        return {B, desc_B, V, desc_V};
    }

    std::tuple<Tensor, MatrixLayout, Tensor, MatrixLayout> GetOperandA(const LinearWeight& weight,
                                                                       const Tensor&       input,
                                                                       const Tensor&       input_scales,
                                                                       Buffer_<int>        indices,
                                                                       const Buffer_<int>& offsets)
    {
        auto st = core::Context::stream().handle();

        Tensor A;
        Tensor U;

        // Size-0 / null Buffer must not count as indexed MoE (w2 / down path).
        const bool has_indices = static_cast<bool>(indices) && indices.size() > 0;
        const int  m           = has_indices ? indices.size() : input.shape(0);

        // Currently, FP8 only; INT8 may be added later
        if (input.dtype() != weight.input_dtype()) {
            TM_SCOPE_CALL(QuantizeSymm(A, U, input, st));
        }
        else {
            A = input;
            if (weight.input_dtype() == kFloat8_e4m3) {
                TM_CHECK(input_scales) << "FP8 input requires input_scales companion (dynamic group scales)";
                U = input_scales;
            }
        }

        // SM100+ grouped bf16/fp16: use chunk() weights so Activation() runs separately.
        // FP8 SM90 gathers A/U in-kernel (GemmUniversalSm90_v3 indexed); do not host-dispatch.
        const bool is_cublas_grouped = offsets && getSMVersion() == 100 && weight.weight_format.dtype == kBfloat16;
        if (has_indices && is_cublas_grouped) {
            const int  k                = A.shape(1);
            Tensor     A_e              = {{m, k}, A.dtype(), kDEVICE};
            const int* num_valid_tokens = offsets ? offsets.data() + offsets.size() - 1 : nullptr;
            TM_SCOPE_CALL(invokeMoeDispatch(A_e, A, indices.data(), m, num_valid_tokens, st));
            if (U) {
                Tensor U_e;
                TM_SCOPE_CALL(invokeMoeDispatchScales(U_e, U, indices.data(), m, num_valid_tokens, st));
                U = U_e;
            }
            A       = A_e;
            indices = {};  // indices already applied
            // has_indices no longer used below
        }

        MatrixLayout desc_A{A.dtype(), gemm::Order::kRowMajor, m, (int)A.shape(1), (int)A.stride(0)};
        MatrixLayout desc_U{};
        if (U) {
            desc_U = {U.dtype(), kColMajor, (int)U.shape(1), (int)U.shape(0), (int)U.stride(0)};
        }
        if (offsets) {
            desc_A.num = desc_U.num = weight.k_desc.num;
            desc_A.offsets = desc_U.offsets = const_cast<int*>(offsets.data());
        }
        // Re-check after optional dispatch clears indices.
        if (indices && indices.size() > 0) {
            desc_A.idxs = desc_U.idxs = const_cast<int*>(indices.data());
        }

        return {A, desc_A, U, desc_U};
    }

    void AllocOutputScales(Tensor& scales, int m, int out_dim)
    {
        constexpr int group_size = 128;
        constexpr int alignment  = 16 / sizeof(float);  // match QuantizeSymm
        const int     s_dim      = cdiv(out_dim, group_size);
        const int     aligned_m  = round_up(m, alignment);
        if (!scales || scales.shape(0) != s_dim || scales.shape(1) != m || scales.stride(0) != aligned_m) {
            scales = Tensor_<float>{{{s_dim, m}, {aligned_m, 1}}, kDEVICE};
        }
    }

    void Forward(const Tensor&       input,
                 const Tensor&       input_scales,
                 const LinearWeight& weight,
                 const Buffer_<int>& indices,
                 const Buffer_<int>& offsets,
                 Tensor&             output,
                 Tensor&             output_scales)
    {
        TM_FUNCTION_SCOPE();
        using namespace gemm;

        Operation op{};
        op.dispatch  = dispatch_policy_;
        op.epilogue  = weight.epilogue;
        op.quant_a   = MakeQuantDesc(weight.input_format);
        op.quant_b   = MakeQuantDesc(weight.weight_format);
        op.batch_dim = 0;

        auto&& [A, desc_A, U, desc_U] = GetOperandA(weight, input, input_scales, indices, offsets);
        auto&& [B, desc_B, V, desc_V] = GetOperandB(weight);

        Tensor& D = output;
        if (!D) {
            int dim = weight.epilogue == Epilogue::kGatedSilu ? weight.output_dim / 2 : weight.output_dim;
            D       = Tensor{{desc_A.rows, dim}, weight.output_dtype(), kDEVICE};
        }

        MatrixLayout desc_D{
            output.dtype(),
            kRowMajor,
            (int)output.shape(0),
            weight.output_dim,
            (int)output.stride(0),
        };

        if (offsets) {
            desc_D.num     = desc_B.num;
            desc_D.offsets = const_cast<int*>(offsets.data());
        }

        MatrixLayout desc_W{};
        void*        W_ptr = nullptr;
        if (weight.output_dtype() == kFloat8_e4m3) {
            const int out_cols = weight.epilogue == Epilogue::kGatedSilu ? weight.output_dim / 2 : weight.output_dim;
            AllocOutputScales(output_scales, desc_A.rows, out_cols);
            desc_W = {output_scales.dtype(),
                      kColMajor,
                      (int)output_scales.shape(1),
                      (int)output_scales.shape(0),
                      (int)output_scales.stride(0)};
            W_ptr  = output_scales.raw_data();
        }

        auto ec = gemm_.Run(op,
                            1.f,
                            A.raw_data(),
                            desc_A,
                            U.data_or((void*)nullptr),
                            desc_U,
                            B.raw_data(),
                            desc_B,
                            V.data_or((void*)nullptr),
                            desc_V,
                            0.f,
                            D.raw_data(),
                            desc_D,
                            D.raw_data(),
                            desc_D,
                            W_ptr,
                            desc_W,
                            workspace_,
                            core::Context::stream().handle());

        if (ec) {
            TM_LOG_ERROR("{}: {}", __PRETTY_FUNCTION__, ec);
        }
    }

    gemm::Gemm           gemm_;
    gemm::DispatchPolicy dispatch_policy_{gemm::DispatchPolicy::kDefault};

    gemm::Workspace workspace_;
};

LlamaLinear::LlamaLinear(): impl_{std::make_shared<Impl>()} {}

void LlamaLinear::Forward(const Tensor&       input,  //
                          const LinearWeight& weight,
                          Ref<Tensor>         output)
{
    Forward(input, weight, {}, {}, output);
}

void LlamaLinear::Forward(const Tensor&       input,  //
                          const LinearWeight& weight,
                          const Buffer_<int>& indices,
                          const Buffer_<int>& offsets,
                          Ref<Tensor>         output)
{
    Tensor in = input.view({-1, input.shape(-1)});

    if (output.get()) {
        output.get() = output.get().view({-1, output.get().shape(-1)});
    }

    Tensor in_s, out_s;
    impl_->Forward(in, in_s, weight, indices, offsets, output.get(), out_s);
}

void LlamaLinear::Forward(const Tensor&       input,
                          const Tensor&       input_scales,
                          const LinearWeight& weight,
                          const Buffer_<int>& indices,
                          const Buffer_<int>& offsets,
                          Ref<Tensor>         output,
                          Ref<Tensor>         output_scales)
{
    Tensor in = input.view({-1, input.shape(-1)});

    if (output.get()) {
        output.get() = output.get().view({-1, output.get().shape(-1)});
    }

    impl_->Forward(in, input_scales, weight, indices, offsets, output.get(), output_scales.get());
}

void LlamaLinear::Forward(const Tensor&       input,
                          const Tensor&       input_scales,
                          const LinearWeight& weight,
                          Ref<Tensor>         output,
                          Ref<Tensor>         output_scales)
{
    Forward(input, input_scales, weight, {}, {}, output, output_scales);
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
