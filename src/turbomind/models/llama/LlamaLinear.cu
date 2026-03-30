// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/cuda_data_type.h"
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/kernels/quantization.h"

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
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

        check_cuda_error(cudaMallocAsync(&workspace_.barriers, workspace_.barriers_size, st));
        check_cuda_error(cudaMallocAsync(&workspace_.partials, workspace_.partials_size, st));
        check_cuda_error(cudaMallocAsync(&workspace_.tensormaps, workspace_.partials_size, st));
        check_cuda_error(cudaMemsetAsync(workspace_.barriers, 0, workspace_.barriers_size, st));
        check_cuda_error(cudaMallocAsync(&workspace_.flags, sizeof(int), st));

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

    std::tuple<Tensor, MatrixLayout, Tensor, MatrixLayout> GetOperandB(const LlamaDenseWeight& dense)
    {
        const Tensor& B      = dense.weight;
        const Tensor& V      = dense.scales;
        MatrixLayout  desc_B = dense.k_desc;
        MatrixLayout  desc_V = dense.q_desc;
        return {B, desc_B, V, desc_V};
    }

    std::tuple<Tensor, MatrixLayout, Tensor, MatrixLayout>
    GetOperandA(const LlamaDenseWeight& dense, const Tensor& input, Buffer_<int> indices, const Buffer_<int>& offsets)
    {
        auto st = core::Context::stream().handle();

        Tensor A;
        Tensor U;

        const int m = indices ? indices.size() : input.shape(0);

        // Currently, FP8 only; INT8 may be added later
        if (input.dtype() != dense.input_type) {
            QuantizeSymm(A, U, input, st);
            sync_check_cuda_error();
        }
        else {
            A = input;
        }

        if (indices && A.dtype() == kFloat8_e4m3) {
            const auto [bsz, k] = A.shapes(0, 1);
            const int e         = indices.size() / bsz;
            Tensor    A_e       = {{m, k}, A.dtype(), kDEVICE};
            invokeMoeDispatch(A_e, A, indices.data(), e, st);
            sync_check_cuda_error();
            Tensor U_e;
            invokeMoeDispatchScales(U_e, U, indices.data(), e, st);
            sync_check_cuda_error();
            A       = A_e;
            U       = U_e;
            indices = {};  // indices already applied
        }

        MatrixLayout desc_A{A.dtype(), gemm::Order::kRowMajor, m, (int)A.shape(1), (int)A.stride(0)};
        MatrixLayout desc_U{};
        if (U) {
            desc_U = {U.dtype(), kColMajor, (int)U.shape(1), (int)U.shape(0), (int)U.stride(0)};
        }
        if (offsets) {
            desc_A.num = desc_U.num = dense.k_desc.num;
            desc_A.offsets = desc_U.offsets = const_cast<int*>(offsets.data());
        }
        if (indices) {
            desc_A.idxs = desc_U.idxs = const_cast<int*>(indices.data());
        }

        return {A, desc_A, U, desc_U};
    }

    void Forward(Tensor&                 output,
                 const Tensor&           input,  //
                 const LlamaDenseWeight& dense,
                 const Buffer_<int>&     indices,
                 const Buffer_<int>&     offsets)
    {
        using namespace gemm;

        Operation op{};
        op.dispatch  = dispatch_policy_;
        op.epilogue  = dense.epilogue;
        op.quant_a   = dense.input_quant;
        op.quant_b   = dense.weight_quant;
        op.batch_dim = 0;

        auto&& [A, desc_A, U, desc_U] = GetOperandA(dense, input, indices, offsets);
        auto&& [B, desc_B, V, desc_V] = GetOperandB(dense);

        Tensor& D = output;
        if (!D) {
            int dim = dense.epilogue == Epilogue::kGatedSilu ? dense.output_dim / 2 : dense.output_dim;
            D       = Tensor{{desc_A.rows, dim}, dense.data_type, kDEVICE};
        }

        // std::cout << "D: " << D << " " << desc_B.num << "\n";

        MatrixLayout desc_D{
            output.dtype(),
            kRowMajor,
            (int)output.shape(0),
            dense.output_dim,
            (int)output.stride(0),
        };

        if (offsets) {
            desc_D.num     = desc_B.num;
            desc_D.offsets = const_cast<int*>(offsets.data());
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
                            workspace_,
                            core::Context::stream().handle());

        if (ec) {
            TM_LOG_ERROR("%s: %d", __PRETTY_FUNCTION__, ec);
        }
    }

    gemm::Gemm           gemm_;
    gemm::DispatchPolicy dispatch_policy_{gemm::DispatchPolicy::kDefault};

    gemm::Workspace workspace_;
};

LlamaLinear::LlamaLinear(): impl_{std::make_shared<Impl>()} {}

Tensor LlamaLinear::Forward(const Tensor&           input,  //
                            const LlamaDenseWeight& weight,
                            std::optional<Tensor>   output)
{
    return Forward(input, weight, {}, {}, output);
}

Tensor LlamaLinear::Forward(const Tensor&           input,  //
                            const LlamaDenseWeight& weight,
                            const Buffer_<int>&     indices,
                            const Buffer_<int>&     offsets,
                            std::optional<Tensor>   output)
{
    Tensor in = input.view({-1, input.shape(-1)});
    Tensor out;

    if (output) {
        out = output->view({-1, output->shape(-1)});
    }

    impl_->Forward(out, in, weight, indices, offsets);

    return out;
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
