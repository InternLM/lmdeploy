#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "src/turbomind/core/cuda_data_type.h"
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/cublas.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cstdio>
#include <vector>

namespace turbomind::gemm {

class CublasKernel: public Kernel {
public:
    CublasKernel(const Family& family, bool (*available)(int)): Kernel{family}, cublas_{}, available_{available}
    {
        cublasCreate(&cublas_);
        if (0) {
            cublasSetMathMode(cublas_, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        }

        desc_.backend    = 1;
        desc_.group_axis = -1;

        info_.chunk_size_k      = 1;
        info_.dynamic_smem_size = 0;

        info_.name = GetName();
    }

    ~CublasKernel() override
    {
        cublasDestroy(cublas_);
        cublas_ = {};
    }

    int Launch(const Operation&    operation,
               float               alpha,
               const void*         A,
               const MatrixLayout& Adesc,
               const void*         U,
               const MatrixLayout& Udesc,
               const void*         B,
               const MatrixLayout& Bdesc,
               const void*         V,
               const MatrixLayout& Vdesc,
               const void*         global_scale,
               const MatrixLayout& global_scale_desc,
               float               beta,
               const void*         C,
               const MatrixLayout& Cdesc,
               void*               D,
               const MatrixLayout& Ddesc,
               void*               W,
               const MatrixLayout& Wdesc,
               int                 swizzle,
               int                 splits,
               Workspace&          workspace,
               cudaStream_t        stream) override
    {
        (void)W;
        (void)Wdesc;
        (void)global_scale;
        (void)global_scale_desc;
        cublasOperation_t transa = Adesc.order == kColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t transb = Bdesc.order == kColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;

        const int m = Adesc.rows;
        const int n = Bdesc.cols;
        const int k = Adesc.cols;

        TM_CHECK_EQ(Bdesc.rows, k);
        TM_CHECK_EQ(Ddesc.rows, m);
        TM_CHECK_EQ(Ddesc.cols, n);

        TM_CHECK(C == nullptr || C == D);

        if (stream_ != stream) {
            cublasSetStream(cublas_, stream);
            stream_ = stream;
        }

        if (workspace_ != workspace.partials || workspace_size_ != workspace.partials_size) {
            cublasSetWorkspace(cublas_, workspace.partials, workspace.partials_size);
            workspace_      = workspace.partials;
            workspace_size_ = workspace.partials_size;
        }

        auto ec = cublasGemmEx(cublas_,
                               transa,
                               transb,
                               m,
                               n,
                               k,
                               &alpha,
                               A,
                               to_cuda_dtype(Adesc.type),
                               Adesc.ld,
                               B,
                               to_cuda_dtype(Bdesc.type),
                               Bdesc.ld,
                               &beta,
                               D,
                               to_cuda_dtype(Ddesc.type),
                               Ddesc.ld,
                               CUDA_R_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        return ec == CUBLAS_STATUS_SUCCESS ? 0 : 1;
    }

    bool is_available(int arch) const noexcept override
    {
        return Kernel::is_available(arch) && available_(arch);
    }

    bool is_feasible(const GemmDesc& desc) const noexcept override
    {
        if (desc.family && desc.family != family().id) {
            return false;
        }
        constexpr std::tuple flat3{Striding::kFlat, Striding::kFlat, Striding::kFlat};

        if (std::tie(desc.striding_a, desc.striding_b, desc.striding_c) != flat3) {
            return false;
        }
        if (std::tie(desc.pack_a, desc.pack_b, desc.pack_u, desc.pack_v) != std::tuple{0, 0, 0, 0}) {
            return false;
        }
        if (desc.epilogue != Epilogue::kNone) {
            return false;
        }
        if (desc.num > 1) {
            return false;
        }
        if (desc.quant_a || desc.quant_b) {
            return false;
        }
        if (desc.group_axis >= 0) {
            return false;
        }
        if (desc.order_c != kColMajor) {
            return false;
        }
        if (desc.type_a != kHalf && desc.type_a != kBfloat16 && desc.type_a != kFloat) {
            return false;
        }
        if (desc.type_b != desc.type_a) {
            return false;
        }
        if (desc.type_c != desc.type_a && desc.type_c != kFloat) {
            return false;
        }
        return true;
    }

    int GetMaxSwizzle(const int4&) const override
    {
        return 0;
    }

    int GetMaxSplits(const int4&, int, size_t, size_t) const override
    {
        return 1;
    }

private:
    cublasHandle_t cublas_{};
    cudaStream_t   stream_{};
    void*          workspace_{};
    size_t         workspace_size_{};
    bool (*available_)(int);
};

void add_cublas(Collector& collector, bool (*available)(int))
{
    add<CublasKernel>(collector, available);
}

namespace {
bool bf16_available(int arch)
{
    return arch >= Sm80::value;
}

template<DataType Dtype>
std::optional<WeightBridge> supports(const DataFormat& format, bool)
{
    return format == DataFormat{Dtype} ? std::optional{WeightBridge{}} : std::nullopt;
}

template<DataType Dtype>
void pack(LinearWeight& linear, cudaStream_t)
{
    TM_CHECK_EQ(linear.weight.dtype(), Dtype);
    linear.k_desc =
        MatrixLayout{Dtype, kRowMajor, linear.input_dim, linear.output_dim, linear.output_dim, 0, 0, nullptr, nullptr};
    linear.q_desc = {};
}

// Priorities above every native dense FP16 (190) / BF16 (200, 250) family so dense
// GEMMs route to cuBLAS at weight-plan time.
const Family dense_f16{100, 260, kHalf, kHalf, 1, 1, 1, 1, false, false, supports<kHalf>, pack<kHalf>};
const Family dense_bf16{101, 260, kBfloat16, kBfloat16, 1, 1, 1, 1, false, false, supports<kBfloat16>, pack<kBfloat16>};
const Family f16_f32{104, 90, kHalf, kFloat, 1, 1, 1, 1, false, false, supports<kHalf>, pack<kHalf>};
const Family bf16_f32{105, 100, kBfloat16, kFloat, 1, 1, 1, 1, false, false, supports<kBfloat16>, pack<kBfloat16>};

Registrar reg[]{
    {dense_f16, [](Collector& c) { add_cublas(c); }},
    {dense_bf16, [](Collector& c) { add_cublas(c, bf16_available); }},
    {f16_f32, [](Collector& c) { add_cublas(c); }},
    {bf16_f32, [](Collector& c) { add_cublas(c, bf16_available); }},
};

}  // namespace

}  // namespace turbomind::gemm
