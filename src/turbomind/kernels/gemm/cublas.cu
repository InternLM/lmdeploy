#include <cublas_v2.h>

#include "src/turbomind/core/cuda_data_type.h"
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

class CublasKernel: public Kernel {
public:
    explicit CublasKernel(): cublas_{}
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
               float               beta,
               const void*         C,
               const MatrixLayout& Cdesc,
               void*               D,
               const MatrixLayout& Ddesc,
               int                 swizzle,
               int                 splits,
               Workspace&          workspace,
               cudaStream_t        stream) override
    {
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

    bool is_feasible(const GemmDesc& desc) const noexcept override
    {
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
};

void Registry::cublas_float()
{
    Add(std::make_unique<CublasKernel>());
}

}  // namespace turbomind::gemm
