#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "src/turbomind/core/cuda_data_type.h"
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cstdio>
#include <vector>

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

#if defined(ENABLE_CUBLAS_GROUPED)

// Grouped GEMM via cublasGemmGroupedBatchedEx (CUDA 12.5+, SM100).
// Requires standard (K,N) row-major weight; GetConverters skips tiled conversion on SM100 for grouped BF16.
// Problem (row-major): D_i = A_i * B_i^T  with A_i (M_i,K), B_i (N,K), D_i (M_i,N).
// cuBLAS (col-major):  C = alpha*op(A)*op(B) + beta*C.
// Map: C_cublas = D^T (N,M_i), A_cublas = B_i (N,K), B_cublas = A_i as (K,M_i) column-major (same bytes as row-major
// input).
//      C = A*B = weight * input^T = D_i^T.  transa=N, transb=N, ldb=K.
// Per group: m=N, n=M_i, k=K; lda=N, ldb=K, ldc=N. A/B/C arrays are device ptrs to each group's base.

class CublasGroupedKernel: public Kernel {
public:
    explicit CublasGroupedKernel(): cublas_{}
    {
        cublasCreate(&cublas_);
        cublasSetWorkspace(cublas_, nullptr, 0);
        cublasSetMathMode(cublas_, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);  // match Reference::gemm

        desc_.backend    = 1;
        desc_.group_axis = 0;  // batch dim along M (ragged M per group)
        desc_.arch       = 1000;
        desc_.order_a    = kRowMajor;
        desc_.order_b    = kColMajor;
        desc_.order_c    = kRowMajor;
        desc_.type_a     = turbomind::kBfloat16;  // match MoE; Half also supported in is_feasible
        desc_.type_b     = turbomind::kBfloat16;
        desc_.type_c     = turbomind::kBfloat16;
        desc_.striding_a = Striding::kIndexed;
        desc_.striding_b = Striding::kBlocked;
        desc_.striding_c = Striding::kBlocked;
        desc_.align      = {1, 1, 1};
        desc_.cta_tile   = {256, 256, 1};  // batch_dim uses .x when group_axis=0; allow batch_size up to 256

        info_.chunk_size_k      = 1;
        info_.dynamic_smem_size = 0;
        info_.name              = GetName();
    }

    ~CublasGroupedKernel() override
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
        if (!Adesc.offsets || !Ddesc.offsets || Adesc.offsets == reinterpret_cast<int*>(1)
            || Ddesc.offsets == reinterpret_cast<int*>(1)) {
            fprintf(
                stderr,
                "[TM][GEMM] CublasGrouped: missing or invalid offsets (Adesc.offsets=%p Ddesc.offsets=%p) num=%d rows=%d\n",
                (void*)Adesc.offsets,
                (void*)Ddesc.offsets,
                Adesc.num,
                Adesc.rows);
            return 1;
        }
        const int group_count = Adesc.num;
        if (group_count <= 0 || Bdesc.num != group_count || Ddesc.num != group_count) {
            fprintf(stderr,
                    "[TM][GEMM] CublasGrouped: group/num mismatch group_count=%d Bdesc.num=%d Ddesc.num=%d\n",
                    group_count,
                    Bdesc.num,
                    Ddesc.num);
            return 1;
        }
        (void)cudaGetLastError();

        if (stream_ != stream) {
            cublasSetStream(cublas_, stream);
            stream_ = stream;
        }

        if (workspace_ != workspace.partials || workspace_size_ != workspace.partials_size) {
            cublasSetWorkspace(cublas_, workspace.partials, workspace.partials_size);
            workspace_      = workspace.partials;
            workspace_size_ = workspace.partials_size;
        }

        const bool weight_is_strided_ptrs = (Bdesc.ld == 0);

        std::vector<int>        host_offsets;
        std::vector<StridedPtr> host_strided;
        const int*              ptr_offsets   = Adesc.offsets;
        bool                    need_d2h_sync = false;

        {
            cudaPointerAttributes attr{};
            if (cudaPointerGetAttributes(&attr, (void*)Adesc.offsets) == cudaSuccess
                && attr.type == cudaMemoryTypeDevice) {
                host_offsets.resize(group_count + 1);
                cudaMemcpyAsync(host_offsets.data(),
                                Adesc.offsets,
                                (group_count + 1) * sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream);
                need_d2h_sync = true;
            }
        }

        if (weight_is_strided_ptrs) {
            host_strided.resize(group_count);
            cudaMemcpyAsync(host_strided.data(), B, group_count * sizeof(StridedPtr), cudaMemcpyDeviceToHost, stream);
            need_d2h_sync = true;
        }

        if (need_d2h_sync) {
            if (cudaStreamSynchronize(stream) != cudaSuccess) {
                fprintf(
                    stderr, "[TM][GEMM] CublasGrouped: D2H sync failed: %s\n", cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
        }

        if (!host_offsets.empty()) {
            ptr_offsets = host_offsets.data();
        }

        const cudaDataType cuda_type = turbomind::to_cuda_dtype(Adesc.type);
        const size_t       elem_size = turbomind::byte_size(Adesc.type, 1);

        const int N = Bdesc.cols;
        const int K = Adesc.cols;

        if (ptr_offsets[group_count] != Adesc.rows) {
            fprintf(stderr,
                    "[TM][GEMM] CublasGrouped: offsets[%d]=%d != Adesc.rows=%d (would OOB)\n",
                    group_count,
                    ptr_offsets[group_count],
                    Adesc.rows);
            return 1;
        }
        if (Adesc.ld < K || Ddesc.ld < N) {
            fprintf(stderr,
                    "[TM][GEMM] CublasGrouped: Adesc.ld=%d (need >= K=%d) or Ddesc.ld=%d (need >= N=%d)\n",
                    Adesc.ld,
                    K,
                    Ddesc.ld,
                    N);
            return 1;
        }
        const void* const kBadAddr = reinterpret_cast<void*>(0x320936400ULL);

        std::vector<int>         n_active;
        std::vector<int>         lda_active;
        std::vector<const void*> a_active;
        std::vector<const void*> b_active;
        std::vector<void*>       c_active;
        n_active.reserve(group_count);
        lda_active.reserve(group_count);
        a_active.reserve(group_count);
        b_active.reserve(group_count);
        c_active.reserve(group_count);

        for (int i = 0; i < group_count; ++i) {
            const int M_i = ptr_offsets[i + 1] - ptr_offsets[i];
            if (M_i <= 0)
                continue;

            const void* weight_ptr;
            if (weight_is_strided_ptrs) {
                weight_ptr = host_strided[i].ptr;
                if (!weight_ptr || weight_ptr == kBadAddr) {
                    fprintf(stderr, "[TM][GEMM] CublasGrouped: weight ptr[%d]=%p (null or sentinel)\n", i, weight_ptr);
                    return 1;
                }
            }
            else {
                const int off_b = Bdesc.offsets ? Bdesc.offsets[i] * Bdesc.ld : i * (K * N);
                weight_ptr      = static_cast<const char*>(B) + off_b * elem_size;
            }

            n_active.push_back(M_i);
            lda_active.push_back(N);
            a_active.push_back(weight_ptr);
            b_active.push_back(static_cast<const char*>(A) + ptr_offsets[i] * Adesc.ld * elem_size);
            c_active.push_back(static_cast<char*>(D) + ptr_offsets[i] * Ddesc.ld * elem_size);
        }

        const int active_count = (int)n_active.size();
        if (active_count == 0) {
            return 0;
        }

        std::vector<cublasOperation_t> transa_array(active_count, CUBLAS_OP_N);
        std::vector<cublasOperation_t> transb_array(active_count, CUBLAS_OP_N);
        std::vector<int>               m_array(active_count, N);
        std::vector<int>               k_array(active_count, K);
        std::vector<int>               ldb_array(active_count, Adesc.ld);
        std::vector<int>               ldc_array(active_count, Ddesc.ld);
        std::vector<float>             alpha_array(active_count, alpha);
        std::vector<float>             beta_array(active_count, beta);
        std::vector<int>               group_size(active_count, 1);

        // Use pre-allocated workspace for device pointer arrays (no cudaMalloc/Free per call)
        const size_t one_array   = active_count * sizeof(void*);
        const size_t total_bytes = 3 * one_array;
        TM_CHECK_LE(total_bytes, workspace.tensormaps_size);

        char* d_ptrs = static_cast<char*>(workspace.tensormaps);
        cudaMemcpyAsync(d_ptrs, a_active.data(), one_array, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ptrs + one_array, b_active.data(), one_array, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ptrs + 2 * one_array, c_active.data(), one_array, cudaMemcpyHostToDevice, stream);

        // Stream ordering guarantees the H2D copies complete before cuBLAS reads the pointers.
        cublasStatus_t status = cublasGemmGroupedBatchedEx(cublas_,
                                                           transa_array.data(),
                                                           transb_array.data(),
                                                           m_array.data(),
                                                           n_active.data(),
                                                           k_array.data(),
                                                           alpha_array.data(),
                                                           reinterpret_cast<const void* const*>(d_ptrs),
                                                           cuda_type,
                                                           lda_active.data(),
                                                           reinterpret_cast<const void* const*>(d_ptrs + one_array),
                                                           cuda_type,
                                                           ldb_array.data(),
                                                           beta_array.data(),
                                                           reinterpret_cast<void* const*>(d_ptrs + 2 * one_array),
                                                           cuda_type,
                                                           ldc_array.data(),
                                                           active_count,
                                                           group_size.data(),
                                                           CUBLAS_COMPUTE_32F);

        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(
                stderr, "[TM][GEMM] CublasGrouped: cublasGemmGroupedBatchedEx failed: %s\n", _cudaGetErrorEnum(status));
            return 1;
        }
        return 0;
    }

    bool is_feasible(const GemmDesc& desc) const noexcept override
    {
        if (desc.num <= 1 || desc.group_axis < 0) {
            return false;
        }
        // Reject group_axis=1 (transposed): TransposedKernel swaps A and B, so CublasGroupedKernel would receive
        // weight descriptor as Adesc; weight has no valid offsets -> Adesc.offsets=(nil) and Launch fails.
        if (desc.group_axis != 0) {
            return false;
        }
        if (!is_arch_compatible(desc_.arch, desc.arch)) {
            return false;
        }
        if (desc.striding_a != Striding::kBlocked && desc.striding_a != Striding::kIndexed) {
            return false;
        }
        if (desc.striding_c != Striding::kBlocked && desc.striding_c != Striding::kIndexed) {
            return false;
        }
        if (desc.striding_b != Striding::kFlat && desc.striding_b != Striding::kBlocked) {
            return false;
        }
        // Allow any epilogue; Launch does plain GEMM, caller applies epilogue if needed
        if (desc.quant_a || desc.quant_b) {
            return false;
        }
        if (desc.type_a != kHalf && desc.type_a != kBfloat16) {
            return false;
        }
        if (desc.type_b != desc.type_a || desc.type_c != desc.type_a) {
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

void Registry::sm100_cublas_grouped_float()
{
    Add(std::make_unique<CublasGroupedKernel>());
}

#endif  // ENABLE_CUBLAS_GROUPED

}  // namespace turbomind::gemm
