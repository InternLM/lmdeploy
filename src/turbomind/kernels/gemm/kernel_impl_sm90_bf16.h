// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstring>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

#include "src/turbomind/kernels/gemm/tma.h"

#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::gemm {

extern __shared__ char smem_buf[];

// Distinct symbol from gemm_kernel_sm90 (FP8) to avoid ODR clash.
template<class Kernel>
__global__ void __launch_bounds__(Kernel::CTA_SIZE, 1) gemm_kernel_sm90_bf16(const __grid_constant__ CUtensorMap tm_a,
                                                                             const __grid_constant__ CUtensorMap tm_b,
                                                                             const __grid_constant__ CUtensorMap tm_c,
                                                                             const __grid_constant__ CUtensorMap tm_u,
                                                                             const __grid_constant__ CUtensorMap tm_v,
                                                                             const MatrixParam          param_A,
                                                                             const MatrixParam          param_B,
                                                                             const MatrixParam          param_U,
                                                                             const MatrixParam          param_V,
                                                                             const MatrixParam          param_C,
                                                                             bool                       fuse_silu,
                                                                             typename Kernel::Scheduler sched,
                                                                             void*                      tensormap_buf)
{

#if __CUDA_ARCH__
    if constexpr (Kernel::Arch::is_compatible(__CUDA_ARCH__)) {
        Kernel kernel;
        kernel(tm_a,
               tm_b,
               tm_c,
               tm_u,
               tm_v,
               param_A,
               param_B,
               param_U,
               param_V,
               param_C,
               fuse_silu,
               sched,
               (CUtensorMap*)tensormap_buf,
               smem_buf);
    }
#endif
}

template<class Gemm>
class KernelImplSm90Bf16: public Kernel {
public:
    // import frequently used constants
    static constexpr int TILE_M = Gemm::TILE_M;
    static constexpr int TILE_N = Gemm::TILE_N;
    static constexpr int TILE_K = Gemm::TILE_K;

    static constexpr auto is_grouped_gemm = Gemm::is_grouped_gemm;

    struct AlgoBits {
        uint32_t family : 8;
        uint32_t math_wgs : 8;
        uint32_t : 16;

        uint32_t u32() const
        {
            static_assert(sizeof(AlgoBits) == sizeof(uint32_t));
            uint32_t v;
            std::memcpy(&v, this, sizeof(v));
            return v;
        }
    };

    KernelImplSm90Bf16()
    {
        // After SM90 BF16 prepare: B is (K, N) ColMajor view of physical (N, K).
        desc_.order_a = kRowMajor;  // A: (M, K)
        desc_.order_b = kColMajor;  // B: (K, N) col-major == (N, K) row-major storage
        desc_.order_c = kRowMajor;  // C: (M, N)

        desc_.type_a = data_type_v<typename Gemm::Ta>;
        desc_.type_b = data_type_v<typename Gemm::Tb>;
        desc_.type_c = data_type_v<typename Gemm::Tc>;

        desc_.striding_a = Gemm::kStridingA;
        desc_.striding_b = Gemm::kStridingB;
        desc_.striding_c = Gemm::kStridingC;

        desc_.pack_a = {};
        desc_.pack_b = {};
        desc_.pack_u = {};
        desc_.pack_v = {};

        // BF16 path: no FP8 scales / quant metadata
        desc_.quant_a = {};
        desc_.quant_b = {};

        desc_.cta_tile            = {TILE_M, TILE_N, TILE_K};
        desc_.mma_tile            = {1, 1, 1};
        desc_.atom_layout         = {cute::size<0>(typename Gemm::AtomLayoutMNK{}),
                             cute::size<1>(typename Gemm::AtomLayoutMNK{}),
                             cute::size<2>(typename Gemm::AtomLayoutMNK{})};
        desc_.supports_fused_silu = Gemm::kSupportsFusedSilu;

        info_.chunk_size_k = Gemm::TILE_K;

        desc_.align.x = 1;
        desc_.align.y = 1;
        desc_.align.z = 1;

        // desc policies mirror the actual operand cache hints: A (activations) / B (weights)
        desc_.policy_a = 0;
        desc_.policy_b = Gemm::kL2HintW;
        desc_.c_tile   = {TILE_M, TILE_N};
        desc_.op_class = OpClass::kGMMA_h64n16;
        desc_.raster   = Gemm::kRasterOrder;

        AlgoBits algo{};
        algo.family   = 1;
        algo.math_wgs = Gemm::WARPGROUPS;
        desc_.algo    = algo.u32();

        desc_.cluster_shape = {Gemm::Cluster::M, Gemm::Cluster::N};

        info_.dynamic_smem_size = Gemm::kSmemSize;

        desc_.stages     = Gemm::Stages;
        desc_.split_k    = 1;
        desc_.group_axis = is_grouped_gemm ? 0 : -1;

        desc_.arch = Gemm::Arch::value;

        auto func = gemm_kernel_sm90_bf16<Gemm>;

        cudaFuncGetAttributes(&info_.attr, func);

        if (info_.dynamic_smem_size > (48 << 10)) {
            cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, info_.dynamic_smem_size);
        }

        cudaFuncSetAttribute(func, cudaFuncAttributeNonPortableClusterSizeAllowed, 16);

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info_.max_active_ctas, func, Gemm::CTA_SIZE, info_.dynamic_smem_size);

        sm_count_ = getSMCount();

        info_.name = GetName();
    }

    int Launch(const Operation&    operation,
               float               alpha,
               const void*         A,
               const MatrixLayout& _Adesc,
               const void*         U,
               const MatrixLayout& Udesc,
               const void*         B,
               const MatrixLayout& _Bdesc,
               const void*         V,
               const MatrixLayout& _Vdesc,
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
        using Sched = typename Gemm::Scheduler;

        MatrixLayout Adesc = _Adesc;

        const int  m          = Ddesc.rows;
        const int  n          = Ddesc.cols;
        const int  k          = Adesc.cols;
        const int  num_groups = std::max(Adesc.num, 1);
        const bool fuse_silu  = ((int)operation.epilogue & (int)Epilogue::kGatedSilu) != 0;

        TM_CHECK_GE(cdiv(k, TILE_K), 2) << "The kernel requires at least 2 k-tiles to work";

        auto transpose = [](MatrixLayout x) {
            std::swap(x.rows, x.cols);
            x.order = gemm::transpose(x.order);
            return x;
        };

        // (K, N) ColMajor → (N, K) RowMajor for TMA (K-contiguous, SW128)
        MatrixLayout Bdesc = transpose(_Bdesc);
        MatrixLayout Vdesc = transpose(_Vdesc);

        auto sched = [&] {
            const int2 tiles = get_tiled_shape(m, n, TILE_M, TILE_N);
            const int4 shape{m, n, k, num_groups};

            swizzle = Sched::get_log_tile(tiles, 1 << swizzle);

            Sched sched{};
            sched.init(shape, swizzle, {TILE_M, TILE_N, TILE_K});

            sched.next_cluster_id_ = TM_CHECK_NOTNULL(workspace.flags);

            sched.offsets_ = nullptr;

            return sched;
        }();

        constexpr int kMulticastA = Gemm::kMulticastA;
        constexpr int kMulticastB = Gemm::kMulticastB;

        constexpr int kTileM = Gemm::TILE_M;
        constexpr int kTileN = Gemm::TILE_N;

        if (Gemm::Scheduler::is_dynamic) {
            TM_CUDA_CHECK(cudaMemsetAsync(workspace.flags, 0, sizeof(int), stream));
        }

        // Indexed A: no static full-A TMA; Flat/Blocked use static TMA templates.
        // Grouped: the prepare kernel rebases TMA maps and materializes scheduler offsets.
        auto tm_a = make_2d_tma_desc(Gemm::kStridingA == Striding::kIndexed ? nullptr : (void*)A,
                                     Adesc,
                                     {kTileM / kMulticastA, TILE_K},
                                     CU_TENSOR_MAP_SWIZZLE_128B);

        // Grouped B: nullptr template + prepare-kernel rebase.
        auto tm_b = make_2d_tma_desc(
            is_grouped_gemm ? nullptr : (void*)B, Bdesc, {kTileN / kMulticastB, TILE_K}, CU_TENSOR_MAP_SWIZZLE_128B);

        using LayoutC = typename Gemm::LayoutC;
        // Fused SiLU: scheduler walks full weight N; C TMA / output are half-width.
        // LlamaLinear: Ddesc.cols = weight.output_dim (full), ld = output.stride (half).
        MatrixLayout Cdesc_tma = Cdesc;
        if (fuse_silu) {
            TM_CHECK_EQ(Cdesc_tma.cols % 2, 0);
            Cdesc_tma.cols /= 2;
        }
        auto tm_c = make_2d_tma_desc((void*)C, Cdesc_tma, {LayoutC::S0, LayoutC::C0}, get_tma_swizzle(Gemm::kSwizzleC));

        // BF16: no scale tensors
        CUtensorMap tm_u{};
        CUtensorMap tm_v{};

        const auto param_A = to_param((void*)A, Adesc);
        const auto param_B = to_param((void*)B, Bdesc);
        const auto param_U = to_param((void*)U, Udesc);
        const auto param_V = to_param((void*)V, Vdesc);
        const auto param_C = to_param((void*)D, Ddesc);

        if constexpr (is_grouped_gemm) {
            sched.offsets_ = Gemm::PrepareTmaDescs(tm_a,
                                                   tm_b,
                                                   tm_c,
                                                   param_A,
                                                   param_B,
                                                   param_C,
                                                   (CUtensorMap*)workspace.tensormaps,
                                                   num_groups,
                                                   m,
                                                   n,
                                                   stream);
            TM_CUDA_CHECK(cudaGetLastError());
        }

        const int sm_count = sm_count_;

        static constexpr int cluster_size = Gemm::kClusterSize;

        // Persistent scheduler: fill all co-resident CTAs (small tiles reach
        // max_active_ctas > 1 per SM); clamped to the true co-resident count below.
        auto       grid  = sm_count * info_.max_active_ctas / cluster_size * cluster_size;
        const auto block = Gemm::CTA_SIZE;

        cudaLaunchConfig_t config{};
        config.gridDim          = grid;
        config.blockDim         = block;
        config.dynamicSmemBytes = info_.dynamic_smem_size;
        config.stream           = stream;

        auto func = gemm_kernel_sm90_bf16<Gemm>;

        [[maybe_unused]] static bool _ = [&] {
            int max_cluster_size = 0;
            cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, func, &config);
            return false;
        }();

        cudaLaunchAttribute attrs[1];

        attrs[0].id               = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = cluster_size;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;

        config.attrs    = attrs;
        config.numAttrs = std::size(attrs);

        int max_active_cluster{};
        cudaOccupancyMaxActiveClusters(&max_active_cluster, func, &config);
        config.gridDim = std::min<int>(config.gridDim.x, max_active_cluster * cluster_size);

        auto ec = cudaLaunchKernelEx(&config,
                                     func,
                                     tm_a,
                                     tm_b,
                                     tm_c,
                                     tm_u,
                                     tm_v,
                                     param_A,
                                     param_B,
                                     param_U,
                                     param_V,
                                     param_C,
                                     fuse_silu,
                                     sched,
                                     workspace.tensormaps);
        TM_CUDA_CHECK(ec);

        return 0;
    }

    std::array<size_t, 2> GetWorkspaceSize(int tiles, int splits) const
    {
        static constexpr bool kSerial = true;

        size_t barriers_size = sizeof(int) * tiles;
        size_t partials_size = sizeof(float) * TILE_M * TILE_N * tiles;

        if constexpr (!kSerial) {
            barriers_size *= splits;
            partials_size *= splits;
        }

        return {barriers_size, partials_size};
    }

    int GetMaxSplits(const int4& shape, int swizzle, size_t bsize, size_t psize) const override
    {
        return 1;
    }

    int GetMaxSwizzle(const int4& shape) const override
    {
        using Map        = typename Gemm::Scheduler;
        const auto tiles = get_tiled_shape(shape.x, shape.y, TILE_M, TILE_N);
        return Map::get_log_tile(tiles, 1 << 10);
    }

    bool is_feasible(const GemmDesc& desc) const noexcept override
    {
        const bool want_fused = ((int)desc.epilogue & (int)Epilogue::kGatedSilu) != 0;
        if (want_fused && !Gemm::kSupportsFusedSilu) {
            return false;
        }
        return Kernel::is_feasible(desc);
    }

private:
    int sm_count_ = 0;
};

}  // namespace turbomind::gemm
