// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstring>
#include <numeric>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/kernels/gemm/tma.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::gemm {

extern __shared__ __align__(1024) char smem_buf[];

template<class Kernel>
__global__ void __launch_bounds__(Kernel::CTA_SIZE, 1) gemm_kernel_sm90_mixed(const __grid_constant__ CUtensorMap tm_a,
                                                                              const __grid_constant__ typename Kernel::TmaPacked tm_b,
                                                                              const __grid_constant__ typename Kernel::TmaQparam tm_v,
                                                                              const __grid_constant__ CUtensorMap tm_c,
                                                                              const MatrixParam          param_A,
                                                                              const MatrixParam          param_G,
                                                                              const MatrixParam          param_C,
                                                                              bool                       fuse_silu,
                                                                              typename Kernel::Scheduler sched,
                                                                              void*                      tensormap_buf)
{
#if __CUDA_ARCH__
    if constexpr (Kernel::Arch::is_compatible(__CUDA_ARCH__)) {
        Kernel kernel;
        kernel(
            tm_a, tm_b, tm_v, tm_c, param_A, param_G, param_C, fuse_silu, sched, (CUtensorMap*)tensormap_buf, smem_buf);
    }
#endif
}

template<class Gemm>
class KernelImplSm90Mixed: public Kernel {
public:
    static constexpr int TILE_M = Gemm::TILE_M;
    static constexpr int TILE_N = Gemm::TILE_N;
    static constexpr int TILE_K = Gemm::TILE_K;

    static constexpr bool is_grouped_gemm = Gemm::is_grouped_gemm;

    struct AlgoBits {
        uint32_t family : 8;
        uint32_t math_wgs : 8;
        uint32_t : 16;

        uint32_t u32() const
        {
            static_assert(sizeof(AlgoBits) == sizeof(uint32_t));
            uint32_t value;
            std::memcpy(&value, this, sizeof(value));
            return value;
        }
    };

    KernelImplSm90Mixed()
    {
        // Direct LlamaLinear API.  Internally the mainloop swaps operands so
        // packed B is WGMMA RS operand A.
        desc_.order_a = kRowMajor;
        desc_.order_b = Gemm::Format::kPublicWeightOrder;
        desc_.order_c = kRowMajor;

        desc_.type_a = data_type_v<typename Gemm::Ta>;
        desc_.type_b = data_type_v<typename Gemm::Tb>;
        desc_.type_c = data_type_v<typename Gemm::Tc>;

        desc_.striding_a = Gemm::kStridingA;
        desc_.striding_b = Gemm::kStridingB;
        desc_.striding_c = Gemm::kStridingC;

        static_assert(Gemm::Format::kWeightPack == (GMMA_64x16_RS | OPERAND_A | 1));
        static_assert(get_mma_tag(Gemm::Format::kQparamPack) == GMMA_64x16_RS);
        static_assert(get_operand_tag(Gemm::Format::kQparamPack) == OPERAND_U);
        desc_.pack_a = {};
        desc_.pack_b = Gemm::Format::kWeightPack;
        desc_.pack_u = {};
        desc_.pack_v = Gemm::Format::kQparamPack;

        desc_.quant_a = {};
        desc_.quant_b = QuantDesc{Gemm::Format::kQuantType, Gemm::kGroupSize};

        desc_.cta_tile    = {TILE_M, TILE_N, TILE_K};
        desc_.mma_tile    = {64, TILE_M, 16};
        desc_.atom_layout = {cute::size<0>(typename Gemm::AtomLayoutMNK{}),
                             cute::size<1>(typename Gemm::AtomLayoutMNK{}),
                             cute::size<2>(typename Gemm::AtomLayoutMNK{})};

        // The packed format requires complete OUT128 tiles and complete K64 /
        // quant-group units. Batch tails are zero-filled/clipped by TMA or
        // indexed cp.async. Launch separately enforces at least two K stages.
        desc_.align = {1, TILE_N, std::lcm(TILE_K, Gemm::kGroupSize)};

        desc_.policy_a = 0;
        desc_.policy_b = 0;
        desc_.c_tile   = {TILE_M, TILE_N};
        desc_.op_class = OpClass::kGMMA_h64n16;
        desc_.raster   = Gemm::kRasterOrder;

        AlgoBits algo{};
        algo.family   = 2;
        algo.math_wgs = Gemm::WARPGROUPS;
        desc_.algo    = algo.u32();

        desc_.cluster_shape       = {Gemm::Cluster::M, Gemm::Cluster::N};
        desc_.stages              = Gemm::Stages;
        desc_.split_k             = 1;
        desc_.supports_fused_silu = Gemm::kSupportsFusedSilu;
        desc_.group_axis          = is_grouped_gemm ? 0 : -1;
        desc_.arch                = Gemm::Arch::value;

        info_.chunk_size_k      = TILE_K;
        info_.dynamic_smem_size = Gemm::kSmemSize;

        auto func = gemm_kernel_sm90_mixed<Gemm>;
        cudaFuncGetAttributes(&info_.attr, func);
        if (info_.dynamic_smem_size > (48 << 10)) {
            cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, info_.dynamic_smem_size);
        }
        cudaFuncSetAttribute(func, cudaFuncAttributeNonPortableClusterSizeAllowed, 16);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info_.max_active_ctas, func, Gemm::CTA_SIZE, info_.dynamic_smem_size);

        sm_count_  = getSMCount();
        info_.name = GetName();
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
        (void)U;
        (void)Udesc;
        (void)C;
        (void)Cdesc;
        (void)W;
        (void)Wdesc;
        (void)splits;

        using Sched = typename Gemm::Scheduler;

        const int  m          = Ddesc.rows;
        const int  n          = Ddesc.cols;
        const int  k          = Adesc.cols;
        const int  num_groups = std::max(Adesc.num, 1);
        const bool fuse_silu  = ((int)operation.epilogue & (int)Epilogue::kGatedSilu) != 0;

        TM_CHECK(!fuse_silu || Gemm::kSupportsFusedSilu);
        TM_CHECK_NOTNULL(A);
        TM_CHECK_NOTNULL(B);
        TM_CHECK_NOTNULL(V);
        TM_CHECK_NOTNULL(D);
        TM_CHECK_EQ(Bdesc.type, data_type_v<typename Gemm::Tb>);
        TM_CHECK_EQ(Vdesc.type, data_type_v<typename Gemm::Tv>);
        TM_CHECK_EQ((int)operation.epilogue & ~(int)Epilogue::kGatedSilu, 0);
        TM_CHECK_EQ(alpha, 1.f);
        TM_CHECK_EQ(beta, 0.f);
        TM_CHECK_EQ(Adesc.rows, m);
        TM_CHECK_EQ(Bdesc.rows, k);
        TM_CHECK_EQ(Bdesc.cols, n);
        TM_CHECK_EQ(Vdesc.rows, k / Gemm::kGroupSize);
        TM_CHECK_EQ(Vdesc.cols, n / Gemm::Format::kScaleGroupN);
        TM_CHECK_EQ(Vdesc.ld % Gemm::Format::kQparamValuesFragment, 0);
        TM_CHECK_EQ(std::max(Bdesc.num, 1), std::max(Vdesc.num, 1));
        if constexpr (Gemm::Format::kHasGlobalScale) {
            TM_CHECK_NOTNULL(global_scale);
            TM_CHECK_EQ(global_scale_desc.type, kFloat);
            TM_CHECK_EQ(global_scale_desc.rows, 1);
            TM_CHECK_EQ(global_scale_desc.cols, 1);
            TM_CHECK_EQ(std::max(global_scale_desc.num, 1), std::max(Bdesc.num, 1));
        }
        if constexpr (is_grouped_gemm) {
            TM_CHECK_EQ(std::max(Bdesc.num, 1), num_groups);
            TM_CHECK_EQ(std::max(Vdesc.num, 1), num_groups);
            TM_CHECK_EQ(std::max(Ddesc.num, 1), num_groups);

            // A grouped implementation may still be measured for an entirely
            // flat, single-GEMM request. Otherwise packed B/V are nonlinear
            // byte streams addressed through one StridedPtr per expert.
            // Offsets/idxs and a shared pitched base have no well-defined
            // packed-row interpretation.
            const bool flat_single = num_groups == 1 && get_mode(Adesc) == Striding::kFlat
                                     && get_mode(Bdesc) == Striding::kFlat && get_mode(Vdesc) == Striding::kFlat
                                     && get_mode(Ddesc) == Striding::kFlat;
            if (!flat_single) {
                TM_CHECK_EQ(Bdesc.ld, 0);
                TM_CHECK_EQ(Vdesc.ld, 0);
                TM_CHECK(Bdesc.offsets == nullptr);
                TM_CHECK(Vdesc.offsets == nullptr);
                TM_CHECK(Bdesc.idxs == nullptr);
                TM_CHECK(Vdesc.idxs == nullptr);
                if constexpr (Gemm::Format::kHasGlobalScale) {
                    TM_CHECK_EQ(global_scale_desc.ld, 0);
                    TM_CHECK(global_scale_desc.offsets == nullptr);
                    TM_CHECK(global_scale_desc.idxs == nullptr);
                }
            }
        }
        else {
            TM_CHECK_EQ(num_groups, 1);
            TM_CHECK_EQ(std::max(Bdesc.num, 1), 1);
            TM_CHECK_EQ(std::max(Vdesc.num, 1), 1);
        }
        TM_CHECK_EQ(n % TILE_N, 0);
        TM_CHECK_EQ(k % Gemm::kGroupSize, 0);
        TM_CHECK_EQ(k % TILE_K, 0);
        TM_CHECK_GE(k / TILE_K, 2);

        auto sched = [&] {
            const int2 tiles = get_tiled_shape(m, n, TILE_M, TILE_N);
            const int4 shape{m, n, k, num_groups};
            swizzle = Sched::get_log_tile(tiles, 1 << swizzle);

            Sched result{};
            result.init(shape, swizzle, {TILE_M, TILE_N, TILE_K});
            result.next_cluster_id_ = TM_CHECK_NOTNULL(workspace.flags);
            result.offsets_         = nullptr;
            return result;
        }();

        if (Sched::is_dynamic) {
            TM_CUDA_CHECK(cudaMemsetAsync(workspace.flags, 0, sizeof(int), stream));
        }

        auto tm_a = make_2d_tma_desc(Gemm::kStridingA == Striding::kIndexed ? nullptr : (void*)A,
                                     Adesc,
                                     {Gemm::kTmaBoxM, TILE_K},
                                     CU_TENSOR_MAP_SWIZZLE_128B);
        auto tm_b = Gemm::MakeTmaPacked(Gemm::is_grouped_gemm ? nullptr : (void*)B, n, k);
        auto tm_v = Gemm::MakeTmaQparam(Gemm::is_grouped_gemm ? nullptr : (void*)V, n, k);

        MatrixLayout Cdesc_tma = Ddesc;
        if (fuse_silu) {
            TM_CHECK_EQ(Cdesc_tma.cols % 2, 0);
            Cdesc_tma.cols /= 2;
        }
        auto tm_c = make_2d_tma_desc((void*)D, Cdesc_tma, {Gemm::kTmaStoreM, Gemm::kTmaStoreN}, get_tma_swizzle(Gemm::kSwizzleC));

        const auto param_A = to_param((void*)A, Adesc);
        const auto param_B = to_param((void*)B, Bdesc);
        const auto param_V = to_param((void*)V, Vdesc);
        const auto param_G = to_param((void*)global_scale, global_scale_desc);
        const auto param_C = to_param((void*)D, Ddesc);

        if constexpr (is_grouped_gemm) {
            const size_t tma_workspace_bytes = (size_t)num_groups * Gemm::kTmaDescNum * sizeof(CUtensorMap)
                                               + (size_t)(num_groups + 1) * sizeof(int);
            TM_CHECK_LE(tma_workspace_bytes, workspace.tensormaps_size);
            sched.offsets_ = Gemm::PrepareTmaDescs(tm_a,
                                                   *tm_b.get_tma_descriptor(),
                                                   *tm_v.get_tma_descriptor(),
                                                   tm_c,
                                                   param_A,
                                                   param_B,
                                                   param_V,
                                                   param_C,
                                                   (CUtensorMap*)workspace.tensormaps,
                                                   num_groups,
                                                   m,
                                                   stream);
            TM_CUDA_CHECK(cudaGetLastError());
        }

        constexpr int cluster_size = Gemm::kClusterSize;
        int grid = sm_count_ * info_.max_active_ctas / cluster_size * cluster_size;

        cudaLaunchConfig_t config{};
        config.gridDim          = grid;
        config.blockDim         = Gemm::CTA_SIZE;
        config.dynamicSmemBytes = info_.dynamic_smem_size;
        config.stream           = stream;

        auto                func = gemm_kernel_sm90_mixed<Gemm>;
        cudaLaunchAttribute attrs[1];
        attrs[0].id               = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = cluster_size;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        config.attrs              = attrs;
        config.numAttrs           = std::size(attrs);

        int max_active_cluster{};
        cudaOccupancyMaxActiveClusters(&max_active_cluster, func, &config);
        config.gridDim = std::min<int>(config.gridDim.x, max_active_cluster * cluster_size);

        auto ec = cudaLaunchKernelEx(
            &config, func, tm_a, tm_b, tm_v, tm_c, param_A, param_G, param_C, fuse_silu, sched, workspace.tensormaps);
        TM_CUDA_CHECK(ec);
        return 0;
    }

    int GetMaxSplits(const int4&, int, size_t, size_t) const override
    {
        return 1;
    }

    int GetMaxSwizzle(const int4& shape) const override
    {
        const auto tiles = get_tiled_shape(shape.x, shape.y, TILE_M, TILE_N);
        return Gemm::Scheduler::get_log_tile(tiles, 1 << 10);
    }

    bool is_feasible(const GemmDesc& desc) const noexcept override
    {
        if ((int)desc.epilogue & ~(int)Epilogue::kGatedSilu) {
            return false;
        }
        const bool want_fused = ((int)desc.epilogue & (int)Epilogue::kGatedSilu) != 0;
        if (want_fused && !Gemm::kSupportsFusedSilu) {
            return false;
        }
        return Kernel::is_feasible(desc);
    }

private:
    int sm_count_{};
};

}  // namespace turbomind::gemm
