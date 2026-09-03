// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <cstring>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/kernels/gemm/tma.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::gemm {

extern __shared__ __align__(1024) char smem_buf[];

template<class Kernel>
__global__ void __launch_bounds__(Kernel::CTA_SIZE, 1)
gemm_universal_sm90_mxfp4_fp8(const __grid_constant__ typename Kernel::TmaActivation tm_a,
                              const __grid_constant__ typename Kernel::TmaPacked tm_b,
                              const __grid_constant__ typename Kernel::TmaQparam tm_v,
                              const __grid_constant__ CUtensorMap tm_u,
                              const __grid_constant__ CUtensorMap tm_c,
                              MatrixParam param_A,
                              MatrixParam param_B,
                              MatrixParam param_V,
                              MatrixParam param_U,
                              MatrixParam param_C,
                              MatrixParam param_W,
                              bool fuse_silu,
                              typename Kernel::Scheduler sched,
                              void* tensormap_buf)
{
#if __CUDA_ARCH__
    if constexpr (Kernel::Arch::is_compatible(__CUDA_ARCH__)) {
        Kernel{}(tm_a,
                 tm_b,
                 tm_v,
                 tm_u,
                 tm_c,
                 param_A,
                 param_B,
                 param_V,
                 param_U,
                 param_C,
                 param_W,
                 fuse_silu,
                 sched,
                 static_cast<CUtensorMap*>(tensormap_buf),
                 smem_buf);
    }
#endif
}

template<class Gemm>
class KernelImplSm90MxFp4Fp8: public Kernel {
public:
    static constexpr int TILE_M = Gemm::TILE_M;
    static constexpr int TILE_N = Gemm::TILE_N;
    static constexpr int TILE_K = Gemm::TILE_K;

    struct AlgoBits {
        uint32_t family : 8;
        uint32_t math_wgs : 8;
        uint32_t folded_pack : 1;
        uint32_t unfolded_pack : 1;
        uint32_t epilogue_stages : 4;
        uint32_t : 10;
        uint32_t u32() const
        {
            uint32_t value;
            static_assert(sizeof(value) == sizeof(*this));
            std::memcpy(&value, this, sizeof(value));
            return value;
        }
    };

    KernelImplSm90MxFp4Fp8()
    {
        desc_.order_a = kRowMajor;
        desc_.order_b = Gemm::Format::kPublicWeightOrder;
        desc_.order_c = kRowMajor;
        desc_.type_a = kFloat8_e4m3;
        desc_.type_b = kFloat4_e2m1;
        desc_.type_c = kBfloat16;
        desc_.striding_a = Gemm::kStridingA;
        desc_.striding_b = Gemm::kStridingB;
        desc_.striding_c = Gemm::kStridingC;
        desc_.pack_a = {};
        desc_.pack_b = Gemm::Format::kWeightPack;
        desc_.pack_u = {};
        desc_.pack_v = Gemm::Format::kQparamPack;
        desc_.quant_a = QuantDesc{QuantType::kK, 128};
        desc_.quant_b = QuantDesc{QuantType::kK, 32};
        desc_.cta_tile = {TILE_M, TILE_N, TILE_K};
        desc_.mma_tile = {64, Gemm::kOpN, 32};
        desc_.atom_layout = {Gemm::kAtomM, Gemm::kAtomN, 1};
        desc_.align = {1, 64, 128};
        desc_.op_class = OpClass::kGMMA_q64n32;
        desc_.raster = Gemm::kRasterOrder;
        AlgoBits algo{};
        algo.family = 3;
        algo.math_wgs = Gemm::kMathWarpGroups;
        algo.folded_pack = Gemm::Format::kFolded;
        algo.unfolded_pack = Gemm::Format::kUnfolded;
        if constexpr (Gemm::Format::kFolded) {
            desc_.c_tile = {Gemm::kEpilogueTileM, Gemm::kEpilogueTileN};
            algo.epilogue_stages = Gemm::kEpilogueStages;
        }
        else {
            desc_.c_tile = {TILE_M, TILE_N};
        }
        desc_.algo = algo.u32();
        desc_.policy_a = 0;
        desc_.policy_b = 0;
        desc_.cluster_shape = {Gemm::Cluster::M, Gemm::Cluster::N};
        desc_.stages = Gemm::Stages;
        desc_.split_k = 1;
        desc_.supports_fused_silu = Gemm::kSupportsFusedSilu;
        desc_.group_axis = Gemm::is_grouped_gemm ? 0 : -1;
        desc_.arch = Gemm::Arch::value;

        info_.chunk_size_k = TILE_K;
        info_.dynamic_smem_size = Gemm::kSmemSize;
        auto func = gemm_universal_sm90_mxfp4_fp8<Gemm>;
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

    int Launch(const Operation& operation,
               float alpha,
               const void* A,
               const MatrixLayout& Adesc,
               const void* U,
               const MatrixLayout& Udesc,
               const void* B,
               const MatrixLayout& Bdesc,
               const void* V,
               const MatrixLayout& Vdesc,
               const void*,
               const MatrixLayout&,
               float beta,
               const void* C,
               const MatrixLayout& Cdesc,
               void* D,
               const MatrixLayout& Ddesc,
               void* W,
               const MatrixLayout& Wdesc,
               int swizzle,
               int splits,
               Workspace& workspace,
               cudaStream_t stream) override
    {
        (void)C;
        (void)Cdesc;
        (void)splits;
        using Sched = typename Gemm::Scheduler;
        const int m = Ddesc.rows;
        const int n = Ddesc.cols;
        const int k = Adesc.cols;
        const int num_groups = std::max(Adesc.num, 1);
        const bool fuse_silu = (static_cast<int>(operation.epilogue) & static_cast<int>(Epilogue::kGatedSilu)) != 0;

        if constexpr (Gemm::Format::kUnfolded) {
            TM_CHECK(operation.epilogue == Epilogue::kNone);
            TM_CHECK(get_mode(Adesc) == Striding::kFlat);
            TM_CHECK(get_mode(Bdesc) == Striding::kFlat);
            TM_CHECK(get_mode(Vdesc) == Striding::kFlat);
            TM_CHECK(get_mode(Ddesc) == Striding::kFlat);
        }
        TM_CHECK_EQ(alpha, 1.f);
        TM_CHECK_EQ(beta, 0.f);
        TM_CHECK(!fuse_silu || Gemm::kSupportsFusedSilu);
        TM_CHECK_NOTNULL(A);
        TM_CHECK_NOTNULL(U);
        TM_CHECK_NOTNULL(B);
        TM_CHECK_NOTNULL(V);
        TM_CHECK_NOTNULL(D);
        TM_CHECK_EQ((int)operation.epilogue & ~(int)Epilogue::kGatedSilu, 0);
        TM_CHECK_EQ(Adesc.rows, m);
        TM_CHECK_EQ(Adesc.cols, k);
        TM_CHECK_EQ(Bdesc.rows, k);
        TM_CHECK_EQ(Bdesc.cols, n);
        TM_CHECK_EQ(Vdesc.rows, k / 32);
        TM_CHECK_EQ(Vdesc.cols, n);
        TM_CHECK_EQ(std::max(Bdesc.num, 1), std::max(Vdesc.num, 1));
        if constexpr (Gemm::is_grouped_gemm) {
            TM_CHECK_EQ(std::max(Bdesc.num, 1), num_groups);
            TM_CHECK_EQ(std::max(Vdesc.num, 1), num_groups);
            TM_CHECK_EQ(std::max(Ddesc.num, 1), num_groups);
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
            }
        }
        else {
            TM_CHECK_EQ(num_groups, 1);
        }
        TM_CHECK_GT(n, 0);
        TM_CHECK_EQ(n % 64, 0);
        if (fuse_silu) {
            TM_CHECK_EQ(n % 256, 0);
            TM_CHECK_NOTNULL(W);
        }
        TM_CHECK_EQ(k % 128, 0);
        TM_CHECK_GE(k / 128, 2);

        const int2 tiles{cute::ceil_div(m, TILE_M), cute::ceil_div(n, TILE_N)};
        swizzle = Sched::get_log_tile(tiles, 1 << swizzle);
        Sched sched{};
        sched.init({m, n, k, num_groups}, swizzle, {TILE_M, TILE_N, TILE_K});
        sched.next_cluster_id_ = TM_CHECK_NOTNULL(workspace.flags);
        sched.offsets_ = nullptr;
        if (Sched::is_dynamic) {
            TM_CUDA_CHECK(cudaMemsetAsync(workspace.flags, 0, sizeof(int), stream));
        }

        TM_CHECK(Adesc.order == kRowMajor);
        auto tm_a = [&] {
            if constexpr (Gemm::Format::kFolded) {
                return make_2d_tma_desc(
                    Gemm::kStridingA == Striding::kIndexed ? nullptr : const_cast<void*>(A),
                    Adesc,
                    {Gemm::kTmaBoxM, TILE_K},
                    CU_TENSOR_MAP_SWIZZLE_128B);
            }
            else {
                return Gemm::MakeTmaActivation(
                    Gemm::kStridingA == Striding::kIndexed ? nullptr : const_cast<void*>(A),
                    m,
                    k,
                    Adesc.ld ? Adesc.ld : k);
            }
        }();
        auto tm_b = Gemm::MakeTmaPacked(Gemm::is_grouped_gemm ? nullptr : const_cast<void*>(B), n, k);
        auto tm_v = Gemm::MakeTmaQparam(Gemm::is_grouped_gemm ? nullptr : const_cast<void*>(V), n, k);
        CUtensorMap tm_u{};
        if constexpr (Gemm::kStridingA != Striding::kIndexed) {
            tm_u = make_2d_tma_desc(
                const_cast<void*>(U),
                Udesc,
                {Gemm::kBoxU / Gemm::kMulticastU, 1},
                CU_TENSOR_MAP_SWIZZLE_NONE);
        }
        MatrixLayout Ddesc_tma = Ddesc;
        if (fuse_silu) {
            TM_CHECK_EQ(Ddesc_tma.cols % 2, 0);
            Ddesc_tma.cols /= 2;
        }
        auto tm_c = fuse_silu
                        ? make_2d_tma_desc(D,
                                           Ddesc_tma,
                                           {static_cast<uint32_t>(TILE_M),
                                            static_cast<uint32_t>(TILE_N / 2)},
                                           CU_TENSOR_MAP_SWIZZLE_NONE)
                        : make_2d_tma_desc(D,
                                           Ddesc_tma,
                                           {static_cast<uint32_t>(Gemm::kTmaStoreM),
                                            static_cast<uint32_t>(Gemm::kTmaStoreN)},
                                           CU_TENSOR_MAP_SWIZZLE_128B);
        const auto param_A = to_param(const_cast<void*>(A), Adesc);
        const auto param_B = to_param(const_cast<void*>(B), Bdesc);
        const auto param_V = to_param(const_cast<void*>(V), Vdesc);
        const auto param_U = to_param(const_cast<void*>(U), Udesc);
        const auto param_C = to_param(D, Ddesc);
        const auto param_W = to_param(W, Wdesc);

        if constexpr (Gemm::is_grouped_gemm) {
            const size_t tma_workspace_bytes = size_t(num_groups) * Gemm::kTmaDescNum * sizeof(CUtensorMap)
                                               + size_t(num_groups + 1) * sizeof(int);
            TM_CHECK_LE(tma_workspace_bytes, workspace.tensormaps_size);
            const CUtensorMap& tm_a_desc = [&]() -> const CUtensorMap& {
                if constexpr (Gemm::Format::kFolded) {
                    return tm_a;
                }
                else {
                    return *tm_a.get_tma_descriptor();
                }
            }();
            sched.offsets_ = Gemm::PrepareTmaDescs(tm_a_desc,
                                                   *tm_b.get_tma_descriptor(),
                                                   *tm_v.get_tma_descriptor(),
                                                   tm_u,
                                                   tm_c,
                                                   param_A,
                                                   param_B,
                                                   param_V,
                                                   param_U,
                                                   param_C,
                                                   fuse_silu,
                                                   static_cast<CUtensorMap*>(workspace.tensormaps),
                                                   num_groups,
                                                   m,
                                                   stream);
            TM_CUDA_CHECK(cudaGetLastError());
        }

        constexpr int cluster_size = Gemm::kClusterSize;
        int grid = sm_count_ * info_.max_active_ctas / cluster_size * cluster_size;
        cudaLaunchConfig_t config{};
        config.gridDim = grid;
        config.blockDim = Gemm::CTA_SIZE;
        config.dynamicSmemBytes = Gemm::kSmemSize;
        config.stream = stream;
        auto func = gemm_universal_sm90_mxfp4_fp8<Gemm>;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = cluster_size;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        config.attrs = attrs;
        config.numAttrs = 1;
        int max_active_cluster{};
        cudaOccupancyMaxActiveClusters(&max_active_cluster, func, &config);
        config.gridDim = std::min<int>(config.gridDim.x, max_active_cluster * cluster_size);
        TM_CUDA_CHECK(cudaLaunchKernelEx(&config,
                                         func,
                                         tm_a,
                                         tm_b,
                                         tm_v,
                                         tm_u,
                                         tm_c,
                                         param_A,
                                         param_B,
                                         param_V,
                                         param_U,
                                         param_C,
                                         param_W,
                                         fuse_silu,
                                         sched,
                                         workspace.tensormaps));
        return 0;
    }

    int GetMaxSplits(const int4&, int, size_t, size_t) const override
    {
        return 1;
    }

    int GetMaxSwizzle(const int4& shape) const override
    {
        const int2 tiles{cute::ceil_div(shape.x, TILE_M), cute::ceil_div(shape.y, TILE_N)};
        return Gemm::Scheduler::get_log_tile(tiles, 1 << 10);
    }

    bool is_feasible(const GemmDesc& desc) const noexcept override
    {
        if constexpr (Gemm::Format::kUnfolded) {
            if (desc.epilogue != Epilogue::kNone || desc.striding_a != Striding::kFlat
                || desc.striding_b != Striding::kFlat || desc.striding_c != Striding::kFlat) {
                return false;
            }
        }
        if ((static_cast<int>(desc.epilogue) & ~static_cast<int>(Epilogue::kGatedSilu)) != 0
            || desc.k < 2 * TILE_K || desc.k % TILE_K != 0 || desc.n <= 0 || desc.n % 64 != 0) {
            return false;
        }
        if constexpr (Gemm::is_grouped_gemm) {
            // Packed grouped descriptors are safe only with the declared
            // activation/output addressing contract.  The generic matcher
            // otherwise treats a single-group fff operation as compatible
            // with ibb/bbb and lets an indexed kernel win dense tuning.
            if (desc.striding_a != Gemm::kStridingA
                || desc.striding_b != Gemm::kStridingB
                || desc.striding_c != Gemm::kStridingC) {
                return false;
            }
        }
        const bool fuse_silu = (static_cast<int>(desc.epilogue) & static_cast<int>(Epilogue::kGatedSilu)) != 0;
        if (fuse_silu) {
            if (!Gemm::kSupportsFusedSilu || desc.type_c != kFloat8_e4m3 || desc.n % TILE_N != 0) {
                return false;
            }
            GemmDesc canonical = desc;
            canonical.type_c = desc_.type_c;
            return Kernel::is_feasible(canonical);
        }
        if constexpr (Gemm::kSupportsFusedSilu) {
            return false;
        }
        return Kernel::is_feasible(desc);
    }

private:
    int sm_count_{};
};

}  // namespace turbomind::gemm
