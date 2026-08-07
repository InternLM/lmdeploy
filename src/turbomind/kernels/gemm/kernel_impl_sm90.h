// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <cstring>
#include <type_traits>

#include "cute/util/debug.hpp"
#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/gemm_universal_sm90_v3.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

#include "src/turbomind/kernels/gemm/tma.h"

#include "src/turbomind/utils/cuda_utils.h"

#define TM_GEMM_CUTLASS_NAME 0

#if TM_GEMM_CUTLASS_NAME
#define gemm_kernel_name cutlass_gemm_kernel_sm90
#else
#define gemm_kernel_name gemm_kernel_sm90
#endif

namespace turbomind::gemm {

extern __shared__ __align__(1024) char smem_buf[];

template<class Kernel>
__global__ void __launch_bounds__(Kernel::CTA_SIZE, 1) gemm_kernel_name(const __grid_constant__ CUtensorMap tm_a,
                                                                        const __grid_constant__ CUtensorMap tm_b,
                                                                        const __grid_constant__ CUtensorMap tm_c,
                                                                        const __grid_constant__ CUtensorMap tm_u,
                                                                        const __grid_constant__ CUtensorMap tm_v,
                                                                        const MatrixParam                   param_A,
                                                                        const MatrixParam                   param_B,
                                                                        const MatrixParam                   param_U,
                                                                        const MatrixParam                   param_V,
                                                                        const MatrixParam                   param_C,
                                                                        const MatrixParam                   param_W,
                                                                        bool                                fuse_silu,
                                                                        typename Kernel::Scheduler          sched,
                                                                        void* tensormap_buf)
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
               param_W,
               fuse_silu,
               sched,
               (CUtensorMap*)tensormap_buf,
               smem_buf);
    }
#endif
}

template<class Gemm>
class KernelImplSm90: public Kernel {
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

    KernelImplSm90()
    {
        desc_.order_a = kRowMajor;  // m, k
        desc_.order_b = kColMajor;  // k, n
        desc_.order_c = kRowMajor;

        desc_.type_a              = data_type_v<typename Gemm::Ta>;
        desc_.type_b              = data_type_v<typename Gemm::Tb>;
        desc_.type_c              = data_type_v<typename Gemm::Tc>;
        desc_.supports_fused_silu = Gemm::kSupportsFusedSilu;

        desc_.striding_a = Gemm::kStridingA;
        desc_.striding_b = Gemm::kStridingB;
        desc_.striding_c = Gemm::kStridingC;

        desc_.pack_a = {};  // OpA::kPack;
        desc_.pack_b = {};  // OpB::kPack;
        desc_.pack_u = {};  // OpU::kPack;
        desc_.pack_v = {};  // OpV::kPack;

        desc_.quant_a = QuantDesc{QuantType::kK, 128};
        desc_.quant_b = QuantDesc{QuantType::kB, 128};

        desc_.cta_tile = {TILE_M, TILE_N, TILE_K};
        desc_.mma_tile = {1, 1, 1};

        info_.chunk_size_k = Gemm::TILE_K;

        desc_.align.x = 1;  // OpA::kOrder == kColMajor ? IterA::ThreadMap::kAccessC : 1;
        desc_.align.y = 1;  // OpB::kOrder == kColMajor ? IterB::ThreadMap::kAccessC : 1;
        desc_.align.z = 1;  // Gemm::TILE_K;

        desc_.policy_a = 0;                 // (int)IterA::Policy::kEvictPolicy;
        desc_.policy_b = 0;                 // (int)IterB::Policy::kEvictPolicy;
        desc_.c_tile   = {TILE_M, TILE_N};  // {Gemm::Epilogue::TM, Gemm::Epilogue::TN};
        desc_.op_class = OpClass::kGMMA_q64n32;
        desc_.raster   = Gemm::kRasterOrder;

        AlgoBits algo{};
        algo.family   = Gemm::kAlgoFamily;
        algo.math_wgs = Gemm::WARPGROUPS;
        desc_.algo    = algo.u32();

        desc_.cluster_shape = {Gemm::Cluster::M, Gemm::Cluster::N};

        info_.dynamic_smem_size = Gemm::kSmemSize;

        desc_.stages     = Gemm::Stages;
        desc_.split_k    = 1;  // Gemm::kSplitK;
        desc_.group_axis = is_grouped_gemm ? 0 : -1;

        desc_.arch = Gemm::Arch::value;

        auto func = gemm_kernel_name<Gemm>;

        cudaFuncGetAttributes(&info_.attr, func);

        if (info_.dynamic_smem_size > (48 << 10)) {
            cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, info_.dynamic_smem_size);
        }

        if (1) {
            cudaFuncSetAttribute(func, cudaFuncAttributeNonPortableClusterSizeAllowed, 16);
        }

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
        using Sched = typename Gemm::Scheduler;

        MatrixLayout Adesc = _Adesc;

        const int  m          = Ddesc.rows;
        const int  n          = Ddesc.cols;
        const int  k          = Adesc.cols;
        const int  num_groups = is_grouped_gemm ? std::max(Adesc.num, 1) : Adesc.num;
        const bool fuse_silu  = ((int)operation.epilogue & (int)Epilogue::kGatedSilu) != 0;

        TM_CHECK_GE(cdiv(k, TILE_K), 2) << "The kernel requires at least 2 k-tiles to work";

        // std::cout << "M: " << m << ", N: " << n << ", K: " << k << "\n";

        auto transpose = [](MatrixLayout x) {
            std::swap(x.rows, x.cols);
            x.order = gemm::transpose(x.order);
            return x;
        };

        // (K, N) -> (N, K)
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
        constexpr int kMulticastU = Gemm::kMulticastU;

        constexpr int kTileM = Gemm::TILE_M;
        constexpr int kTileN = Gemm::TILE_N;

        if (Gemm::Scheduler::is_dynamic) {
            TM_CUDA_CHECK(cudaMemsetAsync(workspace.flags, 0, sizeof(int), stream));
        }

        // Indexed-A: gather activations in-kernel; host A TMA template unused.
        auto tm_a = make_2d_tma_desc(Gemm::kStridingA == Striding::kIndexed ? nullptr : (void*)A,
                                     Adesc,
                                     {kTileM / kMulticastA, TILE_K},
                                     CU_TENSOR_MAP_SWIZZLE_128B);

        // std::cout << "B: " << Bdesc << "\n";
        auto tm_b = make_2d_tma_desc(Gemm::is_grouped_gemm ? nullptr : (void*)B,
                                     Bdesc,
                                     {kTileN / kMulticastB, TILE_K},
                                     CU_TENSOR_MAP_SWIZZLE_128B);

        // std::cout << "C: " << Cdesc << "\n";
        auto make_tm_c = [&](auto fused_silu) {
            constexpr bool kFuseSilu = decltype(fused_silu)::value;
            using Output             = typename Gemm::template Output<kFuseSilu>;
            using LayoutC            = typename Output::LayoutC;

            MatrixLayout Cdesc_tma = Cdesc;
            if constexpr (kFuseSilu) {
                TM_CHECK_EQ(Cdesc_tma.cols % 2, 0);
                Cdesc_tma.cols /= 2;
            }
            return make_2d_tma_desc(
                (void*)C, Cdesc_tma, {LayoutC::S0, LayoutC::C0}, get_tma_swizzle(Output::kSwizzleC));
        };

        CUtensorMap tm_c;
        if constexpr (Gemm::kSupportsFusedSilu) {
            tm_c = fuse_silu ? make_tm_c(std::true_type{}) : make_tm_c(std::false_type{});
        }
        else {
            tm_c = make_tm_c(std::false_type{});
        }

        CUtensorMap tm_u{};
        // Indexed-A also gathers U by idxs; no U TMA template.
        if (U && Gemm::kStridingA != Striding::kIndexed) {
            // std::cout << "U: " << Udesc << "\n";
            tm_u = make_2d_tma_desc((void*)U, Udesc, {Gemm::kBoxU / kMulticastU, 1}, CU_TENSOR_MAP_SWIZZLE_NONE);
        }

        CUtensorMap            tm_v{};
        [[maybe_unused]] uint2 box_v{};
        if (V) {
            // std::cout << "V: " << Vdesc << "\n";
            // box_v = {(uint32_t)round_up(cdiv(k, 128), 4), 2};
            // std::cout << "V: " << Vdesc << ", box: " << box_v.x << "," << box_v.y << "\n";
            // tm_v = make_2d_tma_desc((void*)V, Vdesc, {box_v.y, box_v.x}, CU_TENSOR_MAP_SWIZZLE_NONE);
        }

        const auto param_A = to_param((void*)A, Adesc);
        const auto param_B = to_param((void*)B, Bdesc);
        const auto param_U = to_param((void*)U, Udesc);
        const auto param_V = to_param((void*)V, Vdesc);
        const auto param_C = to_param((void*)D, Ddesc);
        const auto param_W = to_param((void*)W, Wdesc);

        // Grouped: prepare_moe_tma_descs rebases per-expert A/B/U/C before GEMM launch.
        if constexpr (is_grouped_gemm) {
            sched.offsets_ = Gemm::PrepareTmaDescs(tm_a,
                                                   tm_b,
                                                   tm_u,
                                                   tm_c,
                                                   param_A,
                                                   param_B,
                                                   param_U,
                                                   param_C,
                                                   fuse_silu,
                                                   (CUtensorMap*)workspace.tensormaps,
                                                   num_groups,
                                                   m,
                                                   n,
                                                   stream);
            TM_CUDA_CHECK(cudaGetLastError());
        }

        const int sm_count = sm_count_;

        static constexpr int cluster_size = Gemm::kClusterSize;

        auto       grid  = sm_count / cluster_size * cluster_size;
        const auto block = Gemm::CTA_SIZE;

        cudaLaunchConfig_t config{};
        config.gridDim          = grid;
        config.blockDim         = block;
        config.dynamicSmemBytes = Gemm::GetSmemSize(fuse_silu);
        config.stream           = stream;

        auto func = gemm_kernel_name<Gemm>;

        [[maybe_unused]] static bool _ = [&] {
            int max_cluster_size = 0;
            cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, func, &config);
            // std::cout << "max cluster size: " << max_cluster_size << "\n";
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

        // std::cout << "max active cluster: " << max_active_cluster << "\n";

        // std::cout << "swizzle: " << swizzle << ", split: " << splits << "\n";

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
                                     param_W,
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
        using Map = typename Gemm::Scheduler;
        // TODO: fix tiled shape
        const auto tiles = get_tiled_shape(shape.x, shape.y, TILE_M, TILE_N);
        return Map::get_log_tile(tiles, 1 << 10);
    }

    bool is_feasible(const GemmDesc& desc) const noexcept override
    {
        const bool fuse_silu = ((int)desc.epilogue & (int)Epilogue::kGatedSilu) != 0;
        if (fuse_silu) {
            if (!Gemm::kSupportsFusedSilu || desc.type_c != kFloat8_e4m3) {
                return false;
            }
        }
        // A/B strides span K, C stride spans N (half-width for fused SiLU).
        const int c_ld = fuse_silu ? desc.n / 2 : desc.n;
        if (!is_tma_stride_feasible(desc.type_a, desc.k) || !is_tma_stride_feasible(desc.type_b, desc.k)
            || !is_tma_stride_feasible(desc.type_c, c_ld)) {
            return false;
        }
        if (fuse_silu) {
            GemmDesc canonical = desc;
            canonical.type_c   = desc_.type_c;
            return Kernel::is_feasible(canonical);
        }
        return Kernel::is_feasible(desc);
    }

private:
    int sm_count_ = 0;
};

}  // namespace turbomind::gemm
