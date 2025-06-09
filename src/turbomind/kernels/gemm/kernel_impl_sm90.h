// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

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

namespace turbomind::gemm {

extern __shared__ char smem_buf[];

template<class Kernel>
__global__ void __launch_bounds__(Kernel::CTA_SIZE, 1) gemm_kernel_sm90(const __grid_constant__ CUtensorMap tm_a,
                                                                        const __grid_constant__ CUtensorMap tm_b,
                                                                        const __grid_constant__ CUtensorMap tm_c,
                                                                        const __grid_constant__ CUtensorMap tm_u,
                                                                        const __grid_constant__ CUtensorMap tm_v,
                                                                        const MatrixParam                   param_A,
                                                                        const MatrixParam                   param_B,
                                                                        const MatrixParam                   param_U,
                                                                        const MatrixParam                   param_V,
                                                                        const MatrixParam                   param_C,
                                                                        // uint2                               box_V,
                                                                        typename Kernel::Scheduler sched,
                                                                        void*                      tensormap_buf)
{
    // if (cute::thread0()) {
    //     printf("ffs %d\n", __ffs(0x0));
    // }

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
               //    box_V,
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
    // using Impl = typename Gemm::Impl;

    // using OpA = typename Gemm::OperandA;
    // using OpB = typename Gemm::OperandB;
    // using OpU = typename Gemm::OperandU;
    // using OpV = typename Gemm::OperandV;

    KernelImplSm90()
    {
        desc_.order_a = kRowMajor;  // m, k
        desc_.order_b = kColMajor;  // k, n
        desc_.order_c = kRowMajor;

        desc_.type_a = data_type_v<typename Gemm::Ta>;
        desc_.type_b = data_type_v<typename Gemm::Tb>;
        desc_.type_c = data_type_v<typename Gemm::Tc>;

        // using IterA = typename OpA::GmemIter;
        // using IterB = typename OpB::GmemIter;

        desc_.striding_a = {is_grouped_gemm ? Striding::kBlocked : Striding::kFlat};  // IterA::kMode;
        desc_.striding_b = {is_grouped_gemm ? Striding::kBlocked : Striding::kFlat};  // IterB::kMode;
        desc_.striding_c = {is_grouped_gemm ? Striding::kBlocked : Striding::kFlat};  // Gemm::Epilogue::kMode;

        desc_.pack_a = {};  // OpA::kPack;
        desc_.pack_b = {};  // OpB::kPack;
        desc_.pack_u = {};  // OpU::kPack;
        desc_.pack_v = {};  // OpV::kPack;

        // desc_.quant_a = QuantDesc{};
        // desc_.quant_b = QuantDesc{};

        // if constexpr (OpU::SmemLayout::kSize > 1) {
        desc_.quant_a = QuantDesc{QuantType::kDefault, 128};
        // }

        // if constexpr (OpV::SmemLayout::kSize > 1) {
        desc_.quant_b = QuantDesc{QuantType::kDefault, 128};
        // }

        desc_.cta_tile = {TILE_M, TILE_N, TILE_K};
        desc_.mma_tile = {1, 1, 1};
        chunk_size_k_  = Gemm::TILE_K;

        desc_.align.x = 1;  // OpA::kOrder == kColMajor ? IterA::ThreadMap::kAccessC : 1;
        desc_.align.y = 1;  // OpB::kOrder == kColMajor ? IterB::ThreadMap::kAccessC : 1;
        desc_.align.z = 1;  // Gemm::TILE_K;

        desc_.policy_a = 0;                 // (int)IterA::Policy::kEvictPolicy;
        desc_.policy_b = 0;                 // (int)IterB::Policy::kEvictPolicy;
        desc_.c_tile   = {TILE_M, TILE_N};  // {Gemm::Epilogue::TM, Gemm::Epilogue::TN};
        desc_.op_class = OpClass::kGMMA_s64n16;

        desc_.cluster_shape = {Gemm::Cluster::M, Gemm::Cluster::N};

        smem_size_ = Gemm::kSmemSize;

        desc_.stages  = Gemm::Stages;
        desc_.split_k = 1;      // Gemm::kSplitK;
        desc_.sched   = false;  // Gemm::kDynamicSched;

        desc_.arch = Gemm::Arch::value;

        // using Sched = typename Gemm::Sched;

        // auto func = gemm_kernel<Gemm, GemmParam, EpilogueParam, Sched>;

        auto func = gemm_kernel_sm90<Gemm>;

        if (smem_size_ > (48 << 10)) {
            cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_);
        }

        if (1) {
            cudaFuncSetAttribute(func, cudaFuncAttributeNonPortableClusterSizeAllowed, 16);
        }

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&desc_.max_active_ctas, func, Gemm::CTA_SIZE, smem_size_);

        cudaFuncGetAttributes(&desc_.attr, func);

        name_ = GetName();
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
               int                 swizzle,
               int                 splits,
               Workspace&          workspace,
               cudaStream_t        stream) const override
    {
        using Sched = typename Gemm::Scheduler;

        MatrixLayout Adesc = _Adesc;

        [[maybe_unused]] const int m = Ddesc.rows;
        [[maybe_unused]] const int n = Ddesc.cols;
        [[maybe_unused]] const int k = Adesc.cols;

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
            const int4 shape{m, n, k, Adesc.num};

            swizzle = Sched::get_log_tile(tiles, 1 << swizzle);

            Sched sched{};
            sched.init(shape, swizzle, {TILE_M, TILE_N, TILE_K});

            sched.next_cluster_id_ = TM_CHECK_NOTNULL(workspace.flags);

            sched.offsets_ = Adesc.offsets;

            return sched;
        }();

        constexpr int kMulticastA = Gemm::kMulticastA;
        constexpr int kMulticastB = Gemm::kMulticastB;
        constexpr int kMulticastU = Gemm::kMulticastU;

        constexpr int kTileM = Gemm::TILE_M;
        constexpr int kTileN = Gemm::TILE_N;

        check_cuda_error(cudaMemsetAsync(workspace.flags, 0, sizeof(int), stream));

        // std::cout << "A: " << Adesc << "\n";
        auto tm_a = make_2d_tma_desc((void*)A, Adesc, {kTileM / kMulticastA, TILE_K}, CU_TENSOR_MAP_SWIZZLE_128B);

        // std::cout << "B: " << Bdesc << "\n";
        auto tm_b = make_2d_tma_desc(Gemm::is_grouped_gemm ? nullptr : (void*)B,
                                     Bdesc,
                                     {kTileN / kMulticastB, TILE_K},
                                     CU_TENSOR_MAP_SWIZZLE_128B);

        // std::cout << "C: " << Cdesc << "\n";
        using LayoutC = typename Gemm::LayoutC;
        auto tm_c     = make_2d_tma_desc((void*)C, Cdesc, {LayoutC::S0, LayoutC::C0}, get_tma_swizzle(Gemm::kSwizzleC));

        CUtensorMap tm_u{};
        if (U) {
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

        static constexpr int sm_count = 132;

        static constexpr int cluster_size = Gemm::kClusterSize;

        auto       grid  = sm_count / cluster_size * cluster_size;
        const auto block = Gemm::CTA_SIZE;

        cudaLaunchConfig_t config{};
        config.gridDim          = grid;
        config.blockDim         = block;
        config.dynamicSmemBytes = smem_size_;
        config.stream           = stream;

        auto func = gemm_kernel_sm90<Gemm>;

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
                                     to_param((void*)A, Adesc),
                                     to_param((void*)B, Bdesc),
                                     to_param((void*)U, Udesc),
                                     to_param((void*)V, Vdesc),
                                     to_param((void*)D, Ddesc),
                                     //  box_v,
                                     sched,
                                     workspace.tensormaps);
        TM_CHECK_EQ(ec, cudaSuccess) << cudaGetErrorString(ec);

        // gemm_kernel_sm90<Gemm><<<grid, block, smem_size_, stream>>>(tm_a, tm_b, tm_c, (Tc*)C, Cdesc.ld, sched);

        // if constexpr (0) {
        //     [[maybe_unused]] static const int _ = [] {
        //         std::cout << "A:\n";
        //         Print(typename Gemm::OperandA::GmemIter::ThreadMap{});
        //         std::cout << "\nB:\n";
        //         Print(typename Gemm::OperandB::GmemIter::ThreadMap{});
        //         if constexpr (!std::is_same_v<Ta, Tc>) {
        //             std::cout << "\nU:\n";
        //             Print(typename Gemm::OperandU::GmemIter::ThreadMap{});
        //         }
        //         if constexpr (!std::is_same_v<Tb, Tc>) {
        //             std::cout << "\nV:\n";
        //             Print(typename Gemm::OperandV::GmemIter::ThreadMap{});
        //         }
        //         printf("warp count: %d\n", Impl::WARPS);
        //         Print_(typename Gemm::Impl::MMA_Map{});

        //         printf("C:\n");
        //         Print(typename Gemm::Epilogue::Map{});

        //         std::cout << "Smem for mainloop: " << sizeof(Gemm::SharedStorage::mainloop) << "\n";
        //         std::cout << "Smem for epilogue: " << sizeof(Gemm::SharedStorage::epilogue) << "\n";

        //         return 0;
        //     }();
        // }

        // const bool silu_act = ((int)operation.epilogue & (int)Epilogue::kGatedSilu);

        // MatrixLayout Pdesc = Ddesc;
        // Pdesc.ld           = mk2cs<Gemm::kOrderC>(Pdesc.rows, Pdesc.cols).x;

        // MatrixCombination_v3 combin_mat{to_param((void*)C, Cdesc), alpha, beta};

        // EpilogueParam epilogue{to_param((void*)D, Ddesc),
        //                        to_param((void*)workspace.partials, Pdesc),
        //                        (int*)workspace.barriers,
        //                        combin_mat,
        //                        silu_act};

        // std::cout << Adesc.offsets << " " << Adesc.idxs << "\n";

        // GemmParam param{
        //     to_param((void*)A, Adesc),
        //     to_param((void*)B, Bdesc),
        //     to_param((void*)U, Udesc),
        //     to_param((void*)V, Vdesc),
        // };

        // std::cout << grid.x << " " << grid.y << " " << grid.z << "\n";

        // gemm_kernel<Gemm><<<grid, block, smem_size_, stream>>>(param, epilogue, sched);

        return 0;
    }

    std::array<size_t, 2> GetWorkspaceSizesV2(const int4& shape, int tiles, int splits) const
    {
        static constexpr bool kSerial = true;

        const auto& [m, n, _, num] = shape;

        size_t barriers_size = sizeof(int) * tiles;
        size_t partials_size = sizeof(float) * m * n * num;

        if constexpr (!kSerial) {
            barriers_size *= splits;
            partials_size *= splits;
        }

        return {barriers_size, partials_size};
    }

    int GetMaxSplits(const int4& shape, int64_t tiles, size_t bsize, size_t psize) const override
    {
        if (!Gemm::kSplitK) {
            return 1;
        }

        const auto& [m, n, k, _] = shape;

        const auto& [a, b] = GetWorkspaceSizesV2(shape, tiles, 1);

        if (bsize >= a && psize >= b) {
            // Serial split-k requires workspace for 1 split only
            // But it can't exceed num of k chunks
            return cdiv(k, Gemm::kChunkSizeK);
        }
        else {
            return 1;
        }
    }

    int GetSwizzle(int m, int n, int k, int splits, int swizzle) const override
    {
        // return swizzle;
        using Map        = typename Gemm::Scheduler;
        const auto tiles = get_tiled_shape(m, n, TILE_M, TILE_N);
        return Map::get_log_tile(tiles, 1 << swizzle);
    }

    int FixSplits(const int4& shape, int2 tiled_mn, int splits, Workspace& ws) const
    {
        const int tiles            = tiled_mn.x * tiled_mn.y;
        const auto& [bsize, psize] = GetWorkspaceSizesV2(shape, tiles, splits);

        if (ws.barriers_size < bsize || ws.partials_size < psize) {
            const int max_splits       = GetMaxSplits(shape, tiles, ws.barriers_size, ws.partials_size);
            const auto& [m, n, k, num] = shape;
            fprintf(
                stderr,
                "Problem size (%d, %d, %d), workspace size too small (%d, %d) vs required (%d, %d) for %d splits. Force `splits` = %d\n",
                m,
                n,
                k,
                (int)ws.barriers_size,
                (int)ws.partials_size,
                (int)bsize,
                (int)psize,
                splits,
                max_splits);
            splits = max_splits;
        }

        return splits;
    }

    bool is_feasible(const GemmDesc& desc) const noexcept override
    {
        if (desc.striding_a != desc_.striding_a) {
            return false;
        }
        if (desc.striding_b != desc_.striding_b) {
            return false;
        }
        if (desc.striding_c != desc_.striding_c) {
            return false;
        }
        return Kernel::is_feasible(desc);
    }
};

}  // namespace turbomind::gemm