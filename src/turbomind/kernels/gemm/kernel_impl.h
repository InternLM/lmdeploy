// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

template<class Gemm>
class KernelImpl: public Kernel {
public:
    // import frequently used constants
    static constexpr int CTA_M = Gemm::CTA_M;
    static constexpr int CTA_N = Gemm::CTA_N;
    static constexpr int CTA_K = Gemm::CTA_K;

    using Impl = typename Gemm::Impl;

    using OpA = typename Gemm::OperandA;
    using OpB = typename Gemm::OperandB;
    using OpU = typename Gemm::OperandU;
    using OpV = typename Gemm::OperandV;

    KernelImpl()
    {
        desc_.order_a = OpA::kOrder;
        desc_.order_b = transpose(OpB::kOrder);
        desc_.order_c = Gemm::kOrderC;

        desc_.type_a = get_data_type_v<typename Gemm::Ta>;
        desc_.type_b = get_data_type_v<typename Gemm::Tb>;
        desc_.type_c = get_data_type_v<typename Gemm::Tc>;

        desc_.pack_a = OpA::kPack;
        desc_.pack_b = OpB::kPack;
        desc_.pack_u = OpU::kPack;
        desc_.pack_v = OpV::kPack;

        desc_.quant_a = QuantDesc{};
        desc_.quant_b = QuantDesc{};

        if constexpr (OpU::SmemLayout::kSize > 1) {
            desc_.quant_a = QuantDesc{QuantType::kDefault, OpU::kGroupSize};
        }

        if constexpr (OpV::SmemLayout::kSize > 1) {
            desc_.quant_b = QuantDesc{QuantType::kDefault, OpV::kGroupSize};
        }

        desc_.cta_tile = {Gemm::CTA_M, Gemm::CTA_N, Gemm::CTA_K};
        desc_.mma_tile = {Impl::MMA_Map::kGroupM, Impl::MMA_Map::kGroupN, Impl::MMA_Map::kGroupK};
        chunk_size_k_  = Gemm::kChunkSizeK;

        using IterA = typename OpA::GmemIter;
        using IterB = typename OpB::GmemIter;

        desc_.align.x = OpA::kOrder == kColMajor ? IterA::ThreadMap::kAccessC : 1;
        desc_.align.y = OpB::kOrder == kColMajor ? IterB::ThreadMap::kAccessC : 1;
        desc_.align.z = Gemm::CTA_K;

        desc_.policy_a = (int)IterA::Policy::kEvictPolicy;
        desc_.policy_b = (int)IterB::Policy::kEvictPolicy;
        desc_.c_tile   = {Gemm::Epilogue::TM, Gemm::Epilogue::TN};
        desc_.op_class = Impl::kOpClass;

        smem_size_ = sizeof(typename Gemm::SharedStorage);

        desc_.stages  = Impl::Stages;
        desc_.split_k = Gemm::SplitK;

        desc_.arch = Gemm::Arch::value;

        using Params = typename Gemm::Param;
        using CtaMap = typename Gemm::CtaMap;

        auto func = gemm_kernel<Gemm, Params, CtaMap>;

        if (smem_size_ > (48 << 10)) {
            cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_);
        }

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &desc_.max_active_ctas, gemm_kernel<Gemm, Params, CtaMap>, Impl::WARPS * WARP_SIZE, smem_size_);

        cudaFuncGetAttributes(&desc_.attr, func);

        name_ = GetName();
    }

    int Launch(const Operation&    operation,
               float               alpha,
               const void*         A,
               const MatrixLayout& Adesc,
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
               cudaStream_t        stream) override
    {
        using Map = typename Gemm::CtaMap;

        const int m = Ddesc.rows;
        const int n = Ddesc.cols;
        const int k = Adesc.cols;

        auto transpose = [](MatrixLayout x) {
            std::swap(x.rows, x.cols);
            x.order = gemm::transpose(x.order);
            return x;
        };

        const MatrixLayout Bdesc = transpose(_Bdesc);
        const MatrixLayout Vdesc = transpose(_Vdesc);

        const int chunk_cnt = ceil_div(k, Gemm::kChunkSizeK);

        // Limit splits by num of chunks to avoid chaos
        splits = std::min(chunk_cnt, splits);

        auto tiles = Map::get_tiled_shape(m, n, k, CTA_M, CTA_N, splits);

        if (splits > 1) {
            size_t bsize{}, psize{};
            GetWorkspaceSizes(m, n, tiles.x, tiles.y, splits, bsize, psize);
            const int max_splits = GetMaxSplits(m, n, k, workspace.barriers_size, workspace.partials_size);
            if (workspace.barriers_size < bsize || workspace.partials_size < psize) {
                fprintf(
                    stderr,
                    "Problem size (%d, %d, %d), workspace size too small (%d, %d) vs required (%d, %d) for %d splits. Force `splits` = %d\n",
                    m,
                    n,
                    k,
                    (int)workspace.barriers_size,
                    (int)workspace.partials_size,
                    (int)bsize,
                    (int)psize,
                    splits,
                    max_splits);
                splits = max_splits;
                tiles  = Map::get_tiled_shape(m, n, k, CTA_M, CTA_N, splits);
            }
        }

        swizzle = Map::get_log_tile(tiles, 1 << swizzle);

        const auto grid  = Map::get_grid_shape(tiles, swizzle);
        const auto block = Gemm::Impl::WARPS * WARP_SIZE;

        using Ta = typename Gemm::Ta;
        using Tb = typename Gemm::Tb;
        using Tu = typename Gemm::Tu;
        using Tv = typename Gemm::Tv;
        using Tc = typename Gemm::Tc;

        if constexpr (0) {
            [[maybe_unused]] static const int _ = [] {
                std::cout << "A:\n";
                Print(typename Gemm::OperandA::GmemIter::ThreadMap{});
                std::cout << "\nB:\n";
                Print(typename Gemm::OperandB::GmemIter::ThreadMap{});
                if constexpr (!std::is_same_v<Ta, Tc>) {
                    std::cout << "\nU:\n";
                    Print(typename Gemm::OperandU::GmemIter::ThreadMap{});
                }
                if constexpr (!std::is_same_v<Tb, Tc>) {
                    std::cout << "\nV:\n";
                    Print(typename Gemm::OperandV::GmemIter::ThreadMap{});
                }
                printf("warp count: %d\n", Impl::WARPS);
                Print_(typename Gemm::Impl::MMA_Map{});

                printf("C:\n");
                Print(typename Gemm::Epilogue::Map{});

                std::cout << "Smem for mainloop: " << sizeof(Gemm::SharedStorage::mainloop) << "\n";
                std::cout << "Smem for epilogue: " << sizeof(Gemm::SharedStorage::epilogue) << "\n";

                return 0;
            }();
        }

        int lda = Adesc.ld;
        int ldb = Bdesc.ld;

        if (Gemm::kPackA) {
            lda = mk2cs<Gemm::kOrderA>(Packing_v2<Gemm::kPackA, Gemm::kOrderA>::apply({m, k})).x;
        }
        if (Gemm::kPackB) {
            ldb = mk2cs<Gemm::kOrderB>(Packing_v2<Gemm::kPackB, Gemm::kOrderB>::apply({n, k})).x;
        }

        // std::cout << "lda=" << lda << ", ldb=" << ldb << ", ldc=" << Cdesc.ld << "\n";

        // std::cout << "C: " << C << ", D: " << D << "\n";

        const bool silu_act = ((int)operation.epilogue & (int)Epilogue::kGatedSilu);

        const int partial_C_ld = mk2cs<Gemm::kOrderC>(Ddesc.rows, Ddesc.cols).x;

        EpilogueParam<Tc> epilogue{m,
                                   n,
                                   (Tc*)D,
                                   Ddesc.ld,
                                   (float*)workspace.partials,
                                   partial_C_ld,
                                   (int*)workspace.barriers,
                                   {alpha, beta, (const Tc*)C, Cdesc.ld},
                                   silu_act};

        const int chunk_per_split = chunk_cnt / splits;
        const int chunk_remianing = chunk_cnt % splits;
        const int chunk_offset    = splits - chunk_remianing;
        // chunk_id = z * chunk_per_split + max(z - (splits - chunk_remaining), 0);
        // offset_k = chunk_id * kChunkSizeK;
        // gemm_k_size = offset_k + (chunk_per_split + int(z > chunk_offset)) * kChunkSizeK
        // gemm_k_size = std::min(gemm_k_size, k) - offset_k

        // std::cout << k << " " << Gemm::kChunkSizeK << " " << splits << " " << chunk_per_split << " " <<
        // chunk_remianing << " " << chunk_offset << "\n";

        typename Gemm::Param param{m,
                                   n,
                                   k,
                                   typename Gemm::PtrA{(Ta*)A},
                                   lda,
                                   (Tu*)U,
                                   Udesc.ld,
                                   typename Gemm::PtrB{(Tb*)B},
                                   ldb,
                                   (Tv*)V,
                                   Vdesc.ld,
                                   swizzle,
                                   tiles,
                                   chunk_per_split,
                                   chunk_offset,
                                   epilogue};

        gemm_kernel<Gemm><<<grid, block, smem_size_, stream>>>(param, Map{});

        return 0;
    }

    template<class T>
    static auto _cast(T* p)
    {
        if constexpr (bitsof<T> % 8 == 0) {
            return p;
        }
        else {
            return (char*)p;
        }
    }

    // ! This assumes N results in 16 byte aligned partials
    void
    GetWorkspaceSizes(int m, int n, int tiled_m, int tiled_n, int splits, size_t& barriers_size, size_t& partials_size)
    {
        static constexpr bool kSerial = true;

        partials_size = sizeof(float) * m * n;
        barriers_size = sizeof(int) * tiled_m * tiled_n;

        if constexpr (!kSerial) {
            partials_size *= splits;
            barriers_size *= splits;
        }
    }

    int GetMaxSplits(int m, int n, int k, size_t barrier_size, size_t partials_size) override
    {
        if (!Gemm::SplitK) {  // kernel has no split-k support
            return 1;
        }

        const int tiled_m = ceil_div(m, CTA_M);
        const int tiled_n = ceil_div(n, CTA_N);

        size_t bsize_1split{};
        size_t psize_1split{};

        // workspace for 1 non-trival split
        GetWorkspaceSizes(m, n, tiled_m, tiled_n, 1, bsize_1split, psize_1split);

        if (barrier_size >= bsize_1split && partials_size >= psize_1split) {
            // Serial split-k requires workspace for 1 split only
            // But it can't exceed num of k chunks
            const int chunk_cnt = ceil_div(k, Gemm::kChunkSizeK);
            return std::min(chunk_cnt, 32);
        }
        else {
            return 1;
        }
    }

    int GetSwizzle(int m, int n, int k, int splits, int swizzle) override
    {
        using Map        = typename Gemm::CtaMap;
        const auto tiles = Map::get_tiled_shape(m, n, k, CTA_M, CTA_N, splits);
        return Map::get_log_tile(tiles, 1 << swizzle);
    }
};

}  // namespace turbomind::gemm
