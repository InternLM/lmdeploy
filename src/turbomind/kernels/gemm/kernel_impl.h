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

        if (smem_size_ > (48 << 10)) {
            cudaFuncSetAttribute(gemm_kernel<Gemm>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_);
        }

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_ctas_, gemm_kernel<Gemm>, Impl::WARPS * WARP_SIZE, smem_size_);

        name_ = GetName();
    }

    int Launch(const Operation&    operation,
               const void*         alpha,
               const void*         A,
               const MatrixLayout& Adesc,
               const void*         U,
               const MatrixLayout& Udesc,
               const void*         B,
               const MatrixLayout& _Bdesc,
               const void*         V,
               const MatrixLayout& Vdesc,
               const void*         beta,
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

        const auto tiles = Map::get_tiled_shape(m, n, k, CTA_M, CTA_N, splits);

        if (splits > 1 && workspace.barriers == nullptr && workspace.partials == nullptr) {
            GetWorkspaceSizes(m, n, tiles.x, tiles.y, splits, workspace.barriers_size, workspace.partials_size);
            std::cout << "not enough workspace for " << splits << " splits\n";
            return -1;
        }

        // const auto log_tile = Map::get_log_tile(tiles, 8);
        const auto log_tile = swizzle;

        const auto grid  = Map::get_grid_shape(tiles, log_tile);
        const auto block = Gemm::Impl::WARPS * WARP_SIZE;

        using Ta = typename Gemm::Ta;
        using Tb = typename Gemm::Tb;
        using Tu = typename Gemm::Tu;
        using Tv = typename Gemm::Tv;
        using Tc = typename Gemm::Tc;

        if constexpr (1) {
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

        typename Gemm::Epilogue::Param epilogue{m,
                                                n,
                                                (Tc*)D,
                                                Cdesc.ld,
                                                (float*)workspace.partials,
                                                Cdesc.ld,  /// TODO: optimize this
                                                (int*)workspace.barriers,
                                                {},
                                                {1.f, 0.f},
                                                silu_act};

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
                                   log_tile,
                                   tiles,
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
        partials_size = sizeof(float) * m * n * splits;
        barriers_size = sizeof(int) * tiled_m * tiled_n * splits;
    }

    int GetMaxSplits(int m, int n, size_t barrier_size, size_t partials_size) override
    {
        if (!Gemm::SplitK) {  // kernel has no split-k support
            return 1;
        }

        const int tiled_m = ceil_div(m, CTA_M);
        const int tiled_n = ceil_div(n, CTA_N);

        size_t barriers_per_split{};
        size_t partials_per_split{};

        // workspace for 1 non-trival split
        GetWorkspaceSizes(m, n, tiled_m, tiled_n, 1, barriers_per_split, partials_per_split);

        return std::max(1, std::min<int>(barrier_size / barriers_per_split, partials_size / partials_per_split));
    }
};

}  // namespace turbomind::gemm
