// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"

#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
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

    using Impl  = typename Gemm::Impl;
    using Sched = typename Gemm::Scheduler;

    using OpA = typename Gemm::OperandA;
    using OpB = typename Gemm::OperandB;
    using OpU = typename Gemm::OperandU;
    using OpV = typename Gemm::OperandV;

    KernelImpl()
    {
        desc_.order_a = OpA::kOrder;
        desc_.order_b = transpose(OpB::kOrder);
        desc_.order_c = Gemm::kOrderC;

        desc_.type_a = data_type_v<typename Gemm::Ta>;
        desc_.type_b = data_type_v<typename Gemm::Tb>;
        desc_.type_c = data_type_v<typename Gemm::Tc>;

        using IterA = typename OpA::GmemIter;
        using IterB = typename OpB::GmemIter;

        desc_.striding_a = IterA::kMode;
        desc_.striding_b = IterB::kMode;
        desc_.striding_c = Gemm::Epilogue::kMode;

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

        info_.chunk_size_k = Gemm::kChunkSizeK;

        desc_.align.x = OpA::kOrder == kColMajor ? IterA::ThreadMap::kAccessC : 1;
        desc_.align.y = OpB::kOrder == kColMajor ? IterB::ThreadMap::kAccessC : 1;
        desc_.align.z = Gemm::CTA_K;

        desc_.policy_a = (int)IterA::Policy::kEvictPolicy;
        desc_.policy_b = (int)IterB::Policy::kEvictPolicy;
        desc_.c_tile   = {Gemm::Epilogue::TM, Gemm::Epilogue::TN};
        desc_.op_class = Impl::kOpClass;

        desc_.cluster_shape = {1, 1};

        auto func = gemm_kernel<Gemm, GemmParam, EpilogueParam, Sched>;

        cudaFuncGetAttributes(&info_.attr, func);

        info_.dynamic_smem_size = sizeof(typename Gemm::SharedStorage);

        if (info_.dynamic_smem_size > (48 << 10)) {
            cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, info_.dynamic_smem_size);
        }

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info_.max_active_ctas, func, Impl::WARPS * WARP_SIZE, info_.dynamic_smem_size);

        desc_.stages     = Impl::Stages;
        desc_.split_k    = Gemm::kSplitK;
        desc_.group_axis = Sched::group_axis;

        desc_.arch = Gemm::Arch::value;

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
               int                 swizzle,
               int                 splits,
               Workspace&          workspace,
               cudaStream_t        stream) override
    {
        MatrixLayout Adesc = _Adesc;

        const int m = Ddesc.rows;
        const int n = Ddesc.cols;
        const int k = Adesc.cols;
        const int l = std::max(1, Ddesc.num);

        auto transpose = [](MatrixLayout x) {
            std::swap(x.rows, x.cols);
            x.order = gemm::transpose(x.order);
            return x;
        };

        MatrixLayout Bdesc = transpose(_Bdesc);
        MatrixLayout Vdesc = transpose(_Vdesc);

        auto max_splits = GetMaxSplits({m, n, k, l}, swizzle, workspace.barriers_size, workspace.partials_size);

        Sched sched{{m, n, k, l}, swizzle, std::min(splits, max_splits)};
        sched.offsets_ = Ddesc.offsets;

        using Ta = typename Gemm::Ta;
        using Tb = typename Gemm::Tb;
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

        const bool silu_act = ((int)operation.epilogue & (int)Epilogue::kGatedSilu);

        MatrixLayout Pdesc = Ddesc;
        Pdesc.ld           = mk2cs<Gemm::kOrderC>(Pdesc.rows, Pdesc.cols).x;

        MatrixCombination_v3 combin_mat{to_param((void*)C, Cdesc), alpha, beta};

        EpilogueParam epilogue{to_param((void*)D, Ddesc),
                               to_param((void*)workspace.partials, Pdesc),
                               (int*)workspace.barriers,
                               combin_mat,
                               silu_act};

        // std::cout << Adesc.offsets << " " << Adesc.idxs << "\n";

        GemmParam param{
            to_param((void*)A, Adesc),
            to_param((void*)B, Bdesc),
            to_param((void*)U, Udesc),
            to_param((void*)V, Vdesc),
        };

        const auto grid  = sched.get_grid_shape();
        const auto block = Gemm::Impl::WARPS * WARP_SIZE;

        // std::cout << info_.name << " " << splits << " " << swizzle << " " << sched.tiles_[0] << " " <<
        // sched.tiles_[1]
        //           << std::endl;
        // std::cout << grid.x << " " << grid.y << " " << grid.z << "\n";

        gemm_kernel<Gemm><<<grid, block, info_.dynamic_smem_size, stream>>>(param, epilogue, sched);

        return 0;
    }

    std::array<size_t, 2> GetWorkspaceSize(int tiles, int splits) const
    {
        static constexpr bool kSerial = true;

        size_t barriers_size = sizeof(int) * tiles;
        size_t partials_size = sizeof(float) * CTA_M * CTA_N * tiles;

        if constexpr (!kSerial) {
            barriers_size *= splits;
            partials_size *= splits;
        }

        return {barriers_size, partials_size};
    }

    int GetMaxSplits(const int4& shape, int swizzle, size_t bsize, size_t psize) const override
    {
        if (!Gemm::kSplitK) {
            return 1;
        }

        const auto& [m, n, k, l] = shape;

        Sched sched{{m, n, k, l}, swizzle};  // for getting padded tiles

        const auto& [a, b] = GetWorkspaceSize(sched.tiles_[0] * sched.tiles_[1], 1);

        if (bsize >= a && psize >= b) {
            // Serial split-k requires workspace for 1 split only
            // But it can't exceed num of k chunks
            return cdiv(k, Gemm::kChunkSizeK);
        }
        else {
            return 1;
        }
    }

    int GetMaxSwizzle(const int4& shape) const override
    {
        const auto& [m, n, k, l] = shape;

        auto swizzle = Sched{{m, n, k, l}}.get_max_swizzle();
        // std::cout << m << " " << n << " " << k << " " << l << " " << swizzle << "\n";
        return swizzle;
    }
};

}  // namespace turbomind::gemm
