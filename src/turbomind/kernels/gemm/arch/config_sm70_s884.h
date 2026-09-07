// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <numeric>

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/mma_sm70.h"
#include "src/turbomind/kernels/gemm/arch/operand_sm70_s884.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/kernel/config.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"
#include "src/turbomind/kernels/gemm/iterator_sm70.h"
#include "src/turbomind/kernels/gemm/mainloop_sm70.h"
#include "src/turbomind/kernels/gemm/scheduler_sm70.cuh"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm::sm70_s884 {

template<class A,
         class TransformA,
         class U,
         class B,
         class TransformB,
         class V,
         Order order_C,
         class Tc,
         int   group_axis>
struct Sm70_s884 {

    static_assert(A::SmemCopyAtom::K == B::SmemCopyAtom::K);

    static constexpr int SMEM_M = A::SmemCopyAtom::M / A::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_N = B::SmemCopyAtom::M / B::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_K = A::SmemCopyAtom::K;

    static constexpr auto MODE_ = group_axis >= 0 ? Striding::kBlocked : Striding::kFlat;

    static constexpr auto MODE_A = group_axis == 0 ? Striding::kIndexed : MODE_;
    static constexpr auto MODE_B = group_axis == 1 ? Striding::kIndexed : MODE_;
    static constexpr auto MODE_C = MODE_;

    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int GroupSizeU = 1, int GroupSizeV = 1, int EpiM = -1, int EpiN = -1>
    struct Type {
        using Tile = typename Config_::Tile;
        using Groups = typename Config_::Groups;

        static constexpr int CTA_M = Tile::M;
        static constexpr int CTA_N = Tile::N;
        static constexpr int CTA_K = Tile::K;


        // (TM, TN, TK) = R(MMA_Atom, SmemCopy_Atom)
        using MMA_Atom = SM70_MMA_884;

        using Partition = Blocked<Groups::M, Groups::N, kColMajor>;
        using MMA_Map   = gemm::MMA_Map<CTA_M, CTA_N, CTA_K, SMEM_M, SMEM_N, SMEM_K, Partition, Groups::K>;

        using MMA = Tiled_MMA_v2<MMA_Atom, MMA_Map>;

        using Mainloop = MainloopSm70<MMA,
                                      A,
                                      IteratorSm70<MODE_A, PolicyA>,
                                      TransformA,
                                      U,
                                      GroupSizeU,
                                      B,
                                      IteratorSm70<MODE_B, PolicyB>,
                                      TransformB,
                                      V,
                                      GroupSizeV,
                                      Stages,
                                      true>;  // FusePrefetch_

        static constexpr int CHUNK_K = std::lcm(std::lcm(GroupSizeU, GroupSizeV), CTA_K);

        using Scheduler = SchedulerSm70<Raster, CTA_M, CTA_N, CTA_K, CHUNK_K, SplitK, group_axis>;

        static constexpr int TILE_C_M = EpiM == -1 ? CTA_M : EpiM;
        static constexpr int TILE_C_N = EpiN == -1 ? CTA_N : EpiN;

        using Epilogue = gemm::Epilogue_<Tc,
                                         CTA_M,
                                         CTA_N,
                                         TILE_C_M,
                                         TILE_C_N,
                                         MMA::kThreadCount,
                                         Rearrange<MMA>,
                                         Operand_C<float, order_C>,
                                         MODE_C,
                                         SplitK>;

        using Kernel = KernelImpl<GemmUniversal<Sm70, Mainloop, Epilogue, Scheduler>>;
    };
};

template<int GroupSize>
struct Config_U4_d {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, int GroupAxis = -1>
    using Type = typename Sm70_s884<typename GetOperand<HMMA_884, OPERAND_A, half, kRowMajor, false>::Operand, Transform_Default, VoidOperand, typename GetOperand<HMMA_884, OPERAND_B, uint4_t, kRowMajor, true>::Operand, Transform_HMMA_SIMT_B, typename GetOperand<HMMA_884, OPERAND_V, uint32_t, kColMajor, true>::Operand, kRowMajor, half, GroupAxis>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 1, GroupSize, EpiM, EpiN>::Kernel;
};

struct Config_MXF4 {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, int GroupAxis = -1>
    using Type = typename Sm70_s884<Operand_A<half>, Transform_Default, VoidOperand, Operand_B_Pack<fp4_e2m1_t>, Transform_HMMA_SIMT_B, Operand_V_Pack<uint8_t>, kRowMajor, half, GroupAxis>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 1, 32, EpiM, EpiN>::Kernel;
};

struct Config_E4M3 {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, int GroupAxis = -1>
    using Type = typename Sm70_s884<Operand_A<half>, Transform_Default, VoidOperand, Operand_B_Pack<fp8_e4m3_t>, Transform_HMMA_SIMT_B, Operand_V_Pack<uint16_t>, kRowMajor, half, GroupAxis>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 1, 128, EpiM, EpiN>::Kernel;
};

struct Config_F16 {
    template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, int GroupAxis = -1>
    using Type = typename Sm70_s884<Operand_A<half>, Transform_Default, VoidOperand, Operand_B_Pack<half>, Transform_Default, VoidOperand, kRowMajor, half, GroupAxis>::template Type<Config_, Stages, Raster, PolicyA, PolicyB, SplitK, 1, 1, EpiM, EpiN>::Kernel;
};

}  // namespace turbomind::gemm::sm70_s884
