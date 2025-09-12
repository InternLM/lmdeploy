// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <numeric>

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/mma_sm70.h"
#include "src/turbomind/kernels/gemm/arch/operand_sm70_s884.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
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
         Order raster_order,
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

    template<int CTA_M,
             int CTA_N,
             int CTA_K,
             int TG_M,
             int TG_N,
             int TG_K,
             class PolicyA,
             class PolicyB,
             int  Stages,
             bool SplitK,
             int  GroupSizeU = 1,
             int  GroupSizeV = 1,
             int  TILE_C_M_  = -1,
             int  TILE_C_N_  = -1>
    struct Type {

        // (TM, TN, TK) = R(MMA_Atom, SmemCopy_Atom)
        using MMA_Atom = SM70_MMA_884;

        using Partition = Blocked<TG_M, TG_N, kColMajor>;
        using MMA_Map   = MMA_Map<CTA_M, CTA_N, CTA_K, SMEM_M, SMEM_N, SMEM_K, Partition, TG_K>;

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

        using Scheduler = SchedulerSm70<raster_order, CTA_M, CTA_N, CTA_K, CHUNK_K, SplitK, group_axis>;

        static constexpr int TILE_C_M = TILE_C_M_ == -1 ? CTA_M : TILE_C_M_;
        static constexpr int TILE_C_N = TILE_C_N_ == -1 ? CTA_N : TILE_C_N_;

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

        using Kernel = GemmUniversal<Sm70, Mainloop, Epilogue, Scheduler>;
    };
};

template<Order raster_order>
using Config_U4_d = Sm70_s884<typename GetOperand<HMMA_884, OPERAND_A, half, kRowMajor, false>::Operand,
                              Transform_Default,
                              VoidOperand,
                              typename GetOperand<HMMA_884, OPERAND_B, uint4_t, kRowMajor, true>::Operand,
                              Transform_HMMA_SIMT_B,
                              typename GetOperand<HMMA_884, OPERAND_V, uint32_t, kColMajor, true>::Operand,
                              kRowMajor,
                              half,
                              raster_order,
                              -1>;

template<Order raster_order>
using Config_U4_g = Sm70_s884<Operand_A<half>,           // A
                              Transform_Default,         // tarnsform A
                              VoidOperand,               // U
                              Operand_B_Pack<uint4_t>,   // B
                              Transform_HMMA_SIMT_B,     // transform B,
                              Operand_V_Pack<uint32_t>,  // V
                              kRowMajor,                 // order_C
                              half,                      // Tc
                              raster_order,
                              0>;

template<Order raster_order, int group_axis = -1>
using Config_MXF4 = Sm70_s884<Operand_A<half>,             // A
                              Transform_Default,           // tarnsform A
                              VoidOperand,                 // U
                              Operand_B_Pack<fp4_e2m1_t>,  // B
                              Transform_HMMA_SIMT_B,       // transform B,
                              Operand_V_Pack<uint8_t>,     // V
                              kRowMajor,                   // order_C
                              half,                        // Tc
                              raster_order,
                              group_axis>;

template<Order raster_order, int group_axis = -1>
using Config_E4M3 = Sm70_s884<Operand_A<half>,             // A
                              Transform_Default,           // tarnsform A
                              VoidOperand,                 // U
                              Operand_B_Pack<fp8_e4m3_t>,  // B
                              Transform_HMMA_SIMT_B,       // transform B,
                              Operand_V_Pack<uint16_t>,    // V
                              kRowMajor,                   // order_C
                              half,                        // Tc
                              raster_order,
                              group_axis>;

template<Order raster_order, int group_axis = -1>
using Config_F16 = Sm70_s884<Operand_A<half>,       // A
                             Transform_Default,     // tarnsform A
                             VoidOperand,           // U
                             Operand_B_Pack<half>,  // B
                             Transform_Default,     // transform B
                             VoidOperand,           // V
                             kRowMajor,             // order_C
                             half,                  // Tc
                             raster_order,
                             group_axis>;

}  // namespace turbomind::gemm::sm70_s884
