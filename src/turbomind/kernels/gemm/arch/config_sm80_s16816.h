// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <numeric>

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/mma_sm80.h"
#include "src/turbomind/kernels/gemm/arch/operand_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include "src/turbomind/kernels/gemm/mainloop_sm80_v2.h"
#include "src/turbomind/kernels/gemm/scheduler_sm70.cuh"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm::sm80_s16816 {

template<class Arch,
         class Dtype,
         class A,
         class TransformA,
         class U,
         class B,
         class TransformB,
         class V,
         Order order_C,
         class Tc,
         Striding mode_A,
         Striding mode_B,
         Striding mode_C,
         Order    raster_order,
         int      group_axis>
struct Sm80_s16816 {

    static_assert(A::SmemCopyAtom::K == B::SmemCopyAtom::K);

    static constexpr int SMEM_M = A::SmemCopyAtom::M / A::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_N = B::SmemCopyAtom::M / B::SmemCopyAtom::kFragNum;
    static constexpr int SMEM_K = A::SmemCopyAtom::K;

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
             int  GroupSizeU   = 1,
             int  GroupSizeV   = 1,
             int  TILE_C_M_    = -1,
             int  TILE_C_N_    = -1,
             bool FusePrefecth = true>

    struct Type {

        // Raked partition dont support `Pack_M > 1`
        using Partition = Blocked<TG_M, TG_N, kColMajor>;
        using MMA_Map   = MMA_Map<CTA_M, CTA_N, CTA_K, SMEM_M, SMEM_N, SMEM_K, Partition, TG_K>;
        using MMA       = Tiled_MMA_v2<SM80_MMA_16x8x16_F32_F16_F16_F32_TN<Dtype>, MMA_Map>;

        using Mainloop = MainloopSm80_v2<MMA,
                                         A,
                                         IteratorSm80<mode_A, PolicyA>,
                                         TransformA,
                                         U,
                                         GroupSizeU,
                                         B,
                                         IteratorSm80<mode_B, PolicyB>,
                                         TransformB,
                                         V,
                                         GroupSizeV,
                                         Stages,
                                         FusePrefecth>;

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
                                         mode_C,
                                         SplitK>;

        using Kernel = GemmUniversal<Arch, Mainloop, Epilogue, Scheduler>;
    };
};

}  // namespace turbomind::gemm::sm80_s16816
