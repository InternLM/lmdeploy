// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/mma_simt.h"
#include "src/turbomind/kernels/gemm/arch/operand_simt.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/iterator_sm70.h"
#include "src/turbomind/kernels/gemm/mainloop_sm70.h"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace simt {

template<class A, class TransformA, class U, class B, class TransformB, class V, Order order_c, class Tc>
struct Sm75_Simt {

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
             int  GroupSizeU = 1,
             int  GroupSizeV = 1,
             int  TILE_C_M_  = -1,
             int  TILE_C_N_  = -1>
    struct Type {

        // (TM, TN, TK) = R(MMA_Atom, SmemCopy_Atom)
        using MMA_Atom = MMA_SIMT<half>;

        static constexpr int TM = MMA_Atom::M;
        static constexpr int TN = MMA_Atom::N;
        static constexpr int TK = MMA_Atom::K;

        using Partition = Blocked<TG_M, TG_N, kColMajor>;

        using MMA_Map = MMA_Map<CTA_M, CTA_N, CTA_K, SMEM_M, SMEM_N, SMEM_K, Partition, TG_K>;
        using MMA     = Tiled_MMA_v2<MMA_Atom, MMA_Map>;

        // using MMA_Map = RakedThreadGroupMap<CTA_M, CTA_N, CTA_K, TM, TN, TK, WARP_CNT_M, WARP_CNT_N, WARP_CNT_K>;

        using Mainloop = MainloopSm70<MMA,
                                      A,
                                      IteratorSm70<PolicyA>,
                                      TransformA,
                                      U,
                                      GroupSizeU,
                                      B,
                                      IteratorSm70<PolicyB>,
                                      TransformB,
                                      V,
                                      GroupSizeV,
                                      Stages,
                                      true>;

        static constexpr int TILE_C_M = TILE_C_M_ == -1 ? CTA_M : TILE_C_M_;
        static constexpr int TILE_C_N = TILE_C_N_ == -1 ? CTA_N : TILE_C_N_;

        using Epilogue = gemm::Epilogue_<Tc,
                                         CTA_M,
                                         CTA_N,
                                         TILE_C_M,
                                         TILE_C_N,
                                         MMA::kThreadCount,
                                         Rearrange<MMA>,
                                         Operand_C<float, order_c>,
                                         SplitK>;

        using Kernel = GemmUniversal<Sm75, Mainloop, Epilogue, CtaMap>;
    };
};

}  // namespace simt

}  // namespace turbomind::gemm
