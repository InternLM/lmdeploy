// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/operand_sm70_s884.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/iterator_sm70.h"
#include "src/turbomind/kernels/gemm/mainloop_sm80_v2.h"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm::sm70_s884 {

template<class A, class TransformA, class U, class B, class TransformB, class V, Order order_c, class Tc>
struct Sm70_s884 {
    template<int CTA_M,
             int CTA_N,
             int CTA_K,
             int WARP_CNT_M,
             int WARP_CNT_N,
             int WARP_CNT_K,
             class PolicyA,
             class PolicyB,
             int  Stages,
             bool SplitK,
             int  GroupSizeU = 1,
             int  GroupSizeV = 1>
    struct Type {

        // (TM, TN, TK) = R(MMA_Atom, SmemCopy_Atom)
        using MMA_Atom = SM70_MMA_884;

        static constexpr int TM = MMA_Atom::M;
        static constexpr int TN = MMA_Atom::N;
        static constexpr int TK = MMA_Atom::K;

        using MMA_Map = RakedThreadGroupMap<CTA_M, CTA_N, CTA_K, TM, TN, TK, WARP_CNT_M, WARP_CNT_N, WARP_CNT_K>;

        using MMA = Tiled_MMA_v2<MMA_Atom, MMA_Map>;

        using Mainloop = MainloopSm80_v2<CTA_M,
                                         CTA_N,
                                         CTA_K,
                                         MMA,
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
                                         false>;

        using Epilogue = gemm::Epilogue_<Tc,
                                         CTA_M,
                                         CTA_N,
                                         CTA_M,
                                         CTA_N,
                                         MMA::kThreadCount,
                                         typename MMA::Rearrange,
                                         Operand_C<float, order_c>,
                                         SplitK>;

        using Kernel = GemmUniversal<Sm80, Mainloop, Epilogue, CtaMap>;
    };
};

}  // namespace turbomind::gemm::sm70_s884
