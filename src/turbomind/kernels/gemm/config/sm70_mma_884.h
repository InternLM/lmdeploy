// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/impl.h"
#include "src/turbomind/kernels/gemm/iterator.h"
#include "src/turbomind/kernels/gemm/iterator_sm70.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"
#include "src/turbomind/kernels/gemm/mainloop_sm80_v2.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/simt.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/smem_copy_sm70.h"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace sm70_mma_884 {

template<Order order>
struct GetSmemLayout {
    template<int M, int K>
    static constexpr auto apply(pair<M, K>)
    {
        constexpr int2 cs = mk2cs<order>(M, K);
        return SmemLayoutV2<cs.y, cs.x, 1, 1>{};
    }
};

template<class T>
struct Operand_A {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kRowMajor;

    using SmemCopyAtom = SmemCopy_MMA_884_A<T>;

    using GetSmemLayout = GetSmemLayout<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

template<class T>
struct Operand_B {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kRowMajor;  // (n,k)

    using SmemCopyAtom = SmemCopy_MMA_884_B_COL<T>;

    using GetSmemLayout = GetSmemLayout<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

template<class T>
struct Operand_V {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kColMajor;  // (n,k)

    using SmemCopyAtom = SmemCopy_MMA_884_V<T, 1>;

    struct GetSmemLayout {  // m-major
        template<int M, int K>
        static constexpr auto apply(pair<M, K>)
        {
            return SmemLayoutV2<K, M>{};
        }
    };

    using GetGmemIter = GetGmemIter;
};

template<class T, Order order>
struct Operand_C {
    using Dtype = T;

    static constexpr Order kOrder = order;

    struct GetSmemLayout {
        template<int M, int N>
        static constexpr auto apply(pair<M, N>)
        {
            constexpr auto cs = mk2cs<order>(M, N);
            return SmemLayoutV2<cs.y, cs.x, 1, 1>{};
            // return SmemLayoutV2<cs.y, cs.x, 8, 32, Swizzle<2, 3, 2>>{};
        }
    };

    struct GetThreadMap {
        template<int M, int N, int THREADS>
        static constexpr auto apply(pair<M, N>, constant<THREADS>)
        {
            constexpr auto cs    = mk2cs<order>(M, N);
            constexpr int  WARPS = THREADS / WARP_SIZE;

            return ThreadMap_V2<cs.x, cs.y, 4, Raked, WARPS>{};
        }
    };
};

template<class T>
struct Operand_B_Pack {
    using Dtype = T;

    static constexpr int Pack_M = 2;

    static constexpr Pack  kPack  = HMMA_884 | OPERAND_B | Pack_M;
    static constexpr Order kOrder = kRowMajor;

    using SmemCopyAtom = SmemCopyAtom_Pack_v3<T, SmemCopy_MMA_884_B_COL<T>, kOrder, Pack_M>;

    using GetSmemLayout = GetSmemLayout<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

template<class T>
struct Operand_V_Pack {
    using Dtype = T;

    static constexpr int Pack_M = 2;

    static constexpr Pack  kPack  = HMMA_884 | OPERAND_V | Pack_M;
    static constexpr Order kOrder = kColMajor;

    using SmemCopyAtom = SmemCopyAtom_Pack_v3<T, SmemCopy_MMA_884_V<T, 4>, kColMajor, Pack_M>;

    struct GetSmemLayout {  // m-major
        template<int M, int K>
        static constexpr auto apply(pair<M, K>)
        {
            return SmemLayoutV2<K, M>{};
        }
    };

    using GetGmemIter = GetGmemIter;
};

template<class A, class TransformA, class U, class B, class TransformB, class V, Order order_c, class Tc>
struct SM70_MMA_884_F32 {
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

}  // namespace sm70_mma_884

template<class T>
struct GetOperand<HMMA_884, OPERAND_A, T, kRowMajor, false>: std::true_type {
    using Operand = sm70_mma_884::Operand_A<T>;
};

template<class T>
struct GetOperand<HMMA_884, OPERAND_B, T, kRowMajor, false>: std::true_type {
    using Operand = sm70_mma_884::Operand_B<T>;
};

template<class T>
struct GetOperand<HMMA_884, OPERAND_V, T, kColMajor, false>: std::true_type {
    using Operand = sm70_mma_884::Operand_V<T>;
};

template<class T>
struct GetOperand<HMMA_884, OPERAND_B, T, kRowMajor, true>: std::true_type {
    using Operand = sm70_mma_884::Operand_B_Pack<T>;
};

template<class T>
struct GetOperand<HMMA_884, OPERAND_V, T, kColMajor, true>: std::true_type {
    using Operand = sm70_mma_884::Operand_V_Pack<T>;
};

}  // namespace turbomind::gemm
