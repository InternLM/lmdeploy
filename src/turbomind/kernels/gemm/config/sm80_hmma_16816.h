// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/impl.h"
#include "src/turbomind/kernels/gemm/iterator.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"
#include "src/turbomind/kernels/gemm/mainloop_sm80_v2.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/smem_copy_sm80.h"
#include "src/turbomind/kernels/gemm/thread_group_map.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace sm80_hmma_16816 {

struct GetSmemLayout {
    template<int S, int C>
    static constexpr auto apply(pair<S, C>)
    {
        constexpr int S0 = S >= 16 ? 16 : 8;
        constexpr int C0 = C >= 64 ? 64 : (C >= 32 ? 32 : 16);
        using _Small     = std::conditional_t<C0 == 32, Swizzle<2, 3, 3>, Swizzle<1, 3, 3>>;
        using Swizzle    = std::conditional_t<C0 == 64, Swizzle<3, 3, 3>, _Small>;
        return SmemLayoutV2<S, C, S0, C0, Swizzle>{};
    }
};

template<Order order>
struct GetSmemLayoutV2 {
    template<int M, int K>
    static constexpr auto apply(pair<M, K>)
    {
        if constexpr (order == kRowMajor) {
            return GetSmemLayout::apply(pair<M, K>{});
        }
        else {
            return GetSmemLayout::apply(pair<K, M>{});
        }
    }
};

// (m, k)
template<class T>
struct Operand_A_N {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = Order::kColMajor;

    using SmemCopyAtom = SmemCopy_MMA_16816_B<T, true>;

    using GetSmemLayout = GetSmemLayoutV2<kOrder>;
    using GetGmemIter   = GetGmemIter;
};

// (n, k)
template<class T>
struct Operand_B_T {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = Order::kRowMajor;

    using SmemCopyAtom = SmemCopy_MMA_16816_B<T, false>;

    using GetSmemLayout = GetSmemLayoutV2<kOrder>;
    using GetGmemIter   = GetGmemIter;
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
            // Padding 4C halves bank-confict but slow down the kernel
            // return SmemLayoutV2<cs.y, cs.x, 8, cs.x, Swizzle<4, 1, 6>>{};
            // return SmemLayoutV2<cs.y, cs.x + 2>{};
            // return SmemLayoutV2<cs.y, cs.x, 8, cs.x, Swizzle<2, 3, 4>>{};
            return SmemLayoutV2<cs.y, cs.x, 16, cs.x, Swizzle<2, 3, 4>>{};
        }
    };

    struct GetThreadMap {
        template<int M, int N, int THREADS>
        static constexpr auto apply(pair<M, N>, constant<THREADS>)
        {
            constexpr auto cs = mk2cs<order>(M, N);
            return ThreadMap<cs.x, cs.y, 4, THREADS / WARP_SIZE>{};
        }
    };
};

template<class T>
struct Operand_U {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = kColMajor;

    using SmemCopyAtom = SmemCopy_MMA_16816_U<T>;

    struct GetSmemLayout {
        template<int M, int K>
        static constexpr auto apply(pair<M, K>)
        {
            return SmemLayoutV2<K, M>{};
        }
    };
    using GetGmemIter = GetGmemIter;
};

template<class A, class TransformA, class U, class B, class TransformB, class V, Order order_c, class Tc>
struct SM80_HMMA_16816_F32 {
    template<int  CTA_M,
             int  CTA_N,
             int  CTA_K,
             int  WARP_CNT_M,
             int  WARP_CNT_N,
             int  WARP_CNT_K,
             int  Stages,
             bool SplitK,
             bool AlignedM,
             bool AlignedN,
             int  GroupSizeU = 1,
             int  GroupSizeV = 1>
    struct Type {
        using MMA_Map = RakedThreadGroupMap<CTA_M, CTA_N, CTA_K, 16, 16, 16, WARP_CNT_M, WARP_CNT_N, WARP_CNT_K>;
        using MMA     = Tiled_MMA_v2<SM80_MMA_16x8x16_F32_F16_F16_F32_TN, MMA_Map>;

        using Mainloop = MainloopSm80_v2<CTA_M,
                                         CTA_N,
                                         CTA_K,
                                         MMA,
                                         A,
                                         TransformA,
                                         U,
                                         GroupSizeU,
                                         B,
                                         TransformB,
                                         V,
                                         GroupSizeV,
                                         Stages>;

        using Epilogue = gemm::Epilogue_<Tc,
                                         CTA_M,
                                         CTA_N,
                                         CTA_M,
                                         CTA_N,
                                         MMA::kThreadCount,
                                         typename MMA::Rearrange,
                                         Operand_C<float, order_c>,
                                         SplitK>;

        using Kernel = GemmUniversal<void, Mainloop, Epilogue, CtaMap>;
    };
};

}  // namespace sm80_hmma_16816

template<class T>
struct GetOperand<HMMA_16816, OPERAND_A, T, kColMajor, false>: std::true_type {
    using Operand = sm80_hmma_16816::Operand_A_N<T>;
};

template<class T>
struct GetOperand<HMMA_16816, OPERAND_B, T, kRowMajor, false>: std::true_type {
    using Operand = sm80_hmma_16816::Operand_B_T<T>;
};

template<class T>
struct GetOperand<HMMA_16816, OPERAND_U, T, kColMajor, false>: std::true_type {
    using Operand = sm80_hmma_16816::Operand_U<T>;
};

template<class T>
struct GetOperand<HMMA_16816, OPERAND_V, T, kColMajor, false>: std::true_type {
    using Operand = sm80_hmma_16816::Operand_U<T>;
};

namespace sm80_hmma_16816 {

}  // namespace sm80_hmma_16816

}  // namespace turbomind::gemm