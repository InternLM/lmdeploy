// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
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
    template<int C, int S>
    static constexpr auto apply(pair<C, S>)
    {
        constexpr int S0 = S >= 16 ? 16 : 8;
        constexpr int C0 = C >= 64 ? 64 : (C >= 32 ? 32 : 16);
        using _Small     = std::conditional_t<C0 == 32, Swizzle<2, 3, 3>, Swizzle<1, 3, 3>>;
        using Swizzle    = std::conditional_t<C0 == 64, Swizzle<3, 3, 3>, _Small>;
        return SmemLayoutV2<S, C, S0, C0, Swizzle>{};
    }
};

// (m, k)
template<class T>
struct Operand_A_N {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = Order::kColMajor;

    using SmemCopyAtom = SmemCopy_MMA_16816_B<T, true>;

    using GetSmemLayout = GetSmemLayout;
    using GetGmemIter   = GetGmemIter;
};

// (n, k)
template<class T>
struct Operand_B_T {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = Order::kRowMajor;

    using SmemCopyAtom = SmemCopy_MMA_16816_B<T, false>;

    using GetSmemLayout = GetSmemLayout;
    using GetGmemIter   = GetGmemIter;
};

template<class A, class B, class U = VoidOperand, class V = VoidOperand>
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
             bool AlignedN>
    struct Type {
        using MMA_Map  = RakedThreadGroupMap<CTA_M, CTA_N, CTA_K, 16, 16, 16, WARP_CNT_M, WARP_CNT_N, WARP_CNT_K>;
        using MMA      = Tiled_MMA_v2<SM80_MMA_16x8x16_F32_F16_F16_F32_TN, MMA_Map>;
        using Mainloop = MainloopSm80_v2<CTA_M, CTA_N, CTA_K, MMA, A, B, U, V, Transform, Stages>;
        using Kernel   = GemmUniversal<void, Mainloop, CtaMap, AlignedM, AlignedN, SplitK>;
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

namespace sm80_hmma_16816 {

}  // namespace sm80_hmma_16816

}  // namespace turbomind::gemm