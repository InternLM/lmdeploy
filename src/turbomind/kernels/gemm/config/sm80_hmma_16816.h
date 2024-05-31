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
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace sm80_hmma_16816 {

struct _GetSmemLayout {
    template<int C, int S>
    static constexpr auto apply(pair<C, S>)
    {
        constexpr int S0 = S >= 16 ? 16 : 8;
        constexpr int C0 = C >= 64 ? 64 : 32;

        auto swizzle = [] {
            if constexpr (C0 == 64) {
                return Swizzle<3, 3, 3>{};
            }
            else {
                return Swizzle<2, 3, 3>{};
            }
        };

        return SmemLayoutV2<S, C, S0, C0, decltype(swizzle())>{};
    }
};

// (m, k)
template<class T>
struct Operand_A_N {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = Order::kColMajor;

    using GetSmemLayout = _GetSmemLayout;
    using SmemCopy      = SmemCopy_v2_<SmemCopy_MMA_16816_B<T, true>>;
    using GetGmemIter   = gemm::GetGmemIter;
};

// (n, k)
template<class T>
struct Operand_B_T {
    using Dtype = T;

    static constexpr Pack  kPack  = 0;
    static constexpr Order kOrder = Order::kRowMajor;

    using GetSmemLayout = _GetSmemLayout;
    using SmemCopy      = SmemCopy_v2_<SmemCopy_MMA_16816_B<T, false>>;
    using GetGmemIter   = gemm::GetGmemIter;
};

template<class A, class B, class U = VoidOperand, class V = VoidOperand>
struct SM80_HMMA_16816_F32 {
    template<int  CTA_M,
             int  CTA_N,
             int  CTA_K,
             int  WARP_M,
             int  WARP_N,
             int  WARP_K,
             int  Stages,
             bool SplitK,
             bool AlignedM,
             bool AlignedN>
    struct Type {
        using TiledMma = TiledMMA<SM80_MMA_16x8x16_F32_F16_F16_F32_TN, WARP_M, WARP_N, WARP_K>;
        using Mainloop = MainloopSm80_v2<CTA_M, CTA_N, CTA_K, TiledMma, A, B, U, V, Transform, Stages>;
        using Kernel   = GemmUniversal<void, Mainloop, CtaMap, AlignedM, AlignedN, SplitK>;
    };
};

}  // namespace sm80_hmma_16816

template<class T>
struct GetOperand<HMMA_16816, OPERAND_A, T, Order::kColMajor, false>: std::true_type {
    using Operand = sm80_hmma_16816::Operand_A_N<T>;
};

template<class T>
struct GetOperand<HMMA_16816, OPERAND_B, T, Order::kRowMajor, false>: std::true_type {
    using Operand = sm80_hmma_16816::Operand_B_T<T>;
};

namespace sm80_hmma_16816 {

}  // namespace sm80_hmma_16816

}  // namespace turbomind::gemm