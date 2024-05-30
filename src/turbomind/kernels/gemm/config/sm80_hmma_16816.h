// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/impl.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"
#include "src/turbomind/kernels/gemm/mainloop_sm80_v2.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

namespace sm80_hmma_16816 {

// (m, k)
struct Operand_A_N {
    template<class T, int CTA_M, int CTA_K, int WARP_M, int WARP_CNT, bool Align_M>
    struct type {
        using Dtype = T;

        static constexpr Pack  kPack  = 0;
        static constexpr Order kOrder = Order::kColMajor;

        using SmemLayout = std::conditional_t<CTA_M >= 64,
                                              SmemLayoutV2<CTA_K, CTA_M, 16, 64, Swizzle<3, 3, 3>>,
                                              SmemLayoutV2<CTA_K, CTA_M, 16, 32, Swizzle<2, 3, 3>>>;
        using SmemCopy   = SmemCopy_<SmemCopy_MMA_16816_B<T, true>, 16, WARP_M>;

        using _ThreadMap = gemm::ThreadMap<CTA_M, CTA_K, 8, WARP_CNT>;
        using GmemIter   = GmemIteratorSm80<T, _ThreadMap, SmemLayout, kPack, kOrder, Align_M, true>;
    };
};

// (n, k)
struct Operand_B_T {
    template<class T, int CTA_N, int CTA_K, int WARP_N, int WARP_CNT, bool Align_N>
    struct type {
        using Dtype = T;

        static constexpr Pack  kPack  = 0;
        static constexpr Order kOrder = Order::kRowMajor;

        using SmemLayout = std::conditional_t<CTA_K >= 64,
                                              SmemLayoutV2<CTA_N, CTA_K, std::min(16, CTA_N), 64, Swizzle<3, 3, 3>>,
                                              SmemLayoutV2<CTA_N, CTA_K, std::min(16, CTA_N), 32, Swizzle<2, 3, 3>>>;
        using SmemCopy   = SmemCopy_<SmemCopy_MMA_16816_B<T, false>, WARP_N, 16>;

        using _ThreadMap = gemm::ThreadMap<CTA_K, CTA_N, 8, WARP_CNT>;
        using GmemIter   = GmemIteratorSm80<T, _ThreadMap, SmemLayout, kPack, kOrder, true, Align_N>;
    };
};

template<class T, class A_, class B_, class U_ = VoidOperandConst, class V_ = VoidOperandConst>
struct SM80_HMMA_16816 {
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

        static constexpr int WARP_CNT = (CTA_M / WARP_M) * (CTA_N / WARP_N) * (CTA_K / WARP_K);

        using A = typename A_::template type<T, CTA_M, CTA_K, WARP_M, WARP_CNT, AlignedM>;
        using B = typename B_::template type<T, CTA_N, CTA_K, WARP_N, WARP_CNT, AlignedN>;

        using U = typename U_::template type<T, CTA_M, CTA_K, WARP_M, WARP_CNT, AlignedM>;
        using V = typename V_::template type<T, CTA_N, CTA_K, WARP_N, WARP_CNT, AlignedN>;

        using Void = VoidOperand;

        using Mainloop = MainloopSm80_v2<CTA_M, CTA_N, CTA_K, TiledMma, A, B, U, V, Transform, Stages>;
        using Kernel   = GemmUniversal<void, Mainloop, CtaMap, AlignedM, AlignedN, SplitK>;
    };
};

}  // namespace sm80_hmma_16816

template<>
struct GetOperand<HMMA_16816, OPERAND_A, Order::kColMajor, false>: std::true_type {
    using Operand = sm80_hmma_16816::Operand_A_N;
};

template<>
struct GetOperand<HMMA_16816, OPERAND_B, Order::kRowMajor, false>: std::true_type {
    using Operand = sm80_hmma_16816::Operand_B_T;
};

namespace sm80_hmma_16816 {

}  // namespace sm80_hmma_16816

}  // namespace turbomind::gemm