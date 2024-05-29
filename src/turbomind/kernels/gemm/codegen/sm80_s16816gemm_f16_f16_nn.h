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

namespace turbomind::gemm {

namespace sm80_s16816gemm_f16_f16_nn {

template<class T, int CTA_M, int CTA_K, int WARP_M, int WARP_CNT, bool Align_M>
struct OperandA {
    using Dtype = T;

    static constexpr Pack  kPack  = Pack::kNone;
    static constexpr Order kOrder = Order::kColMajor;

    using SmemLayout = std::conditional_t<CTA_M >= 64,
                                          SmemLayoutV2<CTA_K, CTA_M, 16, 64, Swizzle<3, 3, 3>>,
                                          SmemLayoutV2<CTA_K, CTA_M, 16, 32, Swizzle<2, 3, 3>>>;
    using SmemCopy   = SmemCopy_<SmemCopy_MMA_16816_B<T, true>, 16, WARP_M>;

    using _ThreadMap = gemm::ThreadMap<CTA_M, CTA_K, 8, WARP_CNT>;
    using GmemIter   = GmemIteratorSm80<T, _ThreadMap, SmemLayout, kPack, kOrder, Align_M, true>;
};

template<class T, int CTA_N, int CTA_K, int WARP_N, int WARP_CNT, bool Align_N>
struct OperandB {
    using Dtype = T;

    static constexpr Pack  kPack  = Pack::kNone;
    static constexpr Order kOrder = Order::kColMajor;

    using SmemLayout = std::conditional_t<CTA_K >= 64,
                                          SmemLayoutV2<CTA_N, CTA_K, std::min(16, CTA_N), 64, Swizzle<3, 3, 3>>,
                                          SmemLayoutV2<CTA_N, CTA_K, std::min(16, CTA_N), 32, Swizzle<2, 3, 3>>>;
    using SmemCopy   = SmemCopy_<SmemCopy_MMA_16816_B<T, false>, WARP_N, 16>;

    using _ThreadMap = gemm::ThreadMap<CTA_K, CTA_N, 8, WARP_CNT>;
    using GmemIter   = GmemIteratorSm80<T, _ThreadMap, SmemLayout, kPack, kOrder, true, Align_N>;
};

template<class T,
         int  CTA_M,
         int  CTA_N,
         int  CTA_K,
         int  WARP_M,
         int  WARP_N,
         int  WARP_K,
         int  Stages,
         bool SplitK,
         bool AlignedM,
         bool AlignedN>
struct Config {

    using TiledMma = TiledMMA<SM80_MMA_16x8x16_F32_F16_F16_F32_TN, WARP_M, WARP_N, WARP_K>;

    static constexpr int WARP_CNT = (CTA_M / WARP_M) * (CTA_N / WARP_N) * (CTA_K / WARP_K);

    using OperandA = OperandA<T, CTA_M, CTA_K, WARP_M, WARP_CNT, AlignedM>;
    using OperandB = OperandB<T, CTA_N, CTA_K, WARP_N, WARP_CNT, AlignedN>;
    using Void     = VoidOperand;

    using Mainloop = MainloopSm80_v2<CTA_M,
                                     CTA_N,
                                     CTA_K,  //
                                     TiledMma,
                                     OperandA,
                                     OperandB,
                                     Void,
                                     Void,
                                     Transform,
                                     Stages>;

    using Kernel = GemmUniversal<void, Mainloop, CtaMap, AlignedM, AlignedN, SplitK>;
};

}  // namespace sm80_s16816gemm_f16_f16_nn

}  // namespace turbomind::gemm