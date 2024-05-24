// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/impl.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"
#include "src/turbomind/kernels/gemm/mainloop_sm80_v2.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/tiled_mma.h"
#include "src/turbomind/kernels/gemm/transform.h"

namespace turbomind::gemm {

namespace {

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
struct Config {
    using T  = half;
    using Tb = half;
    using Tq = half2;

    using TiledMma = TiledMMA<SM80_MMA_16x8x16_F32_F16_F16_F32_TN, WARP_M, WARP_N, WARP_K>;

    static constexpr int WARP_CNT = (CTA_M / WARP_M) * (CTA_N / WARP_N) * (CTA_K / WARP_K);

    struct OperandA {
        using Dtype      = half;
        using SmemLayout = SmemLayoutV2<CTA_M / 16, CTA_K * 16>;
        using SmemCopy   = SmemCopy_Packed<half, WARP_M, 16, 1, 1>;
        using GmemIter =
            GmemIteratorSm80<Dtype, ThreadMap<CTA_K * 16, CTA_M / 16, 8, WARP_CNT>, SmemLayout, false, true, 0>;
        static constexpr Order kOrder     = Order::kColMajor;
        static constexpr bool  is_k_major = true;
    };

    struct OperandB {
        using Dtype      = half;
        using SmemLayout = SmemLayoutV2<CTA_N, CTA_K, std::min(16, CTA_N), 32, Swizzle<2, 3, 3>>;
        using SmemCopy   = SmemCopy_<SmemCopy_MMA_16816_B<half, false>, WARP_N, 16>;
        using GmemIter = GmemIteratorSm80<T, gemm::ThreadMap<CTA_K, CTA_N, 8, WARP_CNT>, SmemLayout, AlignedN, true, 1>;
        static constexpr Order kOrder     = Order::kColMajor;
        static constexpr bool  is_k_major = true;
    };

    using Mainloop = MainloopSm80_v2<CTA_M, CTA_N, CTA_K, TiledMma, OperandA, OperandB, OperandB, Transform, Stages>;

    using Kernel = GemmUniversal<void, Mainloop, CtaMap, AlignedM, AlignedN, SplitK>;
};

}  // namespace

Kernel& gKernel()
{
    static KernelImpl<typename Config<32, 32, 32, 32, 32, 32, 3, false, 1, 1>::Kernel> inst{};
    return inst;
}

}  // namespace turbomind::gemm