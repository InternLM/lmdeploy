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
#include "src/turbomind/kernels/gemm/types.h"

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
        using SmemLayout = SmemLayoutV2<CTA_K, CTA_M, 16, 64, Swizzle<3, 3, 3>>;
        using SmemCopy   = SmemCopy_<SmemCopy_MMA_16816_B<true>, 16, WARP_M>;
        using GmemIter = GmemIteratorSm80<T, gemm::ThreadMap<CTA_M, CTA_K, 8, WARP_CNT>, SmemLayout, AlignedM, true, 0>;
        static constexpr auto Layout     = LayoutType::kColMajor;
        static constexpr bool is_k_major = false;
    };

    struct OperandB {
        using Dtype      = half;
        using SmemLayout = SmemLayoutV2<CTA_K, CTA_N, 16, 64, Swizzle<3, 3, 3>>;
        using SmemCopy   = SmemCopy_<SmemCopy_MMA_16816_A<true>, 16, WARP_N>;
        using GmemIter = GmemIteratorSm80<T, gemm::ThreadMap<CTA_N, CTA_K, 8, WARP_CNT>, SmemLayout, AlignedN, true, 1>;
        static constexpr auto Layout     = LayoutType::kRowMajor;
        static constexpr bool is_k_major = false;
    };

    using Mainloop = MainloopSm80_v2<CTA_M, CTA_N, CTA_K, TiledMma, OperandA, OperandB, OperandB, Stages>;

    using Kernel = GemmUniversal<void, Mainloop, CtaMap, AlignedM, AlignedN, SplitK>;
};

}  // namespace

void Registry::reigster_sm80_s16816gemm_f16_f16_nt()
{
    // clang-format off
    // Add(std::make_unique<KernelImpl<typename Config<128, 128, 32, 128, 16, 32, 3,  false, 1, 1>::Kernel>>());
    Add(std::make_unique<KernelImpl<typename Config<256, 128, 64, 64, 64, 64, 3, false, 0, 0>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<128, 128, 32, 64, 64, 32, 3,  false, 0, 0>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<256, 128, 32, 64, 64, 32, 6,  false, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<256, 128, 32, 64, 64, 32, 3,  false, 1, 1>::Kernel>>());

    // Add(std::make_unique<KernelImpl<typename Config< 16,  16, 32,  16, 16, 32, 3,  false, 1, 1>::Kernel>>());
    // clang-format on
}

}  // namespace turbomind::gemm