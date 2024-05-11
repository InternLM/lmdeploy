// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/impl.h"
#include "src/turbomind/kernels/gemm/impl_81616.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"
#include "src/turbomind/kernels/gemm/mainloop_sm80.h"
#include "src/turbomind/kernels/gemm/registry.h"

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
    using Tb = uint4_t;
    using Tq = half2;

    using Impl   = Impl<MMA_81616, T, Tb, Tq, CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, Stages, 1>;
    using Kernel = GemmUniversal<void, Mainloop_sm80<Impl>, CtaMap, AlignedM, AlignedN, SplitK>;
};

}  // namespace

void Registry::register_sm80_f16_s4_asym_g128_extra()
{
    // clang-format off
    // Add(std::make_unique<KernelImpl<typename Config<128, 128, 32, 128, 32, 32, 4,  true, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<128, 128, 32, 128, 32, 32, 5,  true, 0, 1>::Kernel>>());

    // Add(std::make_unique<KernelImpl<typename Config< 96, 128, 64,  96, 32, 64, 4,  true, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config< 96, 128, 64,  96, 32, 64, 5,  true, 0, 1>::Kernel>>());

    // Add(std::make_unique<KernelImpl<typename Config< 64, 128, 64,  64, 64, 32, 4,  true, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config< 64, 128, 64,  64, 64, 32, 5,  true, 0, 1>::Kernel>>());

    // Add(std::make_unique<KernelImpl<typename Config< 32, 128, 64,  32, 64, 32, 4,  true, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config< 32, 128, 64,  32, 64, 32, 5,  true, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config< 32, 128, 64,  32, 64, 32, 6,  true, 0, 1>::Kernel>>());

    // Add(std::make_unique<KernelImpl<typename Config< 16, 128, 64,  16, 64, 32, 4,  true, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config< 16, 128, 64,  16, 64, 32, 5,  true, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config< 16, 128, 64,  16, 64, 32, 6,  true, 0, 1>::Kernel>>());

    // Add(std::make_unique<KernelImpl<typename Config<  8, 128, 64,   8, 64, 32, 4,  true, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<  8, 128, 64,   8, 64, 32, 5,  true, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<  8, 128, 64,   8, 64, 32, 6,  true, 0, 1>::Kernel>>());
    // clang-format on
}

}  // namespace turbomind::gemm