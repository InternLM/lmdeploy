// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/impl.h"
#include "src/turbomind/kernels/gemm/impl_16816.h"
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
    using Tb = half;
    using Tq = half2;

    using Impl   = Impl<MMA_16816, T, Tb, Tq, CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, Stages, 1>;
    using Kernel = GemmUniversal<void, Mainloop_sm80<Impl>, CtaMap, AlignedM, AlignedN, SplitK>;
};

}  // namespace

void Registry::reigster_sm80_s16816gemm_f16_f16()
{
    // clang-format off
    // Add(std::make_unique<KernelImpl<typename Config<128, 128, 32, 128, 16, 32, 3,  false, 1, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<128, 128, 32, 64, 64, 32, 3,  false, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<256, 128, 32, 64, 64, 32, 6,  false, 0, 1>::Kernel>>());
    Add(std::make_unique<KernelImpl<typename Config<128, 128, 32, 64, 64, 32, 3,  false, 0, 1>::Kernel>>());

    // Add(std::make_unique<KernelImpl<typename Config< 16,  16, 16,  16, 16, 16, 3,  false, 0, 1>::Kernel>>());
    // clang-format on
}

}  // namespace turbomind::gemm