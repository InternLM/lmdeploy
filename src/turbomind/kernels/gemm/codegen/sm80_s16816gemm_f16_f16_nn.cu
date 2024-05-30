// Copyright (c) OpenMMLab. All rights reserved.

#include "sm80_s16816gemm_f16_f16_nn.h"
#include "src/turbomind/kernels/gemm/registry.h"

namespace turbomind::gemm {

void Registry::reigster_sm80_s16816gemm_f16_f16_v2()
{
    using sm80_s16816gemm_f16_f16_nn::Config;

    // clang-format off
    // Add(std::make_unique<KernelImpl<typename Config<128, 128, 32, 128, 16, 32, 3,  false, 1, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<half, 256, 128, 64, 64, 64, 64, 3, false, 1, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<half, 128, 128, 32, 64, 64, 32, 3, false, 0, 0>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<256, 128, 32, 64, 64, 32, 6,  false, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<256, 128, 32, 64, 64, 32, 3,  false, 1, 1>::Kernel>>());

    // Add(std::make_unique<KernelImpl<typename Config< 16,  16, 32,  16, 16, 32, 3,  false, 1, 1>::Kernel>>());
    // clang-format on
}

}  // namespace turbomind::gemm