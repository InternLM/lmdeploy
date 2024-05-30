// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/config/sm80_hmma_16816.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::reigster_sm80_s16816gemm_f16_f16_v2()
{
    using Config =
        sm80_hmma_16816::SM80_HMMA_16816<half,
                                         typename GetOperand<HMMA_16816, OPERAND_A, kColMajor, false>::Operand,
                                         typename GetOperand<HMMA_16816, OPERAND_B, kRowMajor, false>::Operand>;

    // clang-format off
    // Add(std::make_unique<KernelImpl<typename Config<128, 128, 32, 128, 16, 32, 3,  false, 1, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<half, 256, 128, 64, 64, 64, 64, 3, false, 1, 1>::Kernel>>());
    Add(std::make_unique<KernelImpl<Config::Type<128, 128, 32, 64, 64, 32, 3, false, 0, 0>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<256, 128, 32, 64, 64, 32, 6,  false, 0, 1>::Kernel>>());
    // Add(std::make_unique<KernelImpl<typename Config<256, 128, 32, 64, 64, 32, 3,  false, 1, 1>::Kernel>>());

    // Add(std::make_unique<KernelImpl<typename Config< 16,  16, 32,  16, 16, 32, 3,  false, 1, 1>::Kernel>>());
    // clang-format on
}

}  // namespace turbomind::gemm