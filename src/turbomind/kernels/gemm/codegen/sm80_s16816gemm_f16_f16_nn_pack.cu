// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/config/sm80_hmma_16816_pack.h"
#include "src/turbomind/kernels/gemm/registry.h"

namespace turbomind::gemm {

void Registry::reigster_sm80_s16816gemm_f16_f16_nn_packed()
{
    using Config = sm80_hmma_16816::SM80_HMMA_16816_F32<
        typename GetOperand<HMMA_16816, OPERAND_A, half, kColMajor, true>::Operand,
        typename GetOperand<HMMA_16816, OPERAND_B, half, kRowMajor, true>::Operand>;

    Add(std::make_unique<KernelImpl<Config::Type<256, 128, 64, 4, 2, 1, 3, false, 0, 0>::Kernel>>());

    // Add(std::make_unique<KernelImpl<typename Config<half, 32, 32, 32, 32, 32, 32, 3, false, 0, 0>::Kernel>>());
}

}  // namespace turbomind::gemm