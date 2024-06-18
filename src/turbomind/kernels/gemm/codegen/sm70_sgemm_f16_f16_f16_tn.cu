// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/config/sm70_mma_simt.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::reigster_sm70_sgemm_f16_f16_f16_tn()
{
    {
        using Config =
            sm70_mma_simt::SM70_MMA_F32<typename GetOperand<HMMA_SIMT, OPERAND_A, half, kRowMajor, false>::Operand,
                                        Transform_Default,
                                        VoidOperand,
                                        typename GetOperand<HMMA_SIMT, OPERAND_B, half, kRowMajor, false>::Operand,
                                        Transform_Default,
                                        VoidOperand,
                                        half>;

        Add(std::make_unique<KernelImpl<Config::Type<128, 128, 16, 4, 2, 1, 3, false, 0, 0>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<16, 16, 16, 1, 1, 1, 3, false, 0, 0>::Kernel>>());
    }

    {  // packed B
        using Config =
            sm70_mma_simt::SM70_MMA_F32<typename GetOperand<HMMA_SIMT, OPERAND_A, half, kRowMajor, false>::Operand,
                                        Transform_Default,
                                        VoidOperand,
                                        typename GetOperand<HMMA_SIMT, OPERAND_B, half, kRowMajor, true>::Operand,
                                        Transform_Default,
                                        VoidOperand,
                                        half>;

        Add(std::make_unique<KernelImpl<Config::Type<128, 128, 16, 4, 2, 1, 3, false, 0, 0>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<16, 16, 16, 1, 1, 1, 3, false, 0, 0>::Kernel>>());
    }
}

}  // namespace turbomind::gemm