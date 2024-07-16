// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm75_s16816.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::f16_u4g128_f16_tnt_sm75_s16816()
{

    {  // fp x fp
       // using Config = sm75_hmma_1688::SM75_HMMA_1688_F32<
       //     typename GetOperand<HMMA_16816, OPERAND_A, half, kRowMajor, false>::Operand,
       //     Transform_Default,
       //     VoidOperand,
       //     typename GetOperand<HMMA_16816, OPERAND_B, half, kColMajor, false>::Operand,
       //     Transform_Default,
       //     VoidOperand,
       //     kRowMajor,
       //     half>;

        // Add(std::make_unique<KernelImpl<Config::Type<128, 256, 64, 2, 4, 1, 2, false, 1, 1>::Kernel>>());
    }

    {  // fp x u4
       // using Config = sm75_hmma_1688::SM75_HMMA_1688_F32<
       //     typename GetOperand<HMMA_16816, OPERAND_A, half, kRowMajor, false>::Operand,
       //     Transform_Default,
       //     VoidOperand,
       //     typename GetOperand<HMMA_16816, OPERAND_B, uint4_t, kColMajor, true>::Operand,
       //     Transform_HMMA_16816<1, 0>,                                                      // Transform_Default,
       //     typename GetOperand<HMMA_16816, OPERAND_V, uint32_t, kColMajor, true>::Operand,  // VoidOperand,
       //     kRowMajor,
       //     half>;

        // Add(std::make_unique<KernelImpl<Config::Type<128, 256, 64, 1, 8, 1, 2, false, 1, 128>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 32, 1, 4, 1, 2, false, 1, 128>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<16, 128, 64, 1, 4, 1, 2, true, 1, 128>::Kernel>>());
    }
}

}  // namespace turbomind::gemm
