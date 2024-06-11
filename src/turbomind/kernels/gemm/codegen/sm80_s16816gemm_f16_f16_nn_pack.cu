// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/config/sm80_hmma_16816_pack.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"

namespace turbomind::gemm {

void Registry::reigster_sm80_s16816gemm_f16_f16_nn_packed()
{
    {
        using Config = sm80_hmma_16816::SM80_HMMA_16816_F32<
            typename GetOperand<HMMA_16816, OPERAND_A, half, kColMajor, true>::Operand,
            Transform_Default,
            VoidOperand,
            typename GetOperand<HMMA_16816, OPERAND_B, half, kRowMajor, true>::Operand,
            Transform_Default,
            VoidOperand,
            half>;

        // Add(std::make_unique<KernelImpl<Config::Type<256, 128, 64, 4, 2, 1, 3, false, 0, 0>::Kernel>>());
        Add(std::make_unique<KernelImpl<Config::Type<128, 128, 32, 2, 2, 1, 3, false, 0, 0>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<16, 16, 32, 1, 1, 1, 3, false, 0, 0>::Kernel>>());
    }

    // {
    //     using Config = sm80_hmma_16816::SM80_HMMA_16816_F32<
    //         typename GetOperand<HMMA_16816, OPERAND_A, uint8_t, kColMajor, true>::Operand,
    //         Transform_HMMA_16816<0, 1>,
    //         typename GetOperand<HMMA_16816, OPERAND_U, uint32_t, kColMajor, true>::Operand,
    //         typename GetOperand<HMMA_16816, OPERAND_B, uint8_t, kRowMajor, true>::Operand,
    //         Transform_HMMA_16816<1, 0>,                                                      // Transform_Default,
    //         typename GetOperand<HMMA_16816, OPERAND_V, uint32_t, kColMajor, true>::Operand,  // VoidOperand,
    //         half>;

    //     // Add(std::make_unique<KernelImpl<Config::Type<256, 128, 64, 4, 2, 1, 3, false, 0, 0, 128>::Kernel>>());

    //     Add(std::make_unique<KernelImpl<Config::Type<16, 16, 32, 1, 1, 1, 3, false, 0, 0, 128, 128>::Kernel>>());
    // }

    {
        using Config = sm80_hmma_16816::SM80_HMMA_16816_F32<
            typename GetOperand<HMMA_16816, OPERAND_A, uint8_t, kColMajor, true>::Operand,
            Transform_HMMA_16816<0, 1>,
            typename GetOperand<HMMA_16816, OPERAND_U, uint32_t, kColMajor, true>::Operand,
            typename GetOperand<HMMA_16816, OPERAND_B, half, kRowMajor, true>::Operand,
            Transform_Default,
            VoidOperand,
            half>;

        // Add(std::make_unique<KernelImpl<Config::Type<256, 128, 64, 4, 2, 1, 3, false, 0, 0, 128, 1>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 32, 2, 2, 1, 3, false, 0, 0, 128, 1>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<32, 128, 32, 1, 4, 1, 3, false, 0, 0, 128, 1>::Kernel>>());

        Add(std::make_unique<KernelImpl<Config::Type<256, 128, 32, 8, 1, 1, 4, false, 0, 0, 128, 1>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 32, 2, 2, 1, 3, false, 0, 0, 128, 1>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<16, 16, 32, 1, 1, 1, 3, false, 0, 0, 128, 1>::Kernel>>());
    }

    // Add(std::make_unique<KernelImpl<typename Config<half, 32, 32, 32, 32, 32, 32, 3, false, 0, 0>::Kernel>>());
}

}  // namespace turbomind::gemm