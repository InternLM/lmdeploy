// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/config/sm70_mma_simt.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::reigster_sm70_sgemm_f16_f16_f16_tn()
{
    using namespace sm70_mma_simt;
    {
        using Config = SM70_MMA_F32<typename GetOperand<HMMA_SIMT, OPERAND_A, half, kRowMajor, false>::Operand,
                                    Transform_Default,
                                    VoidOperand,
                                    typename GetOperand<HMMA_SIMT, OPERAND_B, half, kRowMajor, false>::Operand,
                                    Transform_Default,
                                    VoidOperand,
                                    kRowMajor,
                                    half>;

        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 16, 8, 1, 1, 2, false>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<64, 128, 16, 2, 4, 1, 2, false>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<2, 128, 16, 1, 8, 1, 2, true>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<16, 16, 16, 1, 1, 1, 3, false, 0, 0>::Kernel>>());
    }

    {  // packed B
       // using Config =
       //     sm70_mma_simt::SM70_MMA_F32<typename GetOperand<HMMA_SIMT, OPERAND_A, half, kRowMajor, false>::Operand,
       //                                 Transform_Default,
       //                                 VoidOperand,
       //                                 typename GetOperand<HMMA_SIMT, OPERAND_B, half, kRowMajor, true>::Operand,
       //                                 Transform_Default,
       //                                 VoidOperand,
       //                                 half>;

        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 32, 8, 1, 1, 3, false, 0, 0>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 16, 8, 1, 1, 3, false, 0, 0>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<16, 16, 16, 1, 1, 1, 3, false, 0, 0>::Kernel>>());
    }

    {  // quant B
        // using Config = SM70_MMA_F32<typename GetOperand<HMMA_SIMT, OPERAND_A, half, kRowMajor, false>::Operand,
        //                             Transform_Default,
        //                             VoidOperand,
        //                             typename GetOperand<HMMA_SIMT, OPERAND_B, uint4_t, kRowMajor, true>::Operand,
        //                             Transform_HMMA_SIMT_B,
        //                             typename GetOperand<HMMA_SIMT, OPERAND_V, uint32_t, kColMajor, true>::Operand,
        //                             kRowMajor,
        //                             half>;

        using Operand_A    = typename GetOperand<HMMA_SIMT, OPERAND_A, half, kRowMajor, false>::Operand;
        using Operand_B_U4 = typename GetOperand<HMMA_SIMT, OPERAND_B, uint4_t, kRowMajor, true>::Operand;
        using Operand_V    = typename GetOperand<HMMA_SIMT, OPERAND_V, uint32_t, kColMajor, true>::Operand;

        using Config = SM70_MMA_F32<Operand_A,
                                    Transform_Default,
                                    VoidOperand,
                                    Operand_B_U4,
                                    Transform_HMMA_SIMT_B,
                                    Operand_V,
                                    kRowMajor,
                                    half>;

        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 16, 8, 1, 1, 3, true, 1, 128>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 16, 8, 1, 1, 2, true, 1, 128>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<80, 128, 32, 8, 1, 1, 2, true, 1, 128>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<64, 128, 32, 8, 1, 1, 2, true, 1, 128>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<32, 128, 32, 2, 4, 1, 2, true, 1, 128>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<32, 128, 32, 4, 2, 1, 2, true, 1, 128>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<16, 128, 64, 1, 4, 2, 2, true, 1, 128>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<16, 128, 32, 2, 4, 1, 2, true, 1, 128>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<8, 128, 64, 1, 4, 2, 2, true, 1, 128>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<4, 256, 64, 1, 8, 1, 2, true, 1, 128>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<2, 256, 64, 1, 8, 1, 2, true, 1, 128>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<1, 256, 64, 1, 8, 1, 2, true, 1, 128>::Kernel>>());
    }
}

}  // namespace turbomind::gemm
