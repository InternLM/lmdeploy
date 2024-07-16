// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::f16_u4g128_f16_tnt_sm70_s884()
{
    using namespace sm70_s884;
    {
        using Config = Sm70_s884<typename GetOperand<HMMA_884, OPERAND_A, half, kRowMajor, false>::Operand,
                                 Transform_Default,
                                 VoidOperand,
                                 typename GetOperand<HMMA_884, OPERAND_B, half, kRowMajor, false>::Operand,
                                 Transform_Default,
                                 VoidOperand,
                                 kRowMajor,
                                 half>;

        // Add(std::make_unique<KernelImpl<Config::Type<64, 64, 16, 2, 2, 1, 3, false, 0, 0>::Kernel>>());
        // Add(std::make_unique<KernelImpl<Config::Type<16, 16, 16, 1, 1, 1, 3, false, 1, 1>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<32, 32, 16, 2, 2, 1, 3, false>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 16, 2, 2, 1, 2, true>::Kernel>>());
    }

    {  // packed B
       // using Config =
       //     sm70_mma_884::SM70_MMA_884_F32<typename GetOperand<HMMA_884, OPERAND_A, half, kRowMajor, false>::Operand,
       //                                    Transform_Default,
       //                                    VoidOperand,
       //                                    typename GetOperand<HMMA_884, OPERAND_B, half, kRowMajor, true>::Operand,
       //                                    Transform_Default,
       //                                    VoidOperand,
       //                                    kRowMajor,
       //                                    half>;

        // Add(std::make_unique<KernelImpl<Config::Type<64, 64, 16, 2, 2, 1, 3, false, 0, 0>::Kernel>>());
    }

    {  // quant B
        using Config = Sm70_s884<typename GetOperand<HMMA_884, OPERAND_A, half, kRowMajor, false>::Operand,
                                 Transform_Default,
                                 VoidOperand,
                                 typename GetOperand<HMMA_884, OPERAND_B, uint4_t, kRowMajor, true>::Operand,
                                 Transform_HMMA_SIMT_B,
                                 typename GetOperand<HMMA_884, OPERAND_V, uint32_t, kColMajor, true>::Operand,
                                 kRowMajor,
                                 half>;

        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 16, 1, 4, 1, 2, false, 1, 128>::Kernel>>());
    }

    // {  // quant B
    //     using Config =
    //         sm70_mma_simt::SM70_MMA_F32<typename GetOperand<HMMA_SIMT, OPERAND_A, half, kRowMajor, false>::Operand,
    //                                     Transform_Default,
    //                                     VoidOperand,
    //                                     typename GetOperand<HMMA_SIMT, OPERAND_B, uint4_t, kRowMajor, true>::Operand,
    //                                     Transform_HMMA_SIMT_B,
    //                                     typename GetOperand<HMMA_SIMT, OPERAND_V, uint32_t, kColMajor,
    //                                     true>::Operand, half>;

    //     // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 16, 8, 1, 1, 3, false, 0, 0, 1, 128>::Kernel>>());
    //     Add(std::make_unique<KernelImpl<Config::Type<128, 128, 16, 4, 2, 1, 3, false, 0, 0, 1, 128>::Kernel>>());
    //     // Add(std::make_unique<KernelImpl<Config::Type<16, 32, 16, 1, 1, 1, 3, false, 0, 0, 1, 128>::Kernel>>());

    //     // Add(std::make_unique<KernelImpl<Config::Type<16, 16, 16, 1, 1, 1, 3, false, 0, 0, 1, 128>::Kernel>>());
    // }
}

}  // namespace turbomind::gemm
