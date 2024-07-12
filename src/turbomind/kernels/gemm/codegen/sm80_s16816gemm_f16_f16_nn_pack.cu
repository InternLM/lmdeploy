// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/config/sm80_hmma_16816.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::reigster_sm80_s16816gemm_f16_f16_nn_packed()
{
    using namespace sm80_hmma_16816;

    {
        using Config = SM80_HMMA_16816_F32<typename GetOperand<HMMA_16816, OPERAND_A, half, kColMajor, false>::Operand,
                                           Transform_Default,
                                           VoidOperand,
                                           typename GetOperand<HMMA_16816, OPERAND_B, half, kRowMajor, false>::Operand,
                                           Transform_Default,
                                           VoidOperand,
                                           kColMajor,
                                           half>;

        using namespace cache_policy;

        Add<Config::Type<256, 128, 64, 4, 2, 1, Default, Default, 3, false, 1, 1>>();

        // Add<Config::Type<128, 128, 32, 2, 2, 1, Default, Default, 3, false, 1, 1>>();

        // Add<Config::Type<128, 16, 64, 4, 1, 1, Default, Default, 3, false, 1, 1>>();

        // Add<Config::Type<32, 16, 32, 1, 1, 1, Default, Default, 3, false, 1, 1>>();
    }

    {
        // using Config = sm80_hmma_16816::SM80_HMMA_16816_F32<
        //     typename GetOperand<HMMA_16816, OPERAND_A, uint4_t, kColMajor, true>::Operand,
        //     Transform_HMMA_16816<0, 1>,
        //     typename GetOperand<HMMA_16816, OPERAND_U, uint32_t, kColMajor, true>::Operand,
        //     typename GetOperand<HMMA_16816, OPERAND_B, uint4_t, kRowMajor, true>::Operand,
        //     Transform_HMMA_16816<1, 0>,                                                      // Transform_Default,
        //     typename GetOperand<HMMA_16816, OPERAND_V, uint32_t, kColMajor, true>::Operand,  // VoidOperand,
        //     kRowMajor,
        //     half>;

        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 32, 2, 2, 1, 3, false, 0, 0, 128, 128>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<16, 16, 32, 1, 1, 1, 3, false, 0, 0, 128, 128>::Kernel>>());
    }

    {
        using Config = sm80_hmma_16816::SM80_HMMA_16816_F32<
            typename GetOperand<HMMA_16816, OPERAND_A, uint4_t, kColMajor, true>::Operand,
            Transform_HMMA_16816<0, 1>,
            typename GetOperand<HMMA_16816, OPERAND_U, uint32_t, kColMajor, true>::Operand,
            typename GetOperand<HMMA_16816, OPERAND_B, half, kRowMajor, false>::Operand,  // non-packed
            Transform_Default,
            VoidOperand,
            kRowMajor,
            half>;

        // Add(std::make_unique<KernelImpl<Config::Type<256, 128, 64, 4, 2, 1, 3, false, 0, 0, 128, 1>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 32, 2, 2, 1, 3, false, 0, 0, 128, 1>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<32, 128, 32, 1, 4, 1, 3, false, 0, 0, 128, 1>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<256, 128, 32, 8, 1, 1, 3, false, 0, 0, 128, 1>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<256, 128, 32, 4, 2, 1, 3, false, 0, 0, 128, 1>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<128, 128, 32, 2, 2, 1, 3, false, 0, 0, 128, 1>::Kernel>>());

        // Add(std::make_unique<KernelImpl<Config::Type<64, 16, 32, 1, 1, 1, 3, false, 0, 0, 128, 1>::Kernel>>());
    }

    {
        using Config = sm80_hmma_16816::SM80_HMMA_16816_F32<
            typename GetOperand<HMMA_16816, OPERAND_A, half, kRowMajor, false>::Operand,
            Transform_Default,
            VoidOperand,
            typename GetOperand<HMMA_16816, OPERAND_B, uint4_t, kColMajor, true>::Operand,
            Transform_HMMA_16816<1, 0>,                                                      // Transform_Default,
            typename GetOperand<HMMA_16816, OPERAND_V, uint32_t, kColMajor, true>::Operand,  // VoidOperand,
            kRowMajor,
            half>;

        using namespace cache_policy;

        // Add<Config::Type<128, 256, 32, 1, 8, 1, Default, Default, 5, false, 1, 128>::Kernel>();
        // Add<Config::Type<128, 128, 32, 1, 4, 1, 3, false, 1, 128>::Kernel>();
        // Add<Config::Type<128, 256, 32, 1, 8, 1, Default, Default, 5, false, 1, 128>>();
        // Add<Config::Type<128, 128, 32, 1, 4, 1, 5, true, 1, 128>::Kernel>();
        // Add<Config::Type<128, 128, 32, 1, 4, 1, 3, true, 1, 128>::Kernel>();
        // Add<Config::Type<128, 128, 64, 1, 4, 1, 3, true, 1, 128>::Kernel>();

        // Add<Config::Type<128, 128, 32, 1, 4, 1, Default, Default, 5, true, 1, 128>>();
        // Add<Config::Type<128, 128, 32, 1, 4, 1, Default, Default, 3, true, 1, 128>>();
        // Add<Config::Type<128, 128, 64, 1, 4, 1, Default, Default, 3, true, 1, 128>>();
        // Add<Config::Type<64, 128, 64, 1, 4, 1, Default, Default, 3, true, 1, 128>>();
        // Add<Config::Type<64, 128, 32, 1, 4, 1, Default, Default, 3, true, 1, 128>>();

        // Add<Config::Type<128, 128, 32, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();
        // Add<Config::Type<128, 128, 32, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
        // Add<Config::Type<128, 128, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
        // Add<Config::Type<64, 128, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
        // Add<Config::Type<64, 128, 32, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
        // Add<Config::Type<64, 128, 32, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();
        // Add<Config::Type<32, 128, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
        // Add<Config::Type<32, 128, 32, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();
        // Add<Config::Type<16, 128, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
        // Add<Config::Type<16, 128, 32, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();
        // Add<Config::Type<16, 64, 64, 1, 2, 2, Default, Stream, 5, true, 1, 128>>();

        // Add<Config::Type<64, 64, 64, 1, 2, 2, 5, true, 1, 128>::Kernel>();
        // Add<Config::Type<32, 64, 64, 1, 2, 2, 5, true, 1, 128>::Kernel>();
    }
}

}  // namespace turbomind::gemm
