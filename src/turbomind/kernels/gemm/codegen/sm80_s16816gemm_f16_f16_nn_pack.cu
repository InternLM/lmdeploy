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
        static constexpr int N = 16;

        using Config = SM80_HMMA_16816_F32<Operand_A_Pack<uint4_t, kRowMajor>,  // A
                                           Transform_HMMA_16816<0, 1>,          // tarnsform A
                                           Operand_U_Pack<uint32_t>,            // U
                                           Operand_B<half, kRowMajor, N>,       // B
                                           Transform_Default,                   // transform B
                                           VoidOperand,                         // V
                                           kColMajor,                           // order_C
                                           half>;                               // Tc

        using namespace cache_policy;

        // Add<Config::Type<128, 256, 32, 4, 2, 1, Default, Default, 6, false, 128, 1>>();
        // Add<Config::Type<256, 128, 32, 4, 2, 1, Default, Default, 5, false, 128, 1>>();
        // Add<Config::Type<256, 128, 64, 8, 1, 1, Default, Default, 3, false, 128, 1>>();
        // Add<Config::Type<256, 128, 32, 8, 1, 1, Default, Default, 5, false, 128, 1>>();
        // Add<Config::Type<256, 128, 64, 8, 1, 1, Default, Default, 3, true, 128, 1>>();
        // Add<Config::Type<256, 128, 32, 8, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<256, 96, 32, 8, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 192, 32, 4, 2, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 160, 32, 4, 2, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 128, 64, 4, 1, 2, Default, Default, 3, true, 128, 1>>();
        // Add<Config::Type<128, 128, 32, 2, 2, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 128, 32, 4, 1, 1, Default, Default, 5, true, 128, 1, 128, 64>>();
        // Add<Config::Type<128, 128, 32, 4, 1, 1, Default, Default, 3, true, 128, 1>>();
        // Add<Config::Type<128, 128, 32, 4, 1, 1, Default, Default, 3, true, 128, 1, 128, 64>>();
        // Add<Config::Type<128, 128, 32, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 96, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 64, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 64, 32, 4, 1, 1, Default, Default, 3, true, 128, 1>>();
        // Add<Config::Type<128, 64, 32, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 64, 64, 4, 1, 1, Stream, Default, 3, true, 128, 1>>();
        // Add<Config::Type<128, 32, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 32, 32, 4, 1, 1, Default, Default, 3, true, 128, 1>>();
        // Add<Config::Type<128, 32, 32, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 32, 64, 4, 1, 1, Stream, Default, 3, true, 128, 1>>();
        // Add<Config::Type<128, N, 32, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, N, 64, 4, 1, 1, Stream, Default, 6, true, 128, 1>>();
        // Add<Config::Type<128, N, 64, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, N, 64, 4, 1, 1, Stream, Default, 3, true, 128, 1>>();
        // Add<Config::Type<64, N, 64, 2, 1, 2, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<64, N, 64, 2, 1, 2, Stream, Default, 3, true, 128, 1>>();

        // Add<Config::Type<256, 128, 64, 8, 1, 1, Default, Default, 3, true, 128, 1>>();
        // Add<Config::Type<256, 128, 32, 4, 2, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<256, 128, 32, 8, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<256, 112, 32, 8, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<256, 96, 32, 8, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<256, 80, 32, 8, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<256, 64, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<256, 48, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<256, 32, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 256, 32, 4, 2, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 224, 32, 4, 2, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 192, 32, 4, 2, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 160, 32, 4, 2, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 128, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 112, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 96, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 80, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 64, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 48, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 32, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, N, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 256, 32, 4, 2, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 224, 32, 4, 2, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 192, 32, 4, 2, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 160, 32, 4, 2, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 128, 32, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 112, 32, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 96, 32, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 80, 32, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 64, 32, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 48, 32, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 32, 32, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, N, 32, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, 32, 64, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();
        // Add<Config::Type<128, N, 64, 4, 1, 1, Stream, Default, 5, true, 128, 1>>();

        // Add<Config::Type<64, 256, 64, 2, 4, 1, Default, Default, 3, true, 128, 1>>();
        // Add<Config::Type<64, 224, 32, 4, 2, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<64, 192, 64, 2, 4, 1, Default, Default, 3, true, 128, 1>>();
        // Add<Config::Type<64, 160, 32, 4, 2, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<64, 128, 64, 2, 2, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<64, 112, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<64, 96, 64, 2, 2, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<64, 80, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<64, 64, 64, 2, 2, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<64, 48, 32, 4, 1, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<64, 32, 64, 2, 2, 1, Default, Default, 5, true, 128, 1>>();
        // Add<Config::Type<64, N, 64, 2, 1, 2, Default, Default, 5, true, 128, 1>>();
    }

    {
        // using Config = sm80_hmma_16816::SM80_HMMA_16816_F32<
        //     typename GetOperand<HMMA_16816, OPERAND_A, half, kRowMajor, false>::Operand,
        //     Transform_Default,
        //     VoidOperand,
        //     typename GetOperand<HMMA_16816, OPERAND_B, uint4_t, kColMajor, true>::Operand,
        //     Transform_HMMA_16816<1, 0>,                                                      // Transform_Default,
        //     typename GetOperand<HMMA_16816, OPERAND_V, uint32_t, kColMajor, true>::Operand,  // VoidOperand,
        //     kRowMajor,
        //     half>;

        using Config = SM80_HMMA_16816_F32<Operand_A<half, kRowMajor>,          // A
                                           Transform_Default,                   // tarnsform A
                                           VoidOperand,                         // U
                                           Operand_B_Pack<uint4_t, kRowMajor>,  // B
                                           Transform_HMMA_16816<1, 0>,          // transform B
                                           Operand_U_Pack<uint32_t>,            // V
                                           kRowMajor,                           // order_C
                                           half>;                               // Tc

        using namespace cache_policy;

        // Add<Config::Type<128, 256, 64, 1, 8, 1, Default, Default, 3, false, 1, 128>>();
        // Add<Config::Type<128, 256, 32, 1, 8, 1, Default, Default, 5, false, 1, 128>>();
        // Add<Config::Type<96, 256, 64, 1, 8, 1, Default, Default, 3, false, 1, 128>>();
        // Add<Config::Type<96, 256, 32, 1, 8, 1, Default, Default, 5, false, 1, 128>>();

        Add<Config::Type<128, 256, 64, 1, 8, 1, Default, Default, 3, true, 1, 128>>();  // 6
        Add<Config::Type<128, 256, 32, 1, 8, 1, Default, Default, 5, true, 1, 128>>();  // 4
        Add<Config::Type<128, 256, 32, 1, 8, 1, Default, Default, 3, true, 1, 128>>();  // 2
        // Add<Config::Type<96, 256, 64, 1, 8, 1, Default, Default, 3, true, 1, 128>>();
        // Add<Config::Type<96, 256, 32, 1, 8, 1, Default, Default, 5, true, 1, 128>>();

        // Add<Config::Type<256, 128, 32, 2, 4, 1, Default, Default, 5, true, 1, 128>>();
        // Add<Config::Type<192, 128, 32, 2, 4, 1, Default, Default, 5, true, 1, 128>>();
        // Add<Config::Type<160, 128, 32, 2, 4, 1, Default, Default, 5, true, 1, 128>>();
        Add<Config::Type<128, 128, 32, 1, 4, 1, Default, Default, 5, true, 1, 128>>();  // 7
        Add<Config::Type<96, 128, 64, 1, 4, 1, Default, Default, 3, true, 1, 128>>();   // 3
        Add<Config::Type<96, 128, 32, 1, 4, 1, Default, Default, 5, true, 1, 128>>();   // 1
        Add<Config::Type<64, 128, 64, 1, 4, 1, Default, Default, 3, true, 1, 128>>();   // 7
        // Add<Config::Type<64, 128, 32, 1, 4, 1, Default, Default, 5, true, 1, 128>>();

        // Add<Config::Type<256, 128, 32, 2, 4, 1, Default, Stream, 5, true, 1, 128>>();
        Add<Config::Type<192, 128, 32, 2, 4, 1, Default, Stream, 5, true, 1, 128>>();  // 1
        // Add<Config::Type<160, 128, 32, 2, 4, 1, Default, Stream, 5, true, 1, 128>>();
        Add<Config::Type<128, 128, 32, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();  // 7
        Add<Config::Type<96, 128, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();   // 3
        Add<Config::Type<96, 128, 32, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();   // 1
        Add<Config::Type<64, 128, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();   // 7
        // Add<Config::Type<64, 128, 32, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();
        // Add<Config::Type<32, 128, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
        Add<Config::Type<32, 128, 32, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();  // 2
        Add<Config::Type<16, 128, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();  // 1
        // Add<Config::Type<16, 128, 32, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();
        // Add<Config::Type<16, 64, 64, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();
        // Add<Config::Type<16, 64, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
        Add<Config::Type<16, 64, 128, 1, 2, 2, Default, Stream, 3, true, 1, 128>>();  // 22

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
