// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::f16_u4g128_f16_tnt_sm80_s16816()
{
    using namespace sm80_s16816;
    using namespace cache_policy;
    using S = cache_policy::Stream;
    using D = cache_policy::Default;

    using C = Sm80_s16816<Sm80,
                          Operand_A<half, kRowMajor>,          // A
                          Transform_Default,                   // tarnsform A
                          VoidOperand,                         // U
                          Operand_B_Pack<uint4_t, kColMajor>,  // B
                          Transform_HMMA_16816<1, 0>,          // transform B
                          Operand_UV_Pack<uint32_t, true>,     // V
                          kRowMajor,                           // order_C
                          half>;                               // Tc

    // clang-format off
    // Add<C::Type<128, 256,  64, 1, 8, 1, D, S, 3, true, 1, 128>>(); // 0/0
    Add<C::Type<128, 256,  32, 1, 8, 1, D, D, 3, true, 1, 128, 128, 128>>(); // 30/3
    Add<C::Type<128, 256,  32, 1, 8, 1, D, D, 4, true, 1, 128, 128, 128>>(); // --/20
    Add<C::Type<128, 128,  32, 1, 4, 1, D, D, 3, true, 1, 128, 64, 128>>();  // --/13
    Add<C::Type<128, 128,  32, 1, 4, 1, D, S, 4, true, 1, 128, 64, 128>>();  // 21/13
    Add<C::Type<128, 128,  64, 1, 4, 2, D, S, 3, true, 1, 128, 64, 128>>();  // 6/6

    Add<C::Type<96, 256,  32, 1, 8, 1, D, D, 4, true, 1, 128>>();  // --/3
    Add<C::Type<96, 256,  32, 1, 8, 1, D, S, 3, true, 1, 128>>();  // 13/13
    Add<C::Type<96, 128,  32, 1, 4, 1, D, S, 4, true, 1, 128>>();  // 14/10
    Add<C::Type<96, 128, 128, 1, 4, 2, D, S, 3, true, 1, 128>>();  // 2/2

    Add<C::Type<64, 256,  32, 1, 4, 1, D, D, 3, true, 1, 128, 64, 128>>(); // --/21
    Add<C::Type<64, 256,  32, 1, 4, 1, D, S, 4, true, 1, 128, 64, 128>>(); // 27/13
    Add<C::Type<64, 128,  32, 1, 4, 1, D, S, 4, true, 1, 128>>();  // 8/5
    Add<C::Type<64, 128,  64, 1, 4, 1, D, S, 3, true, 1, 128>>();  // 7/5
    Add<C::Type<64, 128, 128, 1, 4, 2, D, S, 3, true, 1, 128>>();  // 6/7
    Add<C::Type<64,  64,  64, 1, 2, 2, D, S, 6, true, 1, 128>>();

    Add<C::Type<48, 256,  64, 1, 4, 1, D, S, 3, true, 1, 128, 48, 128>>(); // 1/1
    Add<C::Type<48, 128,  64, 1, 4, 1, D, S, 4, true, 1, 128>>();  // 1/1
    Add<C::Type<48, 128, 128, 1, 4, 2, D, S, 3, true, 1, 128>>();  // 4/4
    Add<C::Type<48,  64, 128, 1, 2, 2, D, S, 4, true, 1, 128>>();

    Add<C::Type<32, 256,  64, 1, 4, 1, D, S, 3, true, 1, 128>>();
    Add<C::Type<32, 128,  64, 1, 4, 1, D, S, 4, true, 1, 128>>();
    Add<C::Type<32, 128, 128, 1, 4, 2, D, S, 3, true, 1, 128>>();
    Add<C::Type<32,  64, 128, 1, 2, 2, D, S, 3, true, 1, 128>>();
    Add<C::Type<32,  64, 128, 1, 2, 2, D, S, 4, true, 1, 128>>();

    Add<C::Type<16, 128,  64, 1, 4, 1, D, S, 4, true, 1, 128>>();
    Add<C::Type<16, 128, 128, 1, 4, 2, D, S, 3, true, 1, 128>>();
    Add<C::Type<16, 128, 128, 1, 4, 2, D, S, 4, true, 1, 128>>();
    Add<C::Type<16,  64, 128, 1, 2, 2, D, S, 3, true, 1, 128>>();
    Add<C::Type<16,  64, 128, 1, 2, 2, D, S, 4, true, 1, 128>>();
    // clang-format on
}

// sm80_f16_u4g128_f16_ttt_128x256x32_4_s16816_1x8x1_c128x128_a1x32x32_00: 46
// sm80_f16_u4g128_f16_ttt_128x128x32_3_s16816_1x4x1_c64x128_a1x32x32_00: 27
// sm80_f16_u4g128_f16_ttt_64x256x32_3_s16816_1x4x1_c64x128_a1x32x32_00: 21
// sm80_f16_u4g128_f16_ttt_64x256x32_4_s16816_1x4x1_c64x128_a1x32x32_01: 19
// sm80_f16_u4g128_f16_ttt_16x128x128_4_s16816_1x4x2_c16x128_a1x32x128_01: 17
// sm80_f16_u4g128_f16_ttt_32x128x128_3_s16816_1x4x2_c32x128_a1x32x128_01: 16
// sm80_f16_u4g128_f16_ttt_64x128x128_3_s16816_1x4x2_c64x128_a1x32x128_01: 16
// sm80_f16_u4g128_f16_ttt_96x128x32_4_s16816_1x4x1_c96x128_a1x32x32_01: 16
// sm80_f16_u4g128_f16_ttt_96x256x32_4_s16816_1x8x1_c96x256_a1x32x32_00: 15
// sm80_f16_u4g128_f16_ttt_16x64x128_3_s16816_1x2x2_c16x64_a1x32x128_01: 13
// sm80_f16_u4g128_f16_ttt_16x128x64_4_s16816_1x4x1_c16x128_a1x32x64_01: 13
// sm80_f16_u4g128_f16_ttt_48x128x128_3_s16816_1x4x2_c48x128_a1x32x128_01: 13
// sm80_f16_u4g128_f16_ttt_48x256x64_3_s16816_1x4x1_c48x128_a1x32x64_01: 13
// sm80_f16_u4g128_f16_ttt_16x64x128_4_s16816_1x2x2_c16x64_a1x32x128_01: 11
// sm80_f16_u4g128_f16_ttt_64x128x64_3_s16816_1x4x1_c64x128_a1x32x64_01: 9
// sm80_f16_u4g128_f16_ttt_128x128x32_4_s16816_1x4x1_c64x128_a1x32x32_01: 9
// sm80_f16_u4g128_f16_ttt_96x128x128_3_s16816_1x4x2_c96x128_a1x32x128_01: 7
// sm80_f16_u4g128_f16_ttt_96x256x32_3_s16816_1x8x1_c96x256_a1x32x32_01: 7
// sm80_f16_u4g128_f16_ttt_48x128x64_4_s16816_1x4x1_c48x128_a1x32x64_01: 6
// sm80_f16_u4g128_f16_ttt_32x64x128_4_s16816_1x2x2_c32x64_a1x32x128_01: 5
// sm80_f16_u4g128_f16_ttt_32x256x64_3_s16816_1x4x1_c32x256_a1x32x64_01: 5
// sm80_f16_u4g128_f16_ttt_64x64x64_6_s16816_1x2x2_c64x64_a1x32x64_01: 5
// sm80_f16_u4g128_f16_ttt_16x128x128_3_s16816_1x4x2_c16x128_a1x32x128_01: 4
// sm80_f16_u4g128_f16_ttt_32x128x64_4_s16816_1x4x1_c32x128_a1x32x64_01: 4
// sm80_f16_u4g128_f16_ttt_48x64x128_4_s16816_1x2x2_c48x64_a1x32x128_01: 4
// sm80_f16_u4g128_f16_ttt_64x128x32_4_s16816_1x4x1_c64x128_a1x32x32_01: 4
// sm80_f16_u4g128_f16_ttt_128x128x64_3_s16816_1x4x2_c64x128_a1x32x64_01: 4
// sm80_f16_u4g128_f16_ttt_128x256x32_3_s16816_1x8x1_c128x128_a1x32x32_00: 4
// sm80_f16_u4g128_f16_ttt_32x64x128_3_s16816_1x2x2_c32x64_a1x32x128_01: 3
// sm80_f16_u4g128_f16_ttt_128x256x64_3_s16816_1x8x1_c128x256_a1x32x64_01: 0

}  // namespace turbomind::gemm
