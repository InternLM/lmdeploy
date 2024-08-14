// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm80_s16816;
template<int N>
using Config_ = Sm80_s16816<Sm80,
                            Operand_A_Pack<uint4_t, kColMajor>,  // A
                            Transform_HMMA_16816<0, 1>,          // tarnsform A
                            Operand_UV_Pack<uint32_t, false>,    // U
                            Operand_B<half, kRowMajor, N>,       // B
                            Transform_Default,                   // transform B
                            VoidOperand,                         // V
                            kColMajor,                           // order_C
                            half,                                // Tc
                            CtaMapN>;

void Registry::u4g128_f16_f16_nnn_sm80_s16816()
{
    // ! Must be M-major MMA
#if 0
    using namespace cache_policy;

    using C16 = Config_<16>;
    using S   = Stream;
    using D   = Default;

    // clang-format off
    Add<C16::Type<256, 128, 32, 8, 1, 1, D, D, 3, true, 128, 1, 128, 128>>();
    // Add<C16::Type<256, 128, 64, 8, 1, 1, D, D, 3, true, 128, 1, 128, 128>>();
    Add<C16::Type<128, 128, 32, 4, 1, 1, D, D, 6, true, 128, 1, 128, 64>>();
    Add<C16::Type<128, 128, 32, 4, 1, 1, D, D, 4, true, 128, 1, 128, 64>>();
    Add<C16::Type<128, 128, 32, 4, 1, 1, S, D, 6, true, 128, 1, 128, 64>>();
    Add<C16::Type<128, 128, 32, 4, 1, 1, S, D, 4, true, 128, 1, 128, 64>>();

    Add<C16::Type<128, 96, 64, 4, 1, 1, D, D, 3, true, 128, 1, 128, 48>>();
    Add<C16::Type<128, 96, 64, 4, 1, 1, S, D, 3, true, 128, 1, 128, 48>>();

    Add<C16::Type<256, 64, 32, 4, 1, 1, D, D, 3, true, 128, 1, 128, 64>>();
    Add<C16::Type<256, 64, 64, 4, 1, 1, D, D, 3, true, 128, 1, 128, 64>>();
    Add<C16::Type<128, 64, 32, 4, 1, 1, D, D, 5, true, 128, 1>>();
    Add<C16::Type< 64, 64, 64, 2, 1, 2, D, D, 3, true, 128, 1>>();
    Add<C16::Type<256, 64, 32, 4, 1, 1, S, D, 3, true, 128, 1, 128, 64>>();
    Add<C16::Type<256, 64, 64, 4, 1, 1, S, D, 3, true, 128, 1, 128, 64>>();
    Add<C16::Type<128, 64, 32, 4, 1, 1, S, D, 5, true, 128, 1>>();
    Add<C16::Type< 64, 64, 64, 2, 1, 2, S, D, 3, true, 128, 1>>();

    Add<C16::Type<256, 48,  64, 4, 1, 1, S, D, 3, true, 128, 1, 128, 48>>();
    Add<C16::Type<128, 48,  64, 4, 1, 1, S, D, 4, true, 128, 1>>();
    Add<C16::Type<128, 48, 128, 4, 1, 2, S, D, 3, true, 128, 1>>();
    Add<C16::Type< 64, 48, 128, 2, 1, 2, S, D, 3, true, 128, 1>>();

    Add<C16::Type<128, 32,  64, 4, 1, 1, S, D, 4, true, 128, 1>>();
    Add<C16::Type<128, 32, 128, 4, 1, 2, S, D, 5, true, 128, 1>>();
    Add<C16::Type< 64, 32, 128, 2, 1, 2, S, D, 4, true, 128, 1>>();
    Add<C16::Type< 64, 32, 128, 2, 1, 2, S, D, 3, true, 128, 1>>();

    Add<C16::Type<128, 16,  64, 4, 1, 1, S, D, 4, true, 128, 1>>();
    Add<C16::Type<128, 16,  64, 4, 1, 1, S, D, 3, true, 128, 1>>();
    Add<C16::Type<128, 16, 128, 4, 1, 2, S, D, 4, true, 128, 1>>();
    Add<C16::Type< 64, 16, 128, 2, 1, 2, S, D, 4, true, 128, 1>>();
    Add<C16::Type< 64, 16, 128, 2, 1, 2, S, D, 3, true, 128, 1>>();

    using C8 = Config_<8>;
    Add<C8::Type<128, 8, 128, 4, 1, 2, S, D, 4, true, 128, 1>>();
    Add<C8::Type<128, 8,  64, 4, 1, 1, S, D, 3, true, 128, 1>>();
    Add<C8::Type<128, 8,  32, 4, 1, 1, S, D, 6, true, 128, 1>>();
    Add<C8::Type< 64, 8, 128, 2, 1, 2, S, D, 5, true, 128, 1>>();
    Add<C8::Type< 32, 8, 128, 1, 1, 4, S, D, 5, true, 128, 1>>();

    // clang-format on

#endif
}

// sm80_u4g128_f16_f16_nnn_256x128x32_3_s16816_8x1x1_c128x128_a32x1x32_00: 48
// sm80_u4g128_f16_f16_nnn_256x48x64_3_s16816_4x1x1_c128x48_a32x1x64_10: 22
// sm80_u4g128_f16_f16_nnn_128x64x32_5_s16816_4x1x1_c128x64_a32x1x32_10: 17
// sm80_u4g128_f16_f16_nnn_128x96x64_3_s16816_4x1x1_c128x48_a32x1x64_10: 17
// sm80_u4g128_f16_f16_nnn_128x128x32_6_s16816_4x1x1_c128x64_a32x1x32_10: 17
// sm80_u4g128_f16_f16_nnn_128x32x64_4_s16816_4x1x1_c128x32_a32x1x64_10: 16
// sm80_u4g128_f16_f16_nnn_128x48x128_3_s16816_4x1x2_c128x48_a32x1x128_10: 16
// sm80_u4g128_f16_f16_nnn_128x128x32_6_s16816_4x1x1_c128x64_a32x1x32_00: 15
// sm80_u4g128_f16_f16_nnn_256x64x32_3_s16816_4x1x1_c128x64_a32x1x32_00: 14
// sm80_u4g128_f16_f16_nnn_256x64x32_3_s16816_4x1x1_c128x64_a32x1x32_10: 13
// sm80_u4g128_f16_f16_nnn_64x8x128_5_s16816_2x1x2_c64x8_a32x1x128_10: 11
// sm80_u4g128_f16_f16_nnn_128x96x64_3_s16816_4x1x1_c128x48_a32x1x64_00: 11
// sm80_u4g128_f16_f16_nnn_128x128x32_4_s16816_4x1x1_c128x64_a32x1x32_00: 11
// sm80_u4g128_f16_f16_nnn_128x16x128_4_s16816_4x1x2_c128x16_a32x1x128_10: 10
// sm80_u4g128_f16_f16_nnn_64x64x64_3_s16816_2x1x2_c64x64_a32x1x64_10: 10
// sm80_u4g128_f16_f16_nnn_128x64x32_5_s16816_4x1x1_c128x64_a32x1x32_00: 10
// sm80_u4g128_f16_f16_nnn_64x16x128_4_s16816_2x1x2_c64x16_a32x1x128_10: 8
// sm80_u4g128_f16_f16_nnn_64x32x128_3_s16816_2x1x2_c64x32_a32x1x128_10: 8
// sm80_u4g128_f16_f16_nnn_128x48x64_4_s16816_4x1x1_c128x48_a32x1x64_10: 8
// sm80_u4g128_f16_f16_nnn_64x32x128_4_s16816_2x1x2_c64x32_a32x1x128_10: 7
// sm80_u4g128_f16_f16_nnn_128x32x128_5_s16816_4x1x2_c128x32_a32x1x128_10: 7
// sm80_u4g128_f16_f16_nnn_64x16x128_3_s16816_2x1x2_c64x16_a32x1x128_10: 6
// sm80_u4g128_f16_f16_nnn_128x8x128_4_s16816_4x1x2_c128x8_a32x1x128_10: 5
// sm80_u4g128_f16_f16_nnn_128x16x64_4_s16816_4x1x1_c128x16_a32x1x64_10: 5
// sm80_u4g128_f16_f16_nnn_128x8x32_6_s16816_4x1x1_c128x8_a32x1x32_10: 4
// sm80_u4g128_f16_f16_nnn_128x8x64_3_s16816_4x1x1_c128x8_a32x1x64_10: 4
// sm80_u4g128_f16_f16_nnn_128x16x64_3_s16816_4x1x1_c128x16_a32x1x64_10: 4
// sm80_u4g128_f16_f16_nnn_32x8x128_5_s16816_1x1x4_c32x8_a32x1x128_10: 3
// sm80_u4g128_f16_f16_nnn_64x48x128_3_s16816_2x1x2_c64x48_a32x1x128_10: 3
// sm80_u4g128_f16_f16_nnn_256x64x64_3_s16816_4x1x1_c128x64_a32x1x64_10: 2
// sm80_u4g128_f16_f16_nnn_128x128x32_4_s16816_4x1x1_c128x64_a32x1x32_10: 2
// sm80_u4g128_f16_f16_nnn_64x64x64_3_s16816_2x1x2_c64x64_a32x1x64_00: 1
// sm80_u4g128_f16_f16_nnn_256x64x64_3_s16816_4x1x1_c128x64_a32x1x64_00: 1
// sm80_u4g128_f16_f16_nnn_256x128x64_3_s16816_8x1x1_c128x128_a32x1x64_00: 0

}  // namespace turbomind::gemm
