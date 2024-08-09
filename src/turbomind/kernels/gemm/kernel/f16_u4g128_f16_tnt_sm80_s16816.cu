// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/transform.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

void Registry::f16_u4g128_f16_tnt_sm80_s16816()
{
    using namespace sm80_s16816;

    using Config = Sm80_s16816<Sm80,
                               Operand_A<half, kRowMajor>,          // A
                               Transform_Default,                   // tarnsform A
                               VoidOperand,                         // U
                               Operand_B_Pack<uint4_t, kRowMajor>,  // B
                               Transform_HMMA_16816<1, 0>,          // transform B
                               Operand_UV_Pack<uint32_t, true>,     // V
                               kRowMajor,                           // order_C
                               half>;                               // Tc

    using namespace cache_policy;

    // N = 256
    Add<Config::Type<128, 256, 32, 1, 8, 1, Default, Default, 3, true, 1, 128>>();
    Add<Config::Type<128, 256, 32, 1, 8, 1, Default, Default, 5, true, 1, 128>>();
    Add<Config::Type<128, 256, 64, 1, 8, 1, Default, Default, 3, true, 1, 128>>();

    // N = 128, non-streaming
    Add<Config::Type<128, 128, 32, 1, 4, 1, Default, Default, 6, true, 1, 128>>();
    Add<Config::Type<96, 128, 32, 1, 4, 1, Default, Default, 5, true, 1, 128>>();
    Add<Config::Type<96, 128, 64, 1, 4, 1, Default, Default, 3, true, 1, 128>>();
    Add<Config::Type<64, 128, 64, 1, 4, 1, Default, Default, 3, true, 1, 128>>();
    

    if constexpr (1) { // kernels that use smaller smem for `sm_86` and `sm_89`
        Add<Config::Type<128, 256, 32, 1, 8, 1, Default, Default, 3, true, 1, 128, 128, 128>>();
        Add<Config::Type<128, 256, 32, 1, 8, 1, Default, Default, 6, true, 1, 128, 128, 128>>();
        Add<Config::Type<128, 128, 32, 1, 4, 1, Default, Default, 4, true, 1, 128, 64, 128>>();
        Add<Config::Type<64, 128, 32, 1, 4, 1, Default, Default, 4, true, 1, 128>>(); 
    }

    // N = 128, streaming
    Add<Config::Type<128, 128, 32, 1, 4, 1, Default, Stream, 6, true, 1, 128>>();
    Add<Config::Type<128, 128, 32, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
    Add<Config::Type<96, 128, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
    Add<Config::Type<96, 128, 32, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();
    Add<Config::Type<64, 128, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
    Add<Config::Type<64, 128, 64, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();
    Add<Config::Type<64, 128, 32, 1, 4, 1, Default, Stream, 4, true, 1, 128>>();
    Add<Config::Type<48, 128, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
    Add<Config::Type<48, 128, 32, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();
    Add<Config::Type<32, 128, 64, 1, 4, 1, Default, Stream, 5, true, 1, 128>>();
    Add<Config::Type<32, 128, 32, 1, 4, 1, Default, Stream, 5, true, 1, 128>>(); 
    Add<Config::Type<16, 128, 64, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
    Add<Config::Type<16, 128, 128, 1, 4, 1, Default, Stream, 3, true, 1, 128>>();
    Add<Config::Type<16, 128, 128, 1, 4, 2, Default, Stream, 5, true, 1, 128>>();

    // N = 64, streaming
    Add<Config::Type<16, 64, 64, 1, 2, 2, Default, Stream, 5, true, 1, 128>>();
    Add<Config::Type<16, 64, 128, 1, 2, 2, Default, Stream, 3, true, 1, 128>>();
    Add<Config::Type<16, 64, 128, 1, 2, 4, Default, Stream, 5, true, 1, 128>>();
}

}  // namespace turbomind::gemm
