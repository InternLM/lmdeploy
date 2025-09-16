// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm80_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

void Registry::sm90_16816_4()
{
    if constexpr (1) {
        // clang-format off
        using C = Config_U4_d<Sm90, half, kColMajor>;
        Add<C::Type<128, 256,  64, 1, 8, 1, D, D, 3, true, 1, 128>>();
        Add<C::Type<128, 256,  32, 1, 8, 1, D, D, 3, true, 1, 128, 128, 128>>();
        Add<C::Type<128, 256,  32, 1, 8, 1, D, D, 4, true, 1, 128, 128, 128>>();
        Add<C::Type<128, 128,  32, 1, 4, 1, D, D, 3, true, 1, 128, 64, 128>>();
        Add<C::Type<128, 128,  32, 1, 4, 1, D, D, 4, true, 1, 128, 64, 128>>();
        Add<C::Type<128, 128,  64, 1, 4, 2, D, D, 3, true, 1, 128, 64, 128>>();

        Add<C::Type<96, 256,  32, 1, 8, 1, D, D, 4, true, 1, 128>>();
        Add<C::Type<96, 256,  32, 1, 8, 1, D, D, 3, true, 1, 128>>();
        Add<C::Type<96, 128,  32, 1, 4, 1, D, D, 4, true, 1, 128>>();
        Add<C::Type<96, 128, 128, 1, 4, 2, D, D, 3, true, 1, 128>>();

        Add<C::Type<64, 256,  32, 1, 4, 1, D, D, 3, true, 1, 128, 64, 128>>();
        Add<C::Type<64, 256,  32, 1, 4, 1, D, D, 4, true, 1, 128, 64, 128>>();
        Add<C::Type<64, 128,  32, 1, 4, 1, D, D, 4, true, 1, 128>>();
        Add<C::Type<64, 128,  64, 1, 4, 1, D, D, 3, true, 1, 128>>();
        Add<C::Type<64, 128, 128, 1, 4, 2, D, D, 3, true, 1, 128>>();
        Add<C::Type<64,  64,  64, 1, 2, 2, D, D, 6, true, 1, 128>>();

        Add<C::Type<48, 256,  64, 1, 4, 1, D, D, 3, true, 1, 128, 48, 128>>();
        Add<C::Type<48, 128,  64, 1, 4, 1, D, D, 4, true, 1, 128>>();
        Add<C::Type<48, 128, 128, 1, 4, 2, D, D, 3, true, 1, 128>>();
        Add<C::Type<48,  64, 128, 1, 2, 2, D, D, 4, true, 1, 128>>();

        Add<C::Type<32, 256,  64, 1, 4, 1, D, D, 3, true, 1, 128>>();
        Add<C::Type<32, 128,  64, 1, 4, 1, D, D, 4, true, 1, 128>>();
        Add<C::Type<32, 128, 128, 1, 4, 2, D, D, 3, true, 1, 128>>();
        Add<C::Type<32,  64, 128, 1, 2, 2, D, D, 3, true, 1, 128>>();
        Add<C::Type<32,  64, 128, 1, 2, 2, D, D, 4, true, 1, 128>>();

        Add<C::Type<16, 128,  64, 1, 4, 1, D, D, 4, true, 1, 128>>();
        Add<C::Type<16, 128, 128, 1, 4, 2, D, D, 3, true, 1, 128>>();
        Add<C::Type<16, 128, 128, 1, 4, 2, D, D, 4, true, 1, 128>>();
        Add<C::Type<16,  64, 128, 1, 2, 2, D, D, 3, true, 1, 128>>();
        Add<C::Type<16,  64, 128, 1, 2, 2, D, D, 4, true, 1, 128>>();
        // clang-format on
    }

    if constexpr (1) {
        // clang-format off
        using C = Config_U4_g<Sm90, half, kColMajor>;
        Add<C::Type<128, 256,  32, 2, 4, 1, D, D, 3,   0 , 1, 128>>();
        Add<C::Type<128, 128,  32, 1, 4, 1, D, D, 3, true, 1, 128>>();
        Add<C::Type< 64, 128,  64, 1, 4, 1, D, D, 3, true, 1, 128>>();
        Add<C::Type< 64, 256,  32, 1, 4, 1, D, D, 3, true, 1, 128>>();
        Add<C::Type< 32,  64, 128, 1, 2, 2, D, D, 3, true, 1, 128>>();
        Add<C::Type< 32, 128,  64, 1, 4, 1, D, D, 5, true, 1, 128>>();
        Add<C::Type< 32, 256,  64, 1, 4, 1, D, D, 3, true, 1, 128>>();
        Add<C::Type< 16, 256,  64, 1, 4, 1, D, D, 3, true, 1, 128>>();
        Add<C::Type< 16, 256,  32, 1, 4, 1, D, D, 3, true, 1, 128>>();
        Add<C::Type< 16, 128,  64, 1, 4, 1, D, D, 3, true, 1, 128>>();
        Add<C::Type< 16,  64, 128, 1, 2, 2, D, D, 3, true, 1, 128>>();
        // clang-format on
    }

    if constexpr (1) {
        // clang-format off
        using Cd = Config_MXF4<Sm90, bfloat16_t, 16, kColMajor>;
        // Add<Cd::Type<256, 128, 32, 8, 1, 1, D, D, 3, true, 32, 1, 128, 128>>();

        using Cg = Config_MXF4<Sm90, bfloat16_t, 16, kColMajor, 1>;
        Add<Cg::Type<256, 128, 32, 8, 1, 1, D, D, 3, true, 32, 1, 128, 128>>();
        Add<Cg::Type<256,  64, 32, 4, 1, 1, D, D, 3, true, 32, 1, 128,  64>>();
        Add<Cg::Type<256,  32, 32, 4, 1, 1, D, D, 5, true, 32, 1>>();
        Add<Cg::Type<128, 128, 32, 4, 1, 1, D, D, 4, true, 32, 1, 128,  64>>();
        Add<Cg::Type<128,  96, 32, 4, 1, 1, D, D, 3, true, 32, 1>>();
        Add<Cg::Type<128,  64, 32, 4, 1, 1, D, D, 3, true, 32, 1>>();
        Add<Cg::Type<128,  32, 32, 4, 1, 1, D, D, 3, true, 32, 1>>();
        Add<Cg::Type<128,  16, 32, 4, 1, 1, D, D, 5, true, 32, 1>>();
        Add<Cg::Type<128,  16, 64, 4, 1, 1, D, D, 3, true, 32, 1>>();

        using C8 = Config_MXF4<Sm90, bfloat16_t, 8, kColMajor, 1>;
        Add<C8::Type<256, 8,  32, 4, 1, 1, D, D, 5, true, 32, 1>>();
        Add<C8::Type<128, 8,  32, 4, 1, 1, D, D, 5, true, 32, 1>>();
        Add<C8::Type<128, 8,  64, 4, 1, 1, D, D, 3, true, 32, 1>>();
        Add<C8::Type< 64, 8,  64, 4, 1, 1, D, D, 5, true, 32, 1>>();
        // clang-format on
    }
}

}  // namespace turbomind::gemm
