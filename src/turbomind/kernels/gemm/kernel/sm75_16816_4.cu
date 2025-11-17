// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm75_s16816.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm75_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

void Registry::sm75_16816_4()
{
    if constexpr (1) {
        // clang-format off
        using C = Config_U4_d<kColMajor>;
        Add<C::Type<128, 256, 32, 1, 8, 1, D, D, 2, true, 1, 128, 128, 128>>();
        Add<C::Type<128, 128, 32, 1, 4, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 96,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        Add<C::Type< 64, 128, 32, 1, 4, 1, D, D, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 64,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        Add<C::Type< 48, 128, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 48,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        Add<C::Type< 32,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        // clang-format on
    }

    if constexpr (1) {
        // clang-format off
        using C = Config_U4_g<kColMajor>;
        Add<C::Type<128, 256,  32, 2, 4, 1, D, D, 2,    0, 1, 128, 128, 128>>();
        Add<C::Type<128, 128,  32, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 64, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 64, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 256>>();
        Add<C::Type< 32,  64, 128, 1, 2, 2, D, S, 2, true, 1, 128>>();
        Add<C::Type< 32, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16,  64,  64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        // clang-format on
    }

    if constexpr (1) {
        // clang-format off
        using C = Config_MXF4<kColMajor, 1>;
        Add<C::Type<128, 128, 32, 4, 1, 1, D, D, 2, true, 32, 1, 128, 64>>();
        Add<C::Type<128,  64, 32, 4, 1, 1, D, D, 2, true, 32, 1>>();
        Add<C::Type<128,  32, 32, 4, 1, 1, S, D, 2, true, 32, 1>>();
        Add<C::Type<128,  16, 32, 4, 1, 1, S, D, 2, true, 32, 1>>();
        Add<C::Type<128,  16, 64, 4, 1, 1, S, D, 2, true, 32, 1>>();
        Add<C::Type< 64,  16, 64, 4, 1, 1, S, D, 2, true, 32, 1>>();
        // clang-format on
    }
}

}  // namespace turbomind::gemm
