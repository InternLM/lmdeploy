// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

void Registry::sm70_884_16()
{
    if constexpr (1) {
        // clang-format off
        using C = Config_F16<kColMajor, 0>;
        Add<C::Type<256, 128,  16, 4, 2, 1, D, D, 2,   0 , 1, 1, 128, 128>>();
        Add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 1, 128, 128>>();
        Add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 1, 128, 128>>();
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 1,  64, 128>>();
        Add<C::Type< 96,  64,  32, 2, 2, 1, D, D, 2, true, 1, 1>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type< 64,  64,  64, 2, 2, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type< 16, 128,  64, 1, 4, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 1>>();
        // clang-format on
    }
}

}  // namespace turbomind::gemm
