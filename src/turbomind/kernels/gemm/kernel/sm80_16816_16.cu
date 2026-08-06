// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm80_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
Registrar reg([](Collector& c, int /*arch*/) {
    if constexpr (1) {
        // clang-format off
        using C = Config_F16_g<Sm80, half, kColMajor>;
        c.add<C::Type<256, 128,  64, 4, 2, 1, D, D, 3,   0 , 1, 1>>();
        c.add<C::Type<128, 256,  64, 2, 4, 1, D, D, 3,   0 , 1, 1>>(); // 10
        c.add<C::Type<128, 256,  32, 2, 4, 1, D, D, 3,   0 , 1, 1>>();
        c.add<C::Type<128, 128,  32, 2, 2, 1, D, D, 3, true, 1, 1>>(); // 6
        c.add<C::Type<128, 128,  64, 2, 2, 1, D, D, 3, true, 1, 1>>();
        c.add<C::Type<128, 128,  32, 2, 2, 1, D, D, 5, true, 1, 1>>();
        c.add<C::Type< 96,  64,  64, 2, 2, 1, D, D, 3, true, 1, 1>>(); // 2
        c.add<C::Type< 64, 128,  64, 1, 4, 1, D, S, 3, true, 1, 1>>();
        c.add<C::Type< 64,  64,  64, 2, 2, 1, D, S, 3, true, 1, 1>>(); // *
        c.add<C::Type< 64,  64,  64, 2, 2, 1, D, S, 5, true, 1, 1>>();
        c.add<C::Type< 64,  64, 128, 1, 2, 2, D, S, 3, true, 1, 1>>(); // 4
        c.add<C::Type< 32,  64, 128, 1, 2, 2, D, S, 3, true, 1, 1>>();
        c.add<C::Type< 32, 128,  64, 1, 4, 1, D, S, 3, true, 1, 1>>();
        c.add<C::Type< 16,  64, 128, 1, 2, 2, D, S, 3, true, 1, 1>>(); // 10
        c.add<C::Type< 16, 128,  64, 1, 4, 1, D, S, 3, true, 1, 1>>();
        // clang-format on
    }

    if constexpr (1) {
        // clang-format off
        using C = Config_F16_g<Sm80, nv_bfloat16, kColMajor>;
        c.add<C::Type<256, 128,  64, 4, 2, 1, D, D, 3,   0 , 1, 1>>();
        c.add<C::Type<128, 256,  64, 2, 4, 1, D, D, 3,   0 , 1, 1>>(); // 10
        c.add<C::Type<128, 256,  32, 2, 4, 1, D, D, 3,   0 , 1, 1>>();
        c.add<C::Type<128, 128,  32, 2, 2, 1, D, D, 3, true, 1, 1>>(); // 6
        c.add<C::Type<128, 128,  64, 2, 2, 1, D, D, 3, true, 1, 1>>();
        c.add<C::Type<128, 128,  32, 2, 2, 1, D, D, 5, true, 1, 1>>();
        c.add<C::Type< 96,  64,  64, 2, 2, 1, D, D, 3, true, 1, 1>>(); // 2
        c.add<C::Type< 64, 128,  64, 1, 4, 1, D, S, 3, true, 1, 1>>();
        c.add<C::Type< 64,  64,  64, 2, 2, 1, D, S, 3, true, 1, 1>>(); // *
        c.add<C::Type< 64,  64,  64, 2, 2, 1, D, S, 5, true, 1, 1>>();
        c.add<C::Type< 64,  64, 128, 1, 2, 2, D, S, 3, true, 1, 1>>(); // 4
        c.add<C::Type< 32,  64, 128, 1, 2, 2, D, S, 3, true, 1, 1>>();
        c.add<C::Type< 32, 128,  64, 1, 4, 1, D, S, 3, true, 1, 1>>();
        c.add<C::Type< 16,  64, 128, 1, 2, 2, D, S, 3, true, 1, 1>>(); // 10
        c.add<C::Type< 16, 128,  64, 1, 4, 1, D, S, 3, true, 1, 1>>();
        // clang-format on
    }
});
}

}  // namespace turbomind::gemm
