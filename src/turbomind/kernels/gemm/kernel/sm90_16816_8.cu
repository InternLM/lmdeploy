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
        using Cd = Config_E4M3<Sm90, bfloat16_t, 16, kColMajor>;
        // c.add<Cd::Type<256, 128, 32, 8, 1, 1, D, D, 3, true, 128, 1, 128, 128>>();

        using Cg = Config_E4M3<Sm90, bfloat16_t, 16, kColMajor, 1>;
        c.add<Cg::Type<256, 128,  32, 8, 1, 1, D, D, 3, true, 128, 1, 128, 128>>();
        c.add<Cg::Type<256,  64,  32, 4, 1, 1, D, D, 3, true, 128, 1, 128,  64>>();
        c.add<Cg::Type<256,  32,  64, 4, 1, 1, D, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128, 128,  32, 4, 1, 1, D, D, 3, true, 128, 1, 128,  64>>();
        c.add<Cg::Type<128,  96,  32, 4, 1, 1, D, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128,  64,  32, 4, 1, 1, D, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128,  32,  32, 4, 1, 1, D, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128,  16,  64, 4, 1, 1, D, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128,  16,  32, 4, 1, 1, D, D, 5, true, 128, 1>>();

        using C8 = Config_E4M3<Sm90, bfloat16_t, 8, kColMajor, 1>;
        c.add<C8::Type<256, 8,  64, 4, 1, 1, D, D, 3, true, 128, 1>>();
        c.add<C8::Type<128, 8,  64, 4, 1, 1, D, D, 3, true, 128, 1>>();
        c.add<C8::Type< 64, 8, 128, 4, 1, 1, D, D, 3, true, 128, 1>>();
        // clang-format on
    }
});
}

}  // namespace turbomind::gemm
