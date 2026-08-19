// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/config_sm75_s16816.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm75_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
Registrar reg([](Collector& c, int /*arch*/) {
    if constexpr (1) {
        // clang-format off
        using Cg = Config_E4M3<kColMajor, 1>;
        c.add<Cg::Type<256, 128,  32, 8, 1, 1, D, D, 3, true, 128, 1, 128, 128>>();
        c.add<Cg::Type<256,  64,  32, 4, 1, 1, D, D, 3, true, 128, 1, 128,  64>>();
        c.add<Cg::Type<128, 128,  32, 4, 1, 1, D, D, 3, true, 128, 1, 128,  64>>();
        c.add<Cg::Type<128,  96,  32, 4, 1, 1, D, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128,  64,  32, 4, 1, 1, D, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128,  32,  32, 4, 1, 1, S, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128,  16,  64, 4, 1, 1, S, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128,  16,  32, 4, 1, 1, S, D, 5, true, 128, 1>>();
        // clang-format on
    }

    if constexpr (1) {
        // FP8 weights (e4m3, w8a16): B packed 1 byte/element (2x u4 footprint),
        // so 128x128 is the largest CTA that fits smem. V carries one f16 scale
        // per (N, K/128).
        // clang-format off
        using Cf = Config_Fp8W<kColMajor>;
        c.add<Cf::Type<128, 128, 32, 1, 4, 1, D, D, 2, true, 1, 128,  64, 128>>();
        c.add<Cf::Type< 64, 128, 32, 1, 4, 1, D, D, 2, true, 1, 128,  32, 128>>();
        c.add<Cf::Type< 64,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        c.add<Cf::Type< 48,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        c.add<Cf::Type< 32,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        c.add<Cf::Type< 16,  64, 64, 1, 2, 2, D, S, 2, true, 1, 128>>();
        // clang-format on
    }
});
}

}  // namespace turbomind::gemm
