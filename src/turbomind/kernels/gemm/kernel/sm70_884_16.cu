// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/kernel/floating_point.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
constexpr auto f16_packer = pack_fp<Sm70, kRowMajor, HMMA_884 | OPERAND_B | 1, kHalf>;

const Family f16{1, 190, kHalf, kHalf, 16, 8, 1, 1, true, true, supports_fp<kHalf>, f16_packer};

Registrar reg(f16, [](Collector& c) {
    if constexpr (1) {
        // clang-format off
        using C = Config_F16<kColMajor, 0>;
        c.add<C::Type<256, 128,  16, 4, 2, 1, D, D, 2,   0 , 1, 1, 128, 128>>();
        c.add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 1, 128, 128>>();
        c.add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 1, 128, 128>>();
        c.add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 1,  64, 128>>();
        c.add<C::Type< 96,  64,  32, 2, 2, 1, D, D, 2, true, 1, 1>>();
        c.add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        c.add<C::Type< 64,  64,  64, 2, 2, 1, D, S, 2, true, 1, 1>>();
        c.add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        c.add<C::Type< 16, 128,  64, 1, 4, 1, D, S, 2, true, 1, 1>>();
        c.add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        c.add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 1>>();
        // clang-format on
    }
});
}  // namespace

}  // namespace turbomind::gemm
