// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/kernel/e4m3.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
constexpr auto e4m3_packer =
    pack_e4m3<Sm70, kRowMajor, HMMA_884 | OPERAND_B | 1, kColMajor, HMMA_884 | OPERAND_V | 1, kHalf>;

const Family e4m3{2, 190, kHalf, kHalf, 128, 8, 1, 1, true, true, supports_e4m3<kHalf, 1>, e4m3_packer};

Registrar reg(e4m3, [](Collector& c) {
    if constexpr (1) {
        // clang-format off
        using C = Config_E4M3<kColMajor, 0>;
        c.add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        c.add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        c.add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        // clang-format on
    }
});
}  // namespace

}  // namespace turbomind::gemm
