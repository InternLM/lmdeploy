// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/kernel/e4m3.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

using namespace sm80_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
constexpr auto e4m3_packer =
    pack_e4m3<Arch<80>, kColMajor, HMMA_16816 | OPERAND_A | 1, kColMajor, HMMA_16816 | OPERAND_U | 1, kBfloat16>;

const Family e4m3{
    15, 200, kBfloat16, kBfloat16, 128, 8, 1, 1, true, true, supports_e4m3<kBfloat16, 1>, e4m3_packer};

Registrar reg(e4m3, [](Collector& c) {
    if constexpr (1) {
        // clang-format off
        using Cd = Config_E4M3<Sm80, bfloat16_t, 16, kColMajor>;
        // c.add<Cd::Type<256, 128, 32, 8, 1, 1, D, D, 3, true, 128, 1, 128, 128>>();

        using Cg = Config_E4M3<Sm80, bfloat16_t, 16, kColMajor, 1>;
        c.add<Cg::Type<256, 128,  32, 8, 1, 1, D, D, 3, true, 128, 1, 128, 128>>();
        c.add<Cg::Type<256,  64,  32, 4, 1, 1, D, D, 3, true, 128, 1, 128,  64>>();
        c.add<Cg::Type<256,  32,  64, 4, 1, 1, D, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128, 128,  32, 4, 1, 1, D, D, 3, true, 128, 1, 128,  64>>();
        c.add<Cg::Type<128,  96,  32, 4, 1, 1, D, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128,  64,  32, 4, 1, 1, D, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128,  32,  32, 4, 1, 1, S, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128,  16,  64, 4, 1, 1, S, D, 3, true, 128, 1>>();
        c.add<Cg::Type<128,  16,  32, 4, 1, 1, S, D, 5, true, 128, 1>>();

        using C8 = Config_E4M3<Sm80, bfloat16_t, 8, kColMajor, 1>;
        c.add<C8::Type<256, 8,  64, 4, 1, 1, S, D, 3, true, 128, 1>>();
        c.add<C8::Type<128, 8,  64, 4, 1, 1, S, D, 3, true, 128, 1>>();
        c.add<C8::Type< 64, 8, 128, 4, 1, 1, S, D, 3, true, 128, 1>>();
        // clang-format on
    }
});
}  // namespace

}  // namespace turbomind::gemm
