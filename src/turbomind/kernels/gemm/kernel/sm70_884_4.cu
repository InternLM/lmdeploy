// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/kernel/mxfp4.h"
#include "src/turbomind/kernels/gemm/kernel/u4.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
template<int GroupSize>
void pack_u4(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    PackWeight(linear, GetImpl<Sm70, kRowMajor, HMMA_884 | OPERAND_B | 1, uint16_t, uint4_t>(), stream);
    PackQParams(linear,
                GetImpl<Sm70, kColMajor, HMMA_884 | OPERAND_V | 1, uint32_t, uint32_t>(),
                QuantDesc{QuantType::kK, GroupSize},
                stream);
    linear.weight_format = DataFormat{kUint4, {GroupSize, 1}, kHalf, kHalf};
}

const Family u4_g32{3, 140, kHalf, kHalf, 32, 8, 1, 1, true, true, supports_u4<32, kHalf>, pack_u4<32>};

const Family u4_g128{4, 190, kHalf, kHalf, 128, 8, 1, 1, true, true, supports_u4<128, kHalf>, pack_u4<128>};

constexpr auto mxfp4_packer =
    pack_mxfp4<Sm70, kRowMajor, HMMA_884 | OPERAND_B | 1, kColMajor, HMMA_884 | OPERAND_V | 1>;

const Family mxfp4{5, 190, kHalf, kHalf, 32, 8, 1, 1, true, true, supports_mxfp4, mxfp4_packer};

void register_g128(Collector& c)
{
    if constexpr (1) {
        // clang-format off
        using C = Config_U4_d<kColMajor>;
        c.add<C::Type<128, 256, 16, 2, 4, 1, D, D, 2, true, 1, 128, 128, 128>>();
        c.add<C::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 128, 64, 128>>();
        c.add<C::Type<128, 128, 16, 2, 2, 1, D, S, 2, true, 1, 128, 64, 128>>();
        c.add<C::Type< 96, 128, 32, 2, 2, 1, D, S, 2, true, 1, 128, 48, 128>>();
        c.add<C::Type< 64, 128, 32, 2, 2, 1, D, D, 2, true, 1, 128, 32, 128>>();
        c.add<C::Type< 64, 128, 32, 2, 2, 1, D, S, 2, true, 1, 128, 32, 128>>();
        c.add<C::Type< 64, 128, 16, 1, 4, 1, D, S, 2, true, 1, 128, 32, 128>>();
        c.add<C::Type< 64, 256, 16, 1, 4, 1, D, S, 2, true, 1, 128, 64, 128>>();
        c.add<C::Type< 32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type< 32, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128, 32, 128>>();
        c.add<C::Type< 16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type< 16, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type<  8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type<  8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type<  8, 256, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        // clang-format on
    }

    if constexpr (1) {
        // clang-format off
        using C = Config_U4_g<kColMajor>;
        c.add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 128, 128, 128>>();
        c.add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        c.add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        c.add<C::Type< 64, 256,  16, 1, 4, 1, D, S, 2, true, 1, 128,  64, 128>>();
        c.add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type< 32, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type< 16, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        // clang-format on
    }
}

void register_g32(Collector& c)
{
    if constexpr (1) {
        // clang-format off
        using C = Config_U4_d<kColMajor>;
        c.add<C::Type<128, 256, 16, 2, 4, 1, D, D, 2, true, 1, 32, 128, 128>>();
        c.add<C::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 32, 64, 128>>();
        c.add<C::Type<128, 128, 16, 2, 2, 1, D, S, 2, true, 1, 32, 64, 128>>();
        c.add<C::Type< 96, 128, 32, 2, 2, 1, D, S, 2, true, 1, 32, 48, 128>>();
        c.add<C::Type< 64, 128, 32, 2, 2, 1, D, D, 2, true, 1, 32, 32, 128>>();
        c.add<C::Type< 64, 128, 32, 2, 2, 1, D, S, 2, true, 1, 32, 32, 128>>();
        c.add<C::Type< 64, 128, 16, 1, 4, 1, D, S, 2, true, 1, 32, 32, 128>>();
        c.add<C::Type< 64, 256, 16, 1, 4, 1, D, S, 2, true, 1, 32, 64, 128>>();
        c.add<C::Type< 32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 32, 256, 32, 1, 4, 1, D, S, 2, true, 1, 32, 32, 128>>();
        c.add<C::Type< 16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 16, 256, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type<  8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type<  8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type<  8, 256, 64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 48, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 16, 256, 64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 16, 128, 64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type<  8, 256, 32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 32, 128, 64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 64, 256, 32, 1, 4, 1, D, S, 2, true, 1, 32, 64, 128>>();
        // clang-format on
    }

    if constexpr (1) {
        // clang-format off
        using C = Config_U4_g<kColMajor>;
        c.add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 32, 128, 128>>();
        c.add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 32,  64, 128>>();
        c.add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32,  32, 128>>();
        c.add<C::Type< 64, 256,  16, 1, 4, 1, D, S, 2, true, 1, 32,  64, 128>>();
        c.add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 32, 256,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 16, 256,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type<  8, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type<  8, 128, 128, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 48, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 16, 128,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type<  8, 256,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 32, 256,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 64, 256,  32, 1, 4, 1, D, S, 2, true, 1, 32,  64, 128>>();
        // clang-format on
    }
}

void register_mxfp4(Collector& c)
{
    if constexpr (1) {
        // clang-format off
        using C = Config_MXF4<kColMajor, 0>;
        c.add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 32,  64, 128>>();
        c.add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32,  32, 128>>();
        c.add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        c.add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        // clang-format on
    }
}

Registrar reg[]{
    {u4_g32, [](Collector& c) { register_g32(c); }},
    {u4_g128, [](Collector& c) { register_g128(c); }},
    {mxfp4, [](Collector& c) { register_mxfp4(c); }},
};
}  // namespace

}  // namespace turbomind::gemm
