// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/kernel/mxfp4.h"
#include "src/turbomind/kernels/gemm/kernel/u4.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/linear_weight.h"
#include <type_traits>

namespace turbomind::gemm {

using namespace sm80_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
template<int GroupSize, bool Grouped>
void pack_u4(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    if constexpr (Grouped) {
        PackWeight(linear, GetImpl<Arch<80>, kRowMajor, HMMA_16816 | OPERAND_B | 2, uint16_t, uint4_t>(), stream);
    }
    else {
        PackWeight(linear, GetImpl<Arch<80>, kColMajor, HMMA_16816 | OPERAND_B | 2, uint16_t, uint4_t>(), stream);
    }
    PackQParams(linear,
                GetImpl<Arch<80>, kColMajor, HMMA_16816 | OPERAND_V | 1, uint32_t, uint32_t>(),
                QuantDesc{QuantType::kK, GroupSize},
                stream);
    linear.weight_format = DataFormat{kUint4, {GroupSize, 1}, kHalf, kHalf};
}

constexpr auto mxfp4_packer =
    pack_mxfp4<Arch<80>, kColMajor, HMMA_16816 | OPERAND_A | 1, kColMajor, HMMA_16816 | OPERAND_U | 1>;

const Family u4_d_32{22, 140, kHalf, kHalf, 32, 8, 1, 1, true, false, supports_u4<32, kHalf>, pack_u4<32, false>};
const Family u4_g_32{23, 139, kHalf, kHalf, 32, 8, 1, 1, true, true, supports_u4<32, kHalf>, pack_u4<32, true>};
const Family u4_d_128{24, 190, kHalf, kHalf, 128, 8, 1, 1, true, false, supports_u4<128, kHalf>, pack_u4<128, false>};
const Family u4_g_128{25, 189, kHalf, kHalf, 128, 8, 1, 1, true, true, supports_u4<128, kHalf>, pack_u4<128, true>};
const Family mxfp4{26, 200, kBfloat16, kBfloat16, 32, 8, 1, 1, true, true, supports_mxfp4, mxfp4_packer};

template<int kGroupSize>
void register_u4_d(Collector& c)
{
    // clang-format off
        using C = Config_U4_d<Sm90, half, kColMajor>;
        c.add<C::Type<128, 256,  64, 1, 8, 1, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type<128, 256,  32, 1, 8, 1, D, D, 3, true, 1, kGroupSize, 128, 128>>();
        c.add<C::Type<128, 256,  32, 1, 8, 1, D, D, 4, true, 1, kGroupSize, 128, 128>>();
        c.add<C::Type<128, 128,  32, 1, 4, 1, D, D, 3, true, 1, kGroupSize, 64, 128>>();
        c.add<C::Type<128, 128,  32, 1, 4, 1, D, D, 4, true, 1, kGroupSize, 64, 128>>();
        c.add<C::Type<128, 128,  64, 1, 4, 2, D, D, 3, true, 1, kGroupSize, 64, 128>>();

        c.add<C::Type<96, 256,  32, 1, 8, 1, D, D, 4, true, 1, kGroupSize>>();
        c.add<C::Type<96, 256,  32, 1, 8, 1, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type<96, 128,  32, 1, 4, 1, D, D, 4, true, 1, kGroupSize>>();
        c.add<C::Type<96, 128, 128, 1, 4, 2, D, D, 3, true, 1, kGroupSize>>();

        c.add<C::Type<64, 256,  32, 1, 4, 1, D, D, 3, true, 1, kGroupSize, 64, 128>>();
        c.add<C::Type<64, 256,  32, 1, 4, 1, D, D, 4, true, 1, kGroupSize, 64, 128>>();
        c.add<C::Type<64, 128,  32, 1, 4, 1, D, D, 4, true, 1, kGroupSize>>();
        c.add<C::Type<64, 128,  64, 1, 4, 1, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type<64, 128, 128, 1, 4, 2, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type<64,  64,  64, 1, 2, 2, D, D, 6, true, 1, kGroupSize>>();

        c.add<C::Type<48, 256,  64, 1, 4, 1, D, D, 3, true, 1, kGroupSize, 48, 128>>();
        c.add<C::Type<48, 128,  64, 1, 4, 1, D, D, 4, true, 1, kGroupSize>>();
        c.add<C::Type<48, 128, 128, 1, 4, 2, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type<48,  64, 128, 1, 2, 2, D, D, 4, true, 1, kGroupSize>>();

        c.add<C::Type<32, 256,  64, 1, 4, 1, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type<32, 128,  64, 1, 4, 1, D, D, 4, true, 1, kGroupSize>>();
        c.add<C::Type<32, 128, 128, 1, 4, 2, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type<32,  64, 128, 1, 2, 2, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type<32,  64, 128, 1, 2, 2, D, D, 4, true, 1, kGroupSize>>();

        c.add<C::Type<16, 128,  64, 1, 4, 1, D, D, 4, true, 1, kGroupSize>>();
        c.add<C::Type<16, 128, 128, 1, 4, 2, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type<16, 128, 128, 1, 4, 2, D, D, 4, true, 1, kGroupSize>>();
        c.add<C::Type<16,  64, 128, 1, 2, 2, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type<16,  64, 128, 1, 2, 2, D, D, 4, true, 1, kGroupSize>>();
    // clang-format on
}

template<int kGroupSize>
void register_u4_g(Collector& c)
{
    // clang-format off
        using C = Config_U4_g<Sm90, half, kColMajor>;
        c.add<C::Type<128, 256,  32, 2, 4, 1, D, D, 3,   0 , 1, kGroupSize>>();
        c.add<C::Type<128, 128,  32, 1, 4, 1, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type< 64, 128,  64, 1, 4, 1, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type< 64, 256,  32, 1, 4, 1, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type< 32,  64, 128, 1, 2, 2, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type< 32, 128,  64, 1, 4, 1, D, D, 5, true, 1, kGroupSize>>();
        c.add<C::Type< 32, 256,  64, 1, 4, 1, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type< 16, 256,  64, 1, 4, 1, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type< 16, 256,  32, 1, 4, 1, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type< 16, 128,  64, 1, 4, 1, D, D, 3, true, 1, kGroupSize>>();
        c.add<C::Type< 16,  64, 128, 1, 2, 2, D, D, 3, true, 1, kGroupSize>>();
    // clang-format on
}

void register_mxfp4(Collector& c)
{
    // clang-format off
        using Cd = Config_MXF4<Sm90, bfloat16_t, 16, kColMajor>;
        // c.add<Cd::Type<256, 128, 32, 8, 1, 1, D, D, 3, true, 32, 1, 128, 128>>();

        using Cg = Config_MXF4<Sm90, bfloat16_t, 16, kColMajor, 1>;
        c.add<Cg::Type<256, 128, 32, 8, 1, 1, D, D, 3, true, 32, 1, 128, 128>>();
        c.add<Cg::Type<256,  64, 32, 4, 1, 1, D, D, 3, true, 32, 1, 128,  64>>();
        c.add<Cg::Type<256,  32, 32, 4, 1, 1, D, D, 5, true, 32, 1>>();
        c.add<Cg::Type<128, 128, 32, 4, 1, 1, D, D, 4, true, 32, 1, 128,  64>>();
        c.add<Cg::Type<128,  96, 32, 4, 1, 1, D, D, 3, true, 32, 1>>();
        c.add<Cg::Type<128,  64, 32, 4, 1, 1, D, D, 3, true, 32, 1>>();
        c.add<Cg::Type<128,  32, 32, 4, 1, 1, D, D, 3, true, 32, 1>>();
        c.add<Cg::Type<128,  16, 32, 4, 1, 1, D, D, 5, true, 32, 1>>();
        c.add<Cg::Type<128,  16, 64, 4, 1, 1, D, D, 3, true, 32, 1>>();

        using C8 = Config_MXF4<Sm90, bfloat16_t, 8, kColMajor, 1>;
        c.add<C8::Type<256, 8,  32, 4, 1, 1, D, D, 5, true, 32, 1>>();
        c.add<C8::Type<128, 8,  32, 4, 1, 1, D, D, 5, true, 32, 1>>();
        c.add<C8::Type<128, 8,  64, 4, 1, 1, D, D, 3, true, 32, 1>>();
        c.add<C8::Type< 64, 8,  64, 4, 1, 1, D, D, 5, true, 32, 1>>();
    // clang-format on
}

Registrar reg[]{
    {u4_d_32, [](Collector& c) { register_u4_d<32>(c); }},
    {u4_g_32, [](Collector& c) { register_u4_g<32>(c); }},
    {u4_d_128, [](Collector& c) { register_u4_d<128>(c); }},
    {u4_g_128, [](Collector& c) { register_u4_g<128>(c); }},
    {mxfp4, [](Collector& c) { register_mxfp4(c); }},
};
}  // namespace

}  // namespace turbomind::gemm
