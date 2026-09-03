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

const Family u4_d_32{16, 140, kHalf, kHalf, 32, 8, 1, 1, true, false, supports_u4<32, kHalf>, pack_u4<32, false>};
const Family u4_g_32{17, 139, kHalf, kHalf, 32, 8, 1, 1, true, true, supports_u4<32, kHalf>, pack_u4<32, true>};
const Family u4_d_128{18, 190, kHalf, kHalf, 128, 8, 1, 1, true, false, supports_u4<128, kHalf>, pack_u4<128, false>};
const Family u4_g_128{19, 189, kHalf, kHalf, 128, 8, 1, 1, true, true, supports_u4<128, kHalf>, pack_u4<128, true>};
const Family mxfp4{20, 200, kBfloat16, kBfloat16, 32, 8, 1, 1, true, true, supports_mxfp4, mxfp4_packer};

template<int kGroupSize>
void register_u4_d(Collector& c)
{
    // clang-format off
        using C = Config_U4_d<Sm80, half, kColMajor>;
        // c.add<C::Type<128, 256,  64, 1, 8, 1, D, S, 3, true, 1, kGroupSize>>(); // 0/0
        c.add<C::Type<128, 256,  32, 1, 8, 1, D, D, 3, true, 1, kGroupSize, 128, 128>>(); // 30/3
        c.add<C::Type<128, 256,  32, 1, 8, 1, D, D, 4, true, 1, kGroupSize, 128, 128>>(); // --/20
        c.add<C::Type<128, 128,  32, 1, 4, 1, D, D, 3, true, 1, kGroupSize, 64, 128>>();  // --/13
        c.add<C::Type<128, 128,  32, 1, 4, 1, D, S, 4, true, 1, kGroupSize, 64, 128>>();  // 21/13
        c.add<C::Type<128, 128,  64, 1, 4, 2, D, S, 3, true, 1, kGroupSize, 64, 128>>();  // 6/6

        c.add<C::Type<96, 256,  32, 1, 8, 1, D, D, 4, true, 1, kGroupSize>>();  // --/3
        c.add<C::Type<96, 256,  32, 1, 8, 1, D, S, 3, true, 1, kGroupSize>>();  // 13/13
        c.add<C::Type<96, 128,  32, 1, 4, 1, D, S, 4, true, 1, kGroupSize>>();  // 14/10
        c.add<C::Type<96, 128, 128, 1, 4, 2, D, S, 3, true, 1, kGroupSize>>();  // 2/2

        c.add<C::Type<64, 256,  32, 1, 4, 1, D, D, 3, true, 1, kGroupSize, 64, 128>>(); // --/21
        c.add<C::Type<64, 256,  32, 1, 4, 1, D, S, 4, true, 1, kGroupSize, 64, 128>>(); // 27/13
        c.add<C::Type<64, 128,  32, 1, 4, 1, D, S, 4, true, 1, kGroupSize>>();  // 8/5
        c.add<C::Type<64, 128,  64, 1, 4, 1, D, S, 3, true, 1, kGroupSize>>();  // 7/5
        c.add<C::Type<64, 128, 128, 1, 4, 2, D, S, 3, true, 1, kGroupSize>>();  // 6/7
        c.add<C::Type<64,  64,  64, 1, 2, 2, D, S, 6, true, 1, kGroupSize>>();

        c.add<C::Type<48, 256,  64, 1, 4, 1, D, S, 3, true, 1, kGroupSize, 48, 128>>(); // 1/1
        c.add<C::Type<48, 128,  64, 1, 4, 1, D, S, 4, true, 1, kGroupSize>>();  // 1/1
        c.add<C::Type<48, 128, 128, 1, 4, 2, D, S, 3, true, 1, kGroupSize>>();  // 4/4
        c.add<C::Type<48,  64, 128, 1, 2, 2, D, S, 4, true, 1, kGroupSize>>();

        c.add<C::Type<32, 256,  64, 1, 4, 1, D, S, 3, true, 1, kGroupSize>>();
        c.add<C::Type<32, 128,  64, 1, 4, 1, D, S, 4, true, 1, kGroupSize>>();
        c.add<C::Type<32, 128, 128, 1, 4, 2, D, S, 3, true, 1, kGroupSize>>();
        c.add<C::Type<32,  64, 128, 1, 2, 2, D, S, 3, true, 1, kGroupSize>>();
        c.add<C::Type<32,  64, 128, 1, 2, 2, D, S, 4, true, 1, kGroupSize>>();

        c.add<C::Type<16, 128,  64, 1, 4, 1, D, S, 4, true, 1, kGroupSize>>();
        c.add<C::Type<16, 128, 128, 1, 4, 2, D, S, 3, true, 1, kGroupSize>>();
        c.add<C::Type<16, 128, 128, 1, 4, 2, D, S, 4, true, 1, kGroupSize>>();
        c.add<C::Type<16,  64, 128, 1, 2, 2, D, S, 3, true, 1, kGroupSize>>();
        c.add<C::Type<16,  64, 128, 1, 2, 2, D, S, 4, true, 1, kGroupSize>>();
    // clang-format on
}

template<int kGroupSize>
void register_u4_g(Collector& c)
{
    // clang-format off
        using C = Config_U4_g<Sm80, half, kColMajor>;
        c.add<C::Type<128, 256,  32, 2, 4, 1, D, D, 3,   0 , 1, kGroupSize>>();  // 10 + 5 + 4 + 10 + 10, 37
        c.add<C::Type<128, 128,  32, 1, 4, 1, D, D, 3, true, 1, kGroupSize>>();  // 1 + 6 + 4 + 4 + 2, 3
        c.add<C::Type< 64, 128,  64, 1, 4, 1, D, S, 3, true, 1, kGroupSize>>();  // 7 + 4 + 6 + 2 + 4, 26
        c.add<C::Type< 64, 256,  32, 1, 4, 1, D, S, 3, true, 1, kGroupSize>>();  // 18
        c.add<C::Type< 32,  64, 128, 1, 2, 2, D, S, 3, true, 1, kGroupSize>>();  // 2
        c.add<C::Type< 32, 128,  64, 1, 4, 1, D, S, 5, true, 1, kGroupSize>>();  // 1 + 2 + 2 + 2 + 2, 2
        c.add<C::Type< 32, 256,  64, 1, 4, 1, D, S, 3, true, 1, kGroupSize>>();  // 9
        c.add<C::Type< 16, 256,  64, 1, 4, 1, D, S, 3, true, 1, kGroupSize>>();  // 22
        c.add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 3, true, 1, kGroupSize>>();  // 8
        c.add<C::Type< 16, 128,  64, 1, 4, 1, D, S, 3, true, 1, kGroupSize>>();  // 1 + 13 + 9 + 13 + 7, 7
        c.add<C::Type< 16,  64, 128, 1, 2, 2, D, S, 3, true, 1, kGroupSize>>();  // 12 + 2 + 6 + 2 + 8, 42
    // clang-format on
}

void register_mxfp4(Collector& c)
{
    // clang-format off
        using Cd = Config_MXF4<Sm80, bfloat16_t, 16, kColMajor>;
        // c.add<Cd::Type<256, 128, 32, 8, 1, 1, D, D, 3, true, 32, 1, 128, 128>>();

        using Cg = Config_MXF4<Sm80, bfloat16_t, 16, kColMajor, 1>;
        c.add<Cg::Type<256, 128, 32, 8, 1, 1, D, D, 3, true, 32, 1, 128, 128>>();
        c.add<Cg::Type<256,  64, 32, 4, 1, 1, D, D, 3, true, 32, 1, 128,  64>>();
        c.add<Cg::Type<256,  32, 32, 4, 1, 1, S, D, 5, true, 32, 1>>();
        c.add<Cg::Type<128, 128, 32, 4, 1, 1, D, D, 3, true, 32, 1, 128,  64>>();
        c.add<Cg::Type<128,  96, 32, 4, 1, 1, D, D, 3, true, 32, 1>>();
        c.add<Cg::Type<128,  64, 32, 4, 1, 1, S, D, 3, true, 32, 1>>();
        c.add<Cg::Type<128,  32, 32, 4, 1, 1, S, D, 3, true, 32, 1>>();
        c.add<Cg::Type<128,  16, 32, 4, 1, 1, S, D, 5, true, 32, 1>>();
        c.add<Cg::Type<128,  16, 64, 4, 1, 1, S, D, 3, true, 32, 1>>();

        using C8 = Config_MXF4<Sm80, bfloat16_t, 8, kColMajor, 1>;
        c.add<C8::Type<256, 8,  32, 4, 1, 1, S, D, 5, true, 32, 1>>();
        c.add<C8::Type<128, 8,  32, 4, 1, 1, S, D, 5, true, 32, 1>>();
        c.add<C8::Type<128, 8,  64, 4, 1, 1, S, D, 3, true, 32, 1>>();
        c.add<C8::Type< 64, 8,  64, 4, 1, 1, S, D, 5, true, 32, 1>>();
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
