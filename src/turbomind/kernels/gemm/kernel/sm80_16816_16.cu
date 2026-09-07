// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/kernel/floating_point.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

using namespace sm80_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
constexpr auto f16_packer  = pack_fp<Arch<80>, kRowMajor, HMMA_16816 | OPERAND_B | 1, kHalf>;
constexpr auto bf16_packer = pack_fp<Arch<80>, kRowMajor, HMMA_16816 | OPERAND_B | 1, kBfloat16>;

const Family f16{13, 190, kHalf, kHalf, 32, 8, 1, 1, true, true, supports_fp<kHalf>, f16_packer};
const Family bf16{14, 200, kBfloat16, kBfloat16, 32, 8, 1, 1, true, true, supports_fp<kBfloat16>, bf16_packer};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, bool FusePrefetch = true> class K>
void register_kernels(Collector& c)
{
    using config::Config;
    using config::Shape;

    using _256x128x64_4x2x1 = Config<Shape<256, 128, 64>, Shape<4, 2, 1>>;
    using _128x256x64_2x4x1 = Config<Shape<128, 256, 64>, Shape<2, 4, 1>>;
    using _128x256x32_2x4x1 = Config<Shape<128, 256, 32>, Shape<2, 4, 1>>;
    using _128x128x32_2x2x1 = Config<Shape<128, 128, 32>, Shape<2, 2, 1>>;
    using _128x128x64_2x2x1 = Config<Shape<128, 128, 64>, Shape<2, 2, 1>>;
    using _96x64x64_2x2x1 = Config<Shape<96, 64, 64>, Shape<2, 2, 1>>;
    using _64x128x64_1x4x1 = Config<Shape<64, 128, 64>, Shape<1, 4, 1>>;
    using _64x64x64_2x2x1 = Config<Shape<64, 64, 64>, Shape<2, 2, 1>>;
    using _64x64x128_1x2x2 = Config<Shape<64, 64, 128>, Shape<1, 2, 2>>;
    using _32x64x128_1x2x2 = Config<Shape<32, 64, 128>, Shape<1, 2, 2>>;
    using _32x128x64_1x4x1 = Config<Shape<32, 128, 64>, Shape<1, 4, 1>>;
    using _16x64x128_1x2x2 = Config<Shape<16, 64, 128>, Shape<1, 2, 2>>;
    using _16x128x64_1x4x1 = Config<Shape<16, 128, 64>, Shape<1, 4, 1>>;

     {
    add<K<_256x128x64_4x2x1, 3, kColMajor, D, D, false>>(c);
    add<K<_128x256x64_2x4x1, 3, kColMajor, D, D, false>>(c); // 10
    add<K<_128x256x32_2x4x1, 3, kColMajor, D, D, false>>(c);
    add<K<_128x128x32_2x2x1, 3, kColMajor, D, D, true>>(c); // 6
    add<K<_128x128x64_2x2x1, 3, kColMajor, D, D, true>>(c);
    add<K<_128x128x32_2x2x1, 5, kColMajor, D, D, true>>(c);
    add<K<_96x64x64_2x2x1, 3, kColMajor, D, D, true>>(c); // 2
    add<K<_64x128x64_1x4x1, 3, kColMajor, D, S, true>>(c);
    add<K<_64x64x64_2x2x1, 3, kColMajor, D, S, true>>(c); // *
    add<K<_64x64x64_2x2x1, 5, kColMajor, D, S, true>>(c);
    add<K<_64x64x128_1x2x2, 3, kColMajor, D, S, true>>(c); // 4
    add<K<_32x64x128_1x2x2, 3, kColMajor, D, S, true>>(c);
    add<K<_32x128x64_1x4x1, 3, kColMajor, D, S, true>>(c);
    add<K<_16x64x128_1x2x2, 3, kColMajor, D, S, true>>(c); // 10
    add<K<_16x128x64_1x4x1, 3, kColMajor, D, S, true>>(c);
     }

}

using F16 = Config_F16_g<Sm80, half>;
using BF16 = Config_F16_g<Sm80, nv_bfloat16>;

Registrar reg[]{
    {f16, register_kernels<F16::Type>},
    {bf16, register_kernels<BF16::Type>},
};
}  // namespace

}  // namespace turbomind::gemm
