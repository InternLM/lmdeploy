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

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, int GroupAxis = -1> class K>
void register_kernels(Collector& c)
{
    using config::Config;
    using config::Shape;

    using _256x128x16_4x2x1 = Config<Shape<256, 128, 16>, Shape<4, 2, 1>>;
    using _128x256x16_2x4x1 = Config<Shape<128, 256, 16>, Shape<2, 4, 1>>;
    using _128x128x16_2x2x1 = Config<Shape<128, 128, 16>, Shape<2, 2, 1>>;
    using _96x64x32_2x2x1 = Config<Shape<96, 64, 32>, Shape<2, 2, 1>>;
    using _64x128x32_1x4x1 = Config<Shape<64, 128, 32>, Shape<1, 4, 1>>;
    using _64x64x64_2x2x1 = Config<Shape<64, 64, 64>, Shape<2, 2, 1>>;
    using _32x128x32_1x4x1 = Config<Shape<32, 128, 32>, Shape<1, 4, 1>>;
    using _16x128x64_1x4x1 = Config<Shape<16, 128, 64>, Shape<1, 4, 1>>;
    using _16x128x32_1x4x1 = Config<Shape<16, 128, 32>, Shape<1, 4, 1>>;
    using _8x128x64_1x4x1 = Config<Shape<8, 128, 64>, Shape<1, 4, 1>>;

    {
        add<K<_256x128x16_4x2x1, 2, kColMajor, D, D, false, 128, 128, 0>>(c);
        add<K<_128x256x16_2x4x1, 2, kColMajor, D, D, false, 128, 128, 0>>(c);
        add<K<_128x256x16_2x4x1, 2, kColMajor, D, D, false, 128, 128, 0>>(c);
        add<K<_128x128x16_2x2x1, 2, kColMajor, D, D, true, 64, 128, 0>>(c);
        add<K<_96x64x32_2x2x1, 2, kColMajor, D, D, true, -1, -1, 0>>(c);
        add<K<_64x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_64x64x64_2x2x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_32x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x128x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_8x128x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
    }
}

using F16 = Config_F16;

Registrar reg(f16, register_kernels<F16::Type>);
}  // namespace

}  // namespace turbomind::gemm
