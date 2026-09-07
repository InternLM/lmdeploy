// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/config_sm75_s16816.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/kernel/geometry.h"
#include "src/turbomind/kernels/gemm/kernel/floating_point.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

using namespace sm75_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
using namespace config::geometry;

constexpr auto f16_packer = pack_fp<Sm75, kRowMajor, HMMA_16816 | OPERAND_B | 1, kHalf>;

const Family f16{6, 190, kHalf, kHalf, 32, 8, 1, 1, true, true, supports_fp<kHalf>, f16_packer};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, int GroupAxis = -1> class K>
void register_kernels(Collector& c)
{
    {
        add<K<_128x256x32_2x4x1, 2, kColMajor, D, D, false, 128, 128, 0>>(c);
        add<K<_128x128x32_2x2x1, 2, kColMajor, D, D, true, 64, 128, 0>>(c);
        add<K<_96x64x64_2x2x1, 2, kColMajor, D, D, true, -1, -1, 0>>(c);
        add<K<_64x128x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_64x64x64_2x2x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_64x64x128_1x2x2, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_32x64x128_1x2x2, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_32x128x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x64x128_1x2x2, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x128x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
    }
}

using F16 = Config_F16;

Registrar reg(f16, register_kernels<F16::Type>);
}  // namespace

}  // namespace turbomind::gemm
