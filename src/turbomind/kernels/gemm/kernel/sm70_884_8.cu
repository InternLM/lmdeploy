// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/kernel/geometry.h"
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
using namespace config::geometry;

constexpr auto e4m3_packer =
    pack_e4m3<Sm70, kRowMajor, HMMA_884 | OPERAND_B | 1, kColMajor, HMMA_884 | OPERAND_V | 1, kHalf>;

const Family e4m3{2, 190, kHalf, kHalf, 128, 8, 1, 1, true, true, supports_e4m3<kHalf, 1>, e4m3_packer};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, int GroupAxis = -1> class K>
void register_kernels(Collector& c)
{
    {
        add<K<_128x128x16_2x2x1, 2, kColMajor, D, D, true, 64, 128, 0>>(c);
        add<K<_64x128x32_1x4x1, 2, kColMajor, D, S, true, 32, 128, 0>>(c);
        add<K<_32x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_8x128x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
    }
}

using E4M3 = Config_E4M3;

Registrar reg(e4m3, register_kernels<E4M3::Type>);
}  // namespace

}  // namespace turbomind::gemm
