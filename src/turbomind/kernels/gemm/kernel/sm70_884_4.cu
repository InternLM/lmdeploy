// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/kernel/geometry.h"
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
using namespace config::geometry;

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

// NVCC requires defaults on the template-template parameter.
template<template<class Config_,
                  int   Stages,
                  Order Raster,
                  class PolicyA,
                  class PolicyB,
                  bool SplitK,
                  int  EpiM      = -1,
                  int  EpiN      = -1,
                  int  GroupAxis = -1>
         class K>
void register_g32(Collector& c)
{
    {
        add<K<_128x256x16_2x4x1, 2, kColMajor, D, D, true, 128, 128>>(c);
        add<K<_128x128x16_2x2x1, 2, kColMajor, D, D, true, 64, 128>>(c);
        add<K<_128x128x16_2x2x1, 2, kColMajor, D, S, true, 64, 128>>(c);
        add<K<_96x128x32_2x2x1, 2, kColMajor, D, S, true, 48, 128>>(c);
        add<K<_64x128x32_2x2x1, 2, kColMajor, D, D, true, 32, 128>>(c);
        add<K<_64x128x32_2x2x1, 2, kColMajor, D, S, true, 32, 128>>(c);
        add<K<_64x128x16_1x4x1, 2, kColMajor, D, S, true, 32, 128>>(c);
        add<K<_64x256x16_1x4x1, 2, kColMajor, D, S, true, 64, 128>>(c);
        add<K<_32x128x32_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_32x256x32_1x4x1, 2, kColMajor, D, S, true, 32, 128>>(c);
        add<K<_16x128x32_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_16x256x32_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_8x128x64_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_8x128x32_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_8x256x64_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_48x128x32_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_16x256x64_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_16x128x64_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_8x256x32_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_32x128x64_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_64x256x32_1x4x1, 2, kColMajor, D, S, true, 64, 128>>(c);
    }

    {
        add<K<_128x256x16_2x4x1, 2, kColMajor, D, D, false, 128, 128, 0>>(c);
        add<K<_128x128x16_2x2x1, 2, kColMajor, D, D, true, 64, 128, 0>>(c);
        add<K<_64x128x32_1x4x1, 2, kColMajor, D, S, true, 32, 128, 0>>(c);
        add<K<_64x256x16_1x4x1, 2, kColMajor, D, S, true, 64, 128, 0>>(c);
        add<K<_32x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_32x256x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x256x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x256x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_8x128x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_8x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_8x128x128_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_48x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x128x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_8x256x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_8x256x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_32x256x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_64x256x32_1x4x1, 2, kColMajor, D, S, true, 64, 128, 0>>(c);
    }
}

// NVCC requires defaults on the template-template parameter.
template<template<class Config_,
                  int   Stages,
                  Order Raster,
                  class PolicyA,
                  class PolicyB,
                  bool SplitK,
                  int  EpiM      = -1,
                  int  EpiN      = -1,
                  int  GroupAxis = -1>
         class K>
void register_g128(Collector& c)
{
    {
        add<K<_128x256x16_2x4x1, 2, kColMajor, D, D, true, 128, 128>>(c);
        add<K<_128x128x16_2x2x1, 2, kColMajor, D, D, true, 64, 128>>(c);
        add<K<_128x128x16_2x2x1, 2, kColMajor, D, S, true, 64, 128>>(c);
        add<K<_96x128x32_2x2x1, 2, kColMajor, D, S, true, 48, 128>>(c);
        add<K<_64x128x32_2x2x1, 2, kColMajor, D, D, true, 32, 128>>(c);
        add<K<_64x128x32_2x2x1, 2, kColMajor, D, S, true, 32, 128>>(c);
        add<K<_64x128x16_1x4x1, 2, kColMajor, D, S, true, 32, 128>>(c);
        add<K<_64x256x16_1x4x1, 2, kColMajor, D, S, true, 64, 128>>(c);
        add<K<_32x128x32_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_32x256x32_1x4x1, 2, kColMajor, D, S, true, 32, 128>>(c);
        add<K<_16x128x32_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_16x256x32_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_8x128x64_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_8x128x32_1x4x1, 2, kColMajor, D, S, true>>(c);
        add<K<_8x256x64_1x4x1, 2, kColMajor, D, S, true>>(c);
    }

    {
        add<K<_128x256x16_2x4x1, 2, kColMajor, D, D, false, 128, 128, 0>>(c);
        add<K<_128x128x16_2x2x1, 2, kColMajor, D, D, true, 64, 128, 0>>(c);
        add<K<_64x128x32_1x4x1, 2, kColMajor, D, S, true, 32, 128, 0>>(c);
        add<K<_64x256x16_1x4x1, 2, kColMajor, D, S, true, 64, 128, 0>>(c);
        add<K<_32x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_32x256x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x256x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x256x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_8x128x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
    }
}

// NVCC requires defaults on the template-template parameter.
template<template<class Config_,
                  int   Stages,
                  Order Raster,
                  class PolicyA,
                  class PolicyB,
                  bool SplitK,
                  int  EpiM      = -1,
                  int  EpiN      = -1,
                  int  GroupAxis = -1>
         class K>
void register_mxfp4(Collector& c)
{
    {
        add<K<_128x128x16_2x2x1, 2, kColMajor, D, D, true, 64, 128, 0>>(c);
        add<K<_64x128x32_1x4x1, 2, kColMajor, D, S, true, 32, 128, 0>>(c);
        add<K<_32x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_16x128x32_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
        add<K<_8x128x64_1x4x1, 2, kColMajor, D, S, true, -1, -1, 0>>(c);
    }
}

using U4_G32  = Config_U4_d<32>;
using U4_G128 = Config_U4_d<128>;
using MXFP4   = Config_MXF4;

Registrar reg[]{
    {u4_g32, register_g32<U4_G32::Type>},
    {u4_g128, register_g128<U4_G128::Type>},
    {mxfp4, register_mxfp4<MXFP4::Type>},
};
}  // namespace

}  // namespace turbomind::gemm
