// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm75_s16816.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/kernel/geometry.h"
#include "src/turbomind/kernels/gemm/kernel/mxfp4.h"
#include "src/turbomind/kernels/gemm/kernel/u4.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/linear_weight.h"
#include <type_traits>

namespace turbomind::gemm {

using namespace sm75_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
using namespace config::geometry;

template<int GroupSize, bool Grouped>
void pack_u4(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    if constexpr (Grouped) {
        PackWeight(linear, GetImpl<Sm75, kRowMajor, HMMA_16816 | OPERAND_B | 2, uint16_t, uint4_t>(), stream);
    }
    else {
        PackWeight(linear, GetImpl<Sm75, kColMajor, HMMA_16816 | OPERAND_B | 2, uint16_t, uint4_t>(), stream);
    }
    PackQParams(linear,
                GetImpl<Sm75, kColMajor, HMMA_16816 | OPERAND_V | 1, uint32_t, uint32_t>(),
                QuantDesc{QuantType::kK, GroupSize},
                stream);
    linear.weight_format = DataFormat{kUint4, {GroupSize, 1}, kHalf, kHalf};
}

constexpr auto mxfp4_packer =
    pack_mxfp4<Sm75, kColMajor, HMMA_16816 | OPERAND_A | 1, kColMajor, HMMA_16816 | OPERAND_U | 1>;

const Family u4_d_32{8, 140, kHalf, kHalf, 32, 8, 1, 1, true, false, supports_u4<32, kHalf>, pack_u4<32, false>};
const Family u4_g_32{9, 139, kHalf, kHalf, 32, 8, 1, 1, true, true, supports_u4<32, kHalf>, pack_u4<32, true>};
const Family u4_d_128{10, 190, kHalf, kHalf, 128, 8, 1, 1, true, false, supports_u4<128, kHalf>, pack_u4<128, false>};
const Family u4_g_128{11, 189, kHalf, kHalf, 128, 8, 1, 1, true, true, supports_u4<128, kHalf>, pack_u4<128, true>};
const Family mxfp4{12, 190, kHalf, kHalf, 32, 8, 1, 1, true, true, supports_mxfp4, mxfp4_packer};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1> class K>
void register_u4_d(Collector& c)
{
    add<K<_128x256x32_1x8x1, 2, kColMajor, D, D, true, 128, 128>>(c);
    add<K<_128x128x32_1x4x1, 2, kColMajor, D, D, true, 64, 128>>(c);
    add<K<_96x64x64_1x2x2, 2, kColMajor, D, S, true>>(c);
    add<K<_64x128x32_1x4x1, 2, kColMajor, D, D, true, 32, 128>>(c);
    add<K<_64x128x32_1x4x1, 2, kColMajor, D, S, true, 32, 128>>(c);
    add<K<_64x64x64_1x2x2, 2, kColMajor, D, S, true>>(c);
    add<K<_48x128x64_1x4x1, 2, kColMajor, D, S, true>>(c);
    add<K<_48x64x64_1x2x2, 2, kColMajor, D, S, true>>(c);
    add<K<_32x64x64_1x2x2, 2, kColMajor, D, S, true>>(c);
    add<K<_16x128x32_1x4x1, 2, kColMajor, D, S, true>>(c);
    add<K<_16x64x64_1x2x2, 2, kColMajor, D, S, true>>(c);
}

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1> class K>
void register_u4_g(Collector& c)
{
    add<K<_128x256x32_2x4x1, 2, kColMajor, D, D, false, 128, 128>>(c);
    add<K<_128x128x32_2x2x1, 2, kColMajor, D, D, true, 64, 128>>(c);
    add<K<_64x128x64_1x4x1, 2, kColMajor, D, S, true, 32, 128>>(c);
    add<K<_64x256x32_1x4x1, 2, kColMajor, D, S, true, 32, 256>>(c);
    add<K<_32x64x128_1x2x2, 2, kColMajor, D, S, true>>(c);
    add<K<_32x128x64_1x4x1, 2, kColMajor, D, S, true>>(c);
    add<K<_16x128x32_1x4x1, 2, kColMajor, D, S, true>>(c);
    add<K<_16x64x64_1x2x2, 2, kColMajor, D, S, true>>(c);
}

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, int GroupAxis = -1> class K>
void register_mxfp4(Collector& c)
{
    add<K<_128x128x32_4x1x1, 2, kColMajor, D, D, true, 128, 64, 1>>(c);
    add<K<_128x64x32_4x1x1, 2, kColMajor, D, D, true, -1, -1, 1>>(c);
    add<K<_128x32x32_4x1x1, 2, kColMajor, S, D, true, -1, -1, 1>>(c);
    add<K<_128x16x32_4x1x1, 2, kColMajor, S, D, true, -1, -1, 1>>(c);
    add<K<_128x16x64_4x1x1, 2, kColMajor, S, D, true, -1, -1, 1>>(c);
    add<K<_64x16x64_4x1x1, 2, kColMajor, S, D, true, -1, -1, 1>>(c);
}

using U4_D_32 = Config_U4_d<32>;
using U4_G_32 = Config_U4_g<32>;
using U4_D_128 = Config_U4_d<128>;
using U4_G_128 = Config_U4_g<128>;
using MXFP4 = Config_MXF4;

Registrar reg[]{
    {u4_d_32, register_u4_d<U4_D_32::Type>},
    {u4_g_32, register_u4_g<U4_G_32::Type>},
    {u4_d_128, register_u4_d<U4_D_128::Type>},
    {u4_g_128, register_u4_g<U4_G_128::Type>},
    {mxfp4, register_mxfp4<MXFP4::Type>},
};
}  // namespace

}  // namespace turbomind::gemm
