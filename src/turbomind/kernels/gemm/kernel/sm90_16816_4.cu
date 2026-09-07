// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/kernel/geometry.h"
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
using namespace config::geometry;

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

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, bool FusePrefetch = true> class K>
void register_u4_d(Collector& c)
{
    add<K<_128x256x64_1x8x1, 3, kColMajor, D, D, true>>(c);
    add<K<_128x256x32_1x8x1, 3, kColMajor, D, D, true, 128, 128>>(c);
    add<K<_128x256x32_1x8x1, 4, kColMajor, D, D, true, 128, 128>>(c);
    add<K<_128x128x32_1x4x1, 3, kColMajor, D, D, true, 64, 128>>(c);
    add<K<_128x128x32_1x4x1, 4, kColMajor, D, D, true, 64, 128>>(c);
    add<K<_128x128x64_1x4x2, 3, kColMajor, D, D, true, 64, 128>>(c);

    add<K<_96x256x32_1x8x1, 4, kColMajor, D, D, true>>(c);
    add<K<_96x256x32_1x8x1, 3, kColMajor, D, D, true>>(c);
    add<K<_96x128x32_1x4x1, 4, kColMajor, D, D, true>>(c);
    add<K<_96x128x128_1x4x2, 3, kColMajor, D, D, true>>(c);

    add<K<_64x256x32_1x4x1, 3, kColMajor, D, D, true, 64, 128>>(c);
    add<K<_64x256x32_1x4x1, 4, kColMajor, D, D, true, 64, 128>>(c);
    add<K<_64x128x32_1x4x1, 4, kColMajor, D, D, true>>(c);
    add<K<_64x128x64_1x4x1, 3, kColMajor, D, D, true>>(c);
    add<K<_64x128x128_1x4x2, 3, kColMajor, D, D, true>>(c);
    add<K<_64x64x64_1x2x2, 6, kColMajor, D, D, true>>(c);

    add<K<_48x256x64_1x4x1, 3, kColMajor, D, D, true, 48, 128>>(c);
    add<K<_48x128x64_1x4x1, 4, kColMajor, D, D, true>>(c);
    add<K<_48x128x128_1x4x2, 3, kColMajor, D, D, true>>(c);
    add<K<_48x64x128_1x2x2, 4, kColMajor, D, D, true>>(c);

    add<K<_32x256x64_1x4x1, 3, kColMajor, D, D, true>>(c);
    add<K<_32x128x64_1x4x1, 4, kColMajor, D, D, true>>(c);
    add<K<_32x128x128_1x4x2, 3, kColMajor, D, D, true>>(c);
    add<K<_32x64x128_1x2x2, 3, kColMajor, D, D, true>>(c);
    add<K<_32x64x128_1x2x2, 4, kColMajor, D, D, true>>(c);

    add<K<_16x128x64_1x4x1, 4, kColMajor, D, D, true>>(c);
    add<K<_16x128x128_1x4x2, 3, kColMajor, D, D, true>>(c);
    add<K<_16x128x128_1x4x2, 4, kColMajor, D, D, true>>(c);
    add<K<_16x64x128_1x2x2, 3, kColMajor, D, D, true>>(c);
    add<K<_16x64x128_1x2x2, 4, kColMajor, D, D, true>>(c);
}

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, bool FusePrefetch = true> class K>
void register_u4_g(Collector& c)
{
    add<K<_128x256x32_2x4x1, 3, kColMajor, D, D, false>>(c);
    add<K<_128x128x32_1x4x1, 3, kColMajor, D, D, true>>(c);
    add<K<_64x128x64_1x4x1, 3, kColMajor, D, D, true>>(c);
    add<K<_64x256x32_1x4x1, 3, kColMajor, D, D, true>>(c);
    add<K<_32x64x128_1x2x2, 3, kColMajor, D, D, true>>(c);
    add<K<_32x128x64_1x4x1, 5, kColMajor, D, D, true>>(c);
    add<K<_32x256x64_1x4x1, 3, kColMajor, D, D, true>>(c);
    add<K<_16x256x64_1x4x1, 3, kColMajor, D, D, true>>(c);
    add<K<_16x256x32_1x4x1, 3, kColMajor, D, D, true>>(c);
    add<K<_16x128x64_1x4x1, 3, kColMajor, D, D, true>>(c);
    add<K<_16x64x128_1x2x2, 3, kColMajor, D, D, true>>(c);
}

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, class PolicyA, class PolicyB, bool SplitK, int EpiM = -1, int EpiN = -1, bool FusePrefetch = true, int GroupAxis = 1, int OperandN = 16> class K>
void register_mxfp4(Collector& c)
{
    // add<K<_256x128x32_8x1x1, 3, kColMajor, D, D, true, 128, 128, true, -1>>(c);

    add<K<_256x128x32_8x1x1, 3, kColMajor, D, D, true, 128, 128>>(c);
    add<K<_256x64x32_4x1x1, 3, kColMajor, D, D, true, 128, 64>>(c);
    add<K<_256x32x32_4x1x1, 5, kColMajor, D, D, true>>(c);
    add<K<_128x128x32_4x1x1, 4, kColMajor, D, D, true, 128, 64>>(c);
    add<K<_128x96x32_4x1x1, 3, kColMajor, D, D, true>>(c);
    add<K<_128x64x32_4x1x1, 3, kColMajor, D, D, true>>(c);
    add<K<_128x32x32_4x1x1, 3, kColMajor, D, D, true>>(c);
    add<K<_128x16x32_4x1x1, 5, kColMajor, D, D, true>>(c);
    add<K<_128x16x64_4x1x1, 3, kColMajor, D, D, true>>(c);

    add<K<_256x8x32_4x1x1, 5, kColMajor, D, D, true, -1, -1, true, 1, 8>>(c);
    add<K<_128x8x32_4x1x1, 5, kColMajor, D, D, true, -1, -1, true, 1, 8>>(c);
    add<K<_128x8x64_4x1x1, 3, kColMajor, D, D, true, -1, -1, true, 1, 8>>(c);
    add<K<_64x8x64_4x1x1, 5, kColMajor, D, D, true, -1, -1, true, 1, 8>>(c);
}

using U4_D_32 = Config_U4_d<Sm90, half, 32>;
using U4_G_32 = Config_U4_g<Sm90, half, 32>;
using U4_D_128 = Config_U4_d<Sm90, half, 128>;
using U4_G_128 = Config_U4_g<Sm90, half, 128>;
using MXFP4 = Config_MXF4<Sm90, bfloat16_t>;

Registrar reg[]{
    {u4_d_32, register_u4_d<U4_D_32::Type>},
    {u4_g_32, register_u4_g<U4_G_32::Type>},
    {u4_d_128, register_u4_d<U4_D_128::Type>},
    {u4_g_128, register_u4_g<U4_G_128::Type>},
    {mxfp4, register_mxfp4<MXFP4::Type>},
};
}  // namespace

}  // namespace turbomind::gemm
