// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>
#include <numeric>

#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel/geometry.h"
#include "src/turbomind/kernels/gemm/kernel/mxfp4.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/models/linear_weight.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_mixed_reg.h"

namespace turbomind::gemm {
namespace {
using config::Shape;
using namespace config::geometry;

void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    TM_CHECK_EQ(linear.output_dim % kSm90MixedTileN, 0);
    TM_CHECK_EQ(linear.input_dim % std::lcm(kSm90MixedTileK, Sm90MxFp4Format::kGroupSize), 0);
    TM_CHECK_GE(linear.input_dim, 128);
    PackWeight(linear, Sm90MxFp4Format::kWeightPack, PackSm90Fp4PrmtWeight, stream);
    PackQParams(linear,
                QuantDesc{QuantType::kK, Sm90MxFp4Format::kGroupSize},
                Sm90MxFp4Format::kQparamPack,
                PackSm90MxFp4QParams,
                stream);
    linear.weight_format = DataFormat{kFloat4_e2m1, {Sm90MxFp4Format::kGroupSize, 1}, kUint8};
}

const Family mxfp4{
    30, 250, kBfloat16, kBfloat16, 64, 128, 128, 1, true, true, supports_mxfp4, pack, 64, kBfloat16};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, class ClusterShape = Shape<1, 1>, int MmaN = 0, bool SeparateMmaAtoms = false, int EpiM = 0, int EpiStages = 0> class K>
void register_kernels(Collector& c)
{
    ////////////////////////////////// flat //////////////////////////////////
    add<K<_8x128_1x2<80, 80>, 4, kColMajor, Striding::kFlat, true>>(c);
    add<K<_16x128_1x2<80, 80>, 4, kColMajor, Striding::kFlat, true>>(c);
    add<K<_32x128_1x2<80, 80>, 4, kColMajor, Striding::kFlat, true>>(c);
    add<K<_64x128_1x2<80, 80>, 4, kColMajor, Striding::kFlat, true>>(c);
    add<K<_96x128_1x2<80, 96>, 4, kColMajor, Striding::kFlat, true>>(c);
    add<K<_128x128_1x2<80, 112>, 4, kColMajor, Striding::kFlat, true>>(c);
    add<K<_192x128_1x2<80, 208>, 4, kColMajor, Striding::kFlat>>(c);
    add<K<_224x128_1x2<80, 208>, 4, kColMajor, Striding::kFlat>>(c);
    add<K<_256x128_1x2<80, 208>, 4, kColMajor, Striding::kFlat>>(c);
    add<K<_384x128_1x2<40, 232>, 3, kColMajor, Striding::kFlat, false, Shape<1, 1>, 192>>(c);

    ////////////////////////////////// blocked //////////////////////////////////
    add<K<_8x128_1x2<80, 80>, 4, kColMajor, Striding::kBlocked>>(c);
    add<K<_16x128_1x2<80, 80>, 4, kColMajor, Striding::kBlocked>>(c);
    add<K<_32x128_1x2<80, 80>, 4, kColMajor, Striding::kBlocked>>(c);
    add<K<_64x128_1x2<80, 80>, 4, kColMajor, Striding::kBlocked>>(c);
    add<K<_96x128_1x2<80, 96>, 4, kColMajor, Striding::kBlocked>>(c);
    add<K<_128x128_1x2<80, 112>, 4, kColMajor, Striding::kBlocked>>(c);
    add<K<_192x128_1x2<80, 208>, 4, kColMajor, Striding::kBlocked, true>>(c);
    add<K<_224x128_1x2<80, 208>, 4, kColMajor, Striding::kBlocked, true>>(c);
    add<K<_256x128_1x2<80, 208>, 4, kColMajor, Striding::kBlocked, true>>(c);
    add<K<_384x128_1x2<40, 232>, 3, kColMajor, Striding::kBlocked, true, Shape<1, 1>, 192>>(c);

    ////////////////////////////////// indexed //////////////////////////////////
    add<K<_8x128_1x1<120, 128>, 4, kColMajor, Striding::kIndexed, true>>(c);
    add<K<_16x128_1x2<80, 80>, 4, kColMajor, Striding::kIndexed, true>>(c);
    add<K<_32x128_1x2<80, 80>, 4, kColMajor, Striding::kIndexed, true>>(c);

    add<K<_64x128_1x2<120, 192>, 4, kColMajor, Striding::kIndexed, true>>(c);
    add<K<_96x128_1x2<120, 192>, 4, kColMajor, Striding::kIndexed, true>>(c);
    add<K<_192x128_1x2<120, 192>, 3, kColMajor, Striding::kIndexed, true>>(c);
    add<K<_8x256_1x2<80, 80>, 3, kColMajor, Striding::kIndexed, true>>(c);
    add<K<_16x256_1x2<80, 88>, 3, kColMajor, Striding::kIndexed, true>>(c);
    add<K<_32x256_1x2<120, 192>, 3, kColMajor, Striding::kIndexed, true>>(c);
    add<K<_64x256_1x2<120, 192>, 3, kColMajor, Striding::kIndexed, true>>(c);
    add<K<_96x256_1x2<120, 192>, 3, kColMajor, Striding::kIndexed, true>>(c);
    add<K<_128x256_1x2<120, 192>, 3, kColMajor, Striding::kIndexed, true>>(c);
}

using C = detail::C<Sm90MxFp4Format>;

Registrar reg(mxfp4, register_kernels<C::Type>);
}  // namespace
}  // namespace turbomind::gemm

#endif
