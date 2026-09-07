// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>

#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel/geometry.h"
#include "src/turbomind/kernels/gemm/kernel/mxfp4.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/models/linear_weight.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/gemm_universal_sm90_mxfp4_fp8_unfolded.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90_mxfp4_fp8.h"
#include "src/turbomind/kernels/gemm/registrar.h"

namespace turbomind::gemm {
namespace {
using namespace config::geometry;

void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    TM_CHECK_EQ(linear.output_dim % kSm90MixedFragmentN, 0);
    TM_CHECK_EQ(linear.input_dim % 128, 0);
    TM_CHECK_GE(linear.input_dim, 128);
    PackWeight(linear, Sm90MxFp4Fp8UnfoldedFormat::kWeightPack, PackSm90MxFp4Fp8UnfoldedWeight, stream);
    PackQParams(linear,
                QuantDesc{QuantType::kK, Sm90MxFp4Fp8UnfoldedFormat::kGroupSize},
                Sm90MxFp4Fp8UnfoldedFormat::kQparamPack,
                PackSm90MxFp4Fp8UnfoldedQParams,
                stream);
    linear.weight_format = DataFormat{kFloat4_e2m1, {Sm90MxFp4Fp8UnfoldedFormat::kGroupSize, 1}, kUint8};
}

const Family unfolded{
    34, 250, DataFormat{kFloat8_e4m3, {128, 1}, kFloat}, kBfloat16, 128, 64, 256, 1, true, false, supports_mxfp4, pack};

struct C {
    template<class Config_, int Stages, Order Raster, int MmaN = Config_::Tile::M / Config_::Groups::M>
    using Type = KernelImplSm90MxFp4Fp8<GemmUniversalSm90MxFp4Fp8Unfolded<Config_, Stages, Raster, MmaN>>;
};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, int MmaN = Config_::Tile::M / Config_::Groups::M> class K>
void register_kernels(Collector& c)
{
    add<K<_64x128_1x2<40, 232>, 3, kRowMajor>>(c);
}

Registrar reg(unfolded, register_kernels<C::Type>);
}  // namespace
}  // namespace turbomind::gemm

#endif
