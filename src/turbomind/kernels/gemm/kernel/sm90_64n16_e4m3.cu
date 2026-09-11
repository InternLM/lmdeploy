// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>

#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/family.h"
#include "src/turbomind/kernels/gemm/kernel/geometry.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/models/linear_weight.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_mixed_reg.h"

namespace turbomind::gemm {
namespace {
using config::Shape;
using namespace config::geometry;

// The kernel scales one K64 group per TILE_K and one scale per output column,
// so checkpoint blocking is expanded by the weight bridge instead of being
// handled by the packing kernels.  Replication covers whole N128 blocks, so a
// narrower source block would leave the last tile row short of its columns.
std::optional<WeightBridge> supports(const DataFormat& format, bool)
{
    if (format.dtype != kFloat8_e4m3 || format.block_sizes.size() != 2) {
        return std::nullopt;
    }
    if (format.block_sizes[1] % kSm90MixedTileN || format.block_sizes[0] % Sm90Fp8E4M3Format::kGroupSize) {
        return std::nullopt;
    }
    if (!IsFloatFormatType(format.scales.dtype) || format.zeros.present()) {
        return std::nullopt;
    }
    WeightBridge bridge;
    bridge.replicate_scales.x = format.block_sizes[0] / Sm90Fp8E4M3Format::kGroupSize;
    bridge.replicate_scales.y = format.block_sizes[1] / Sm90Fp8E4M3Format::kScaleGroupN;
    bridge.convert_scales     = kFloat;
    return bridge;
}

void pack(LinearWeight& linear, cudaStream_t stream)
{
    TM_CHECK_EQ(linear.output_dim % kSm90MixedFragmentN, 0);
    TM_CHECK_EQ(linear.input_dim % kSm90MixedTileK, 0);
    TM_CHECK_GE(linear.input_dim, 128);
    PackWeight(linear, Sm90Fp8E4M3Format::kWeightPack, PackSm90Fp8E4M3Weight, stream);

    TM_CHECK_EQ(linear.scales.dtype(), kFloat);
    TM_CHECK_EQ(linear.scales.ndim(), 2);
    const int group_count = linear.input_dim / Sm90Fp8E4M3Format::kGroupSize;
    // Bridge replication yields at least one K64 row per group and one column
    // per weight column. Padding past the last K64 group or past the last
    // output column is tolerated, but only columns < output_dim are read, so a
    // row narrowed by the output split is as valid as a whole-block one.
    const int scales_stride = linear.scales.shape(1);
    TM_CHECK_GE(linear.scales.shape(0), group_count);
    TM_CHECK_GE(scales_stride, linear.output_dim);
    TM_CHECK(linear.scales.is_contiguous());
    const int fragments_n = (linear.output_dim + kSm90MixedFragmentN - 1) / kSm90MixedFragmentN;
    Tensor    packed_q{{group_count * fragments_n * Sm90Fp8E4M3Format::kQparamValuesFragment}, kBfloat16, kDEVICE};
    PackSm90Fp8E4M3Scales(static_cast<bfloat16_t*>(packed_q.raw_data()),
                          static_cast<const float*>(linear.scales.raw_data()),
                          group_count,
                          linear.output_dim,
                          scales_stride,
                          stream);
    linear.scales = std::move(packed_q);
    linear.q_desc = MatrixLayout{kBfloat16,
                                 kRowMajor,
                                 group_count,
                                 linear.output_dim,
                                 fragments_n * Sm90Fp8E4M3Format::kQparamValuesFragment,
                                 Sm90Fp8E4M3Format::kQparamPack,
                                 0,
                                 nullptr,
                                 nullptr};
    linear.weight_format =
        DataFormat{kFloat8_e4m3, {Sm90Fp8E4M3Format::kGroupSize, Sm90Fp8E4M3Format::kScaleGroupN}, kBfloat16};
}

const Family e4m3{32,
                  250,
                  kBfloat16,
                  kBfloat16,
                  64,
                  64,
                  128,
                  1,
                  true,
                  true,
                  supports,
                  pack,
                  64,
                  kBfloat16};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_,
                  int      Stages,
                  Order    Raster,
                  Striding Mode,
                  bool     Silu         = false,
                  class ClusterShape    = Shape<1, 1>,
                  int  MmaN             = 0,
                  bool SeparateMmaAtoms = false,
                  int  EpiM             = 0,
                  int  EpiStages        = 0>
         class K>
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

using C = detail::C<Sm90Fp8E4M3Format>;

Registrar reg(e4m3, register_kernels<C::Type>);
}  // namespace
}  // namespace turbomind::gemm

#endif
