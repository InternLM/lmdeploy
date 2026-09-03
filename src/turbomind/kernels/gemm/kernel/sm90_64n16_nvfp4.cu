// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>
#include <numeric>

#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/models/linear_weight.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_mixed_reg.h"
#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_nvfp4_config.h"
#include "src/turbomind/kernels/gemm/kernel/sm90_nvfp4_dequant.h"

namespace turbomind::gemm {
namespace {
std::optional<WeightBridge> supports(const DataFormat& format, bool)
{
    return format == DataFormat{kFloat4_e2m1, {16, 1}, kFloat8_e4m3} ? std::optional{WeightBridge{}} : std::nullopt;
}

void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    TM_CHECK_EQ(linear.output_dim % kSm90MixedTileN, 0);
    TM_CHECK_EQ(linear.input_dim % std::lcm(kSm90MixedTileK, Sm90NvFp4Format::kGroupSize), 0);
    TM_CHECK_GE(linear.input_dim, 128);
    TM_CHECK(linear.global_scale);
    TM_CHECK_EQ(linear.global_scale.dtype(), kFloat);
    TM_CHECK_EQ(linear.global_scale.size(), 1);
    PackWeight(linear, Sm90NvFp4Format::kWeightPack, PackSm90Fp4PrmtWeight, stream);
    PackQParams(linear,
                QuantDesc{QuantType::kK, Sm90NvFp4Format::kGroupSize},
                Sm90NvFp4Format::kQparamPack,
                PackSm90Fp4QParams,
                stream);
    linear.weight_format = DataFormat{kFloat4_e2m1, {Sm90NvFp4Format::kGroupSize, 1}, kFloat8_e4m3};
}

const Family nvfp4{31, 250, kBfloat16, kBfloat16, 64, 128, 128, 1, true, true, supports, pack, 64, kBfloat16};

using Format = Sm90NvFp4Format;
using detail::add;

Registrar reg(nvfp4, [](Collector& c) {
    ////////////////////////////////// flat //////////////////////////////////
    add<Format, kColMajor, Striding::kFlat, Tile_8x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_16x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_32x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_64x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_96x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_128x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_192x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_224x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_256x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_384x128_S3_1x2>(c);

    ////////////////////////////////// blocked //////////////////////////////////
    add<Format, kColMajor, Striding::kBlocked, Tile_8x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_16x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_32x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_64x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_96x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_128x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_192x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_224x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_256x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_384x128_S3_1x2, true>(c);

    ////////////////////////////////// indexed //////////////////////////////////
    add<Format, kColMajor, Striding::kIndexed, Tile_8x128_S4_1x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_32x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_64x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_96x128_S4_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_32x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_64x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_96x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_128x256_S3_1x2, true>(c);
});

}  // namespace
}  // namespace turbomind::gemm

#endif
