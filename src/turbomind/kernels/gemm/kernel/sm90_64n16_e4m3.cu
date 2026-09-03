// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>

#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel/e4m3.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/models/linear_weight.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_e4m3_config.h"
#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_mixed_reg.h"

namespace turbomind::gemm {
namespace {
void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    TM_CHECK_EQ(linear.output_dim % kSm90MixedTileN, 0);
    TM_CHECK_EQ(linear.input_dim % Sm90Fp8E4M3Format::kGroupSize, 0);
    TM_CHECK_GE(linear.input_dim, 128);
    PackWeight(linear, Sm90Fp8E4M3Format::kWeightPack, PackSm90Fp8E4M3Weight, stream);

    TM_CHECK_EQ(linear.scales.dtype(), kFloat);
    TM_CHECK_EQ(linear.scales.ndim(), 2);
    TM_CHECK_EQ(linear.scales.shape(0), linear.input_dim / Sm90Fp8E4M3Format::kGroupSize);
    TM_CHECK_EQ(linear.scales.shape(1), linear.output_dim / Sm90Fp8E4M3Format::kScaleGroupN);
    TM_CHECK(linear.scales.is_contiguous());
    MatrixLayout s_desc{kFloat,
                        kRowMajor,
                        (int)linear.scales.shape(0),
                        (int)linear.scales.shape(1),
                        (int)linear.scales.stride(0),
                        0,
                        0,
                        nullptr,
                        nullptr};
    Tensor       packed_q{{linear.scales.size() * Sm90Fp8E4M3Format::kQparamValuesTile}, kBfloat16, kDEVICE};
    PackSm90Fp8E4M3Scales(static_cast<bfloat16_t*>(packed_q.raw_data()),
                          static_cast<const float*>(linear.scales.raw_data()),
                          s_desc.rows,
                          s_desc.cols,
                          stream);
    linear.scales = std::move(packed_q);
    linear.q_desc = MatrixLayout{kBfloat16,
                                 kRowMajor,
                                 s_desc.rows,
                                 s_desc.cols,
                                 s_desc.cols * Sm90Fp8E4M3Format::kQparamValuesTile,
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
                  128,
                  128,
                  128,
                  1,
                  true,
                  true,
                  supports_e4m3<kFloat, Sm90Fp8E4M3Format::kScaleGroupN>,
                  pack,
                  64,
                  kBfloat16};

using Format = Sm90Fp8E4M3Format;
using detail::add;

// BF16 RS WGMMA uses the shared gate64/up64 fused-SiLU contract; this is
// intentionally independent of the gate128/up128 native FP8xFP8 kernels.
Registrar reg(e4m3, [](Collector& c) {
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
