// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>

#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_e4m3_config.h"
#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_mixed_reg.h"

namespace turbomind::gemm {
namespace {

using Format = Sm90Fp8E4M3Format;
using detail::add;

// BF16 RS WGMMA uses the shared gate64/up64 fused-SiLU contract; this is
// intentionally independent of the gate128/up128 native FP8xFP8 kernels.
Registrar reg([](Collector& c, int /*arch*/) {
    // OUT256 candidates.
    add<Format, kRowMajor, Striding::kFlat, Tile_8x256_S3_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x256_S3_1x2, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x256_S3_1x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_16x256_S3_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_16x256_S3_1x2, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_32x256_S3_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_64x256_S3_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_96x256_S3_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x256_S3_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x256_S3_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_160x256_S3_1x2>(c);

    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_32x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_64x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_96x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_128x256_S3_1x2>(c);

#if 0  // Blocked descriptors intentionally reuse indexed kernels.
    add<Format, kColMajor, Striding::kBlocked, Tile_8x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_8x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_8x256_S3_1x1, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_16x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_16x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_32x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_64x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_96x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_128x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_128x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_160x256_S3_1x2>(c);
#endif

    add<Format, kRowMajor, Striding::kFlat, Tile_8x128_S4_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_16x128_S4_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_32x128_S4_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_64x128_S4_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_96x128_S4_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x128_S4_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_192x128_S4_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S4_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x128_S4_1x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_16x128_S4_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_32x128_S4_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_64x128_S4_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_96x128_S4_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x128_S4_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_192x128_S4_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S4_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S3_2x1, true>(c);

    add<Format, kColMajor, Striding::kIndexed, Tile_8x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_32x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_64x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_96x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_128x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_256x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S3_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_8x128_S4_1x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x128_S4_2x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_32x128_S4_2x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_64x128_S4_2x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_96x128_S4_2x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_128x128_S4_2x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S4_2x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_256x128_S4_2x1, true>(c);

#if 0  // Blocked descriptors intentionally reuse indexed kernels.
    add<Format, kColMajor, Striding::kBlocked, Tile_8x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_16x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_32x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_64x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_96x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_128x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_192x128_S4_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_256x128_S4_1x2>(c);
#endif
});

}  // namespace
}  // namespace turbomind::gemm

#endif
