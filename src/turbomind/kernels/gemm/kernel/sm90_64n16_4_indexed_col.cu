// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>

#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_mixed_reg.h"
#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_4_config.h"

namespace turbomind::gemm {
namespace {

using Format = Sm90U4Format;
using detail::add;

Registrar reg([](Collector& c, int /*arch*/) {
#if 0
    // OUT256 production candidates. Fused-capable
    // entries also accept the normal epilogue.
    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_32x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_64x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_96x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_128x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_128x256_S3_1x2, true>(c);

    // Candidates selected by the cluster-1 full-linear-shape sweep.  `uses`
    // counts tuner dispatch records for this exact path/raster/tile.
    add<Format, kColMajor, Striding::kIndexed, Tile_8x128_S4_1x1, true>(c);  // uses: 2486
    add<Format, kColMajor, Striding::kIndexed, Tile_16x128_S4_2x1, true>(c);  // uses: 269
    add<Format, kColMajor, Striding::kIndexed, Tile_16x128_S4_1x2>(c);  // uses: 349
    add<Format, kColMajor, Striding::kIndexed, Tile_32x128_S4_2x1, true>(c);  // uses: 247
    add<Format, kColMajor, Striding::kIndexed, Tile_32x128_S4_1x2>(c);  // uses: 252
    add<Format, kColMajor, Striding::kIndexed, Tile_64x128_S4_2x1, true>(c);  // uses: 206
    add<Format, kColMajor, Striding::kIndexed, Tile_64x128_S4_1x2>(c);  // uses: 201
    // add<Format, kColMajor, Striding::kIndexed, Tile_96x128_S4_2x1, true>(c);  // uses: 42
    add<Format, kColMajor, Striding::kIndexed, Tile_96x128_S4_1x2>(c);  // uses: 248
    // add<Format, kColMajor, Striding::kIndexed, Tile_128x128_S4_2x1, true>(c);  // uses: 117
    // add<Format, kColMajor, Striding::kIndexed, Tile_128x128_S4_1x2>(c);  // uses: 87
    add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S4_2x1, true>(c);  // uses: 327
    // add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S4_1x2>(c);  // uses: 110
    // add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S3_2x1, true>(c);  // uses: 1
    add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S3_1x2>(c);  // uses: 161
    // add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S2_1x2>(c);  // uses: 27
    // add<Format, kColMajor, Striding::kIndexed, Tile_224x128_S4_2x1, true>(c);  // uses: 6
    // add<Format, kColMajor, Striding::kIndexed, Tile_224x128_S4_1x2>(c);  // uses: 76
    // add<Format, kColMajor, Striding::kIndexed, Tile_224x128_S3_2x1, true>(c);  // uses: 101
    // add<Format, kColMajor, Striding::kIndexed, Tile_224x128_S3_1x2>(c);  // uses: 5
    // add<Format, kColMajor, Striding::kIndexed, Tile_224x128_S2_1x2>(c);  // uses: 1
    // Indexed 256x128 is dominated by 128x256 after the per-fragment gather rewrite.
    // add<Format, kColMajor, Striding::kIndexed, Tile_256x128_S4_2x1, true>(c);  // uses: 396
    // add<Format, kColMajor, Striding::kIndexed, Tile_256x128_S4_1x2>(c);  // uses: 446
    // add<Format, kColMajor, Striding::kIndexed, Tile_256x128_S3_2x1, true>(c);  // uses: 1
    // add<Format, kColMajor, Striding::kIndexed, Tile_256x128_S3_1x2>(c);  // uses: 19
    // Cluster (2,1): multicast prepacked weight / WGMMA operand A across batch CTAs.
    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x2, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x256_S3_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x256_S3_1x2, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_32x256_S3_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_64x256_S3_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_96x256_S3_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_128x256_S3_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_128x256_S3_1x2, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_8x128_S4_1x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x128_S4_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_32x128_S4_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_32x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_64x128_S4_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_64x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_96x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S4_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S3_1x2, false, 1, 2>(c);
    // add<Format, kColMajor, Striding::kIndexed, Tile_256x128_S4_2x1, true, 1, 2>(c);
    // add<Format, kColMajor, Striding::kIndexed, Tile_256x128_S4_1x2, false, 1, 2>(c);

    // Cluster (1,2): multicast BF16 input / WGMMA operand B across output CTAs.
    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x2, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x256_S3_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x256_S3_1x2, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_32x256_S3_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_64x256_S3_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_96x256_S3_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_128x256_S3_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_128x256_S3_1x2, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_8x128_S4_1x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x128_S4_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_16x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_32x128_S4_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_32x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_64x128_S4_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_64x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_96x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S4_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kIndexed, Tile_192x128_S3_1x2, false, 2, 1>(c);
    // add<Format, kColMajor, Striding::kIndexed, Tile_256x128_S4_2x1, true, 2, 1>(c);
    // add<Format, kColMajor, Striding::kIndexed, Tile_256x128_S4_1x2, false, 2, 1>(c);
#endif
    (void)c;
});

}  // namespace
}  // namespace turbomind::gemm

#endif
