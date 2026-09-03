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
    add<Format, kColMajor, Striding::kFlat, Tile_8x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_8x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_8x256_S3_1x1, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_16x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_16x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_32x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_64x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_96x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_128x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_128x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_160x256_S3_1x2>(c);

    // Candidates selected by the cluster-1 full-linear-shape sweep.  `uses`
    // counts tuner dispatch records for this exact path/raster/tile.
    add<Format, kColMajor, Striding::kFlat, Tile_8x128_S4_1x1, true>(c);  // uses: 538
    add<Format, kColMajor, Striding::kFlat, Tile_8x128_S4_1x2>(c);  // uses: 1810
    add<Format, kColMajor, Striding::kFlat, Tile_16x128_S4_2x1, true>(c);  // uses: 126
    add<Format, kColMajor, Striding::kFlat, Tile_16x128_S4_1x2>(c);  // uses: 1237
    add<Format, kColMajor, Striding::kFlat, Tile_32x128_S4_2x1, true>(c);  // uses: 149
    add<Format, kColMajor, Striding::kFlat, Tile_32x128_S4_1x2>(c);  // uses: 667
    add<Format, kColMajor, Striding::kFlat, Tile_64x128_S4_2x1, true>(c);  // uses: 151
    add<Format, kColMajor, Striding::kFlat, Tile_64x128_S4_1x2>(c);  // uses: 670
    // add<Format, kColMajor, Striding::kFlat, Tile_96x128_S4_2x1, true>(c);  // uses: 47
    add<Format, kColMajor, Striding::kFlat, Tile_96x128_S4_1x2>(c);  // uses: 307
    // add<Format, kColMajor, Striding::kFlat, Tile_128x128_S4_2x1, true>(c);  // uses: 95
    add<Format, kColMajor, Striding::kFlat, Tile_128x128_S4_1x2>(c);  // uses: 414
    // add<Format, kColMajor, Striding::kFlat, Tile_192x128_S4_2x1, true>(c);  // uses: 103
    add<Format, kColMajor, Striding::kFlat, Tile_192x128_S4_1x2>(c);  // uses: 519
    // add<Format, kColMajor, Striding::kFlat, Tile_192x128_S3_1x2>(c);  // uses: 1
    // add<Format, kColMajor, Striding::kFlat, Tile_192x128_S2_2x1, true>(c);  // uses: 1
    // add<Format, kColMajor, Striding::kFlat, Tile_224x128_S4_2x1, true>(c);  // uses: 25
    // add<Format, kColMajor, Striding::kFlat, Tile_224x128_S4_1x2>(c);  // uses: 126
    // add<Format, kColMajor, Striding::kFlat, Tile_224x128_S3_2x1, true>(c);  // uses: 5
    // add<Format, kColMajor, Striding::kFlat, Tile_224x128_S2_2x1, true>(c);  // uses: 1
    add<Format, kColMajor, Striding::kFlat, Tile_256x128_S4_2x1, true>(c);  // uses: 180
    add<Format, kColMajor, Striding::kFlat, Tile_256x128_S4_1x2>(c);  // uses: 333
    // add<Format, kColMajor, Striding::kFlat, Tile_256x128_S3_2x1, true>(c);  // uses: 1
    // add<Format, kColMajor, Striding::kFlat, Tile_256x128_S3_1x2>(c);  // uses: 42
    // add<Format, kColMajor, Striding::kFlat, Tile_256x128_S2_2x1, true>(c);  // uses: 1
    // Cluster (2,1): multicast prepacked weight / WGMMA operand A across batch CTAs.
    add<Format, kColMajor, Striding::kFlat, Tile_8x256_S3_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_8x256_S3_1x2, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_8x256_S3_1x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_16x256_S3_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_16x256_S3_1x2, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_32x256_S3_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_64x256_S3_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_96x256_S3_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_128x256_S3_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_128x256_S3_1x2, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_160x256_S3_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_8x128_S4_1x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_8x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_16x128_S4_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_16x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_32x128_S4_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_32x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_64x128_S4_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_64x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_96x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_128x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_192x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_256x128_S4_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_256x128_S4_1x2, false, 1, 2>(c);

    // Cluster (1,2): multicast BF16 input / WGMMA operand B across output CTAs.
    add<Format, kColMajor, Striding::kFlat, Tile_8x256_S3_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_8x256_S3_1x2, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_8x256_S3_1x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_16x256_S3_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_16x256_S3_1x2, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_32x256_S3_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_64x256_S3_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_96x256_S3_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_128x256_S3_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_128x256_S3_1x2, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_160x256_S3_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_8x128_S4_1x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_8x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_16x128_S4_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_16x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_32x128_S4_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_32x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_64x128_S4_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_64x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_96x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_128x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_192x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_256x128_S4_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kFlat, Tile_256x128_S4_1x2, false, 2, 1>(c);
#endif
});

}  // namespace
}  // namespace turbomind::gemm

#endif
