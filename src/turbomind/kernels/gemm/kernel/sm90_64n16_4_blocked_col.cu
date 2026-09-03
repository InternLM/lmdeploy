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
#if 0  // Blocked descriptors intentionally reuse indexed kernels.
    // OUT256 production candidates. Fused-capable
    // entries also accept the normal epilogue.
    add<Format, kColMajor, Striding::kBlocked, Tile_8x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_8x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_8x256_S3_1x1, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_16x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_16x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_32x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_64x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_96x256_S3_2x1, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_128x256_S3_1x2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_128x256_S3_1x2, true>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_160x256_S3_1x2>(c);

    // Candidates selected by the cluster-1 full-linear-shape sweep.  `uses`
    // counts tuner dispatch records for this exact path/raster/tile.
    // add<Format, kColMajor, Striding::kBlocked, Tile_8x128_S4_1x1, true>(c);  // uses: 18
    add<Format, kColMajor, Striding::kBlocked, Tile_8x128_S4_1x2>(c);  // uses: 193
    add<Format, kColMajor, Striding::kBlocked, Tile_16x128_S4_1x2>(c);  // uses: 467
    add<Format, kColMajor, Striding::kBlocked, Tile_32x128_S4_1x2>(c);  // uses: 163
    // add<Format, kColMajor, Striding::kBlocked, Tile_64x128_S4_2x1, true>(c);  // uses: 1
    // add<Format, kColMajor, Striding::kBlocked, Tile_64x128_S4_1x2>(c);  // uses: 72
    // add<Format, kColMajor, Striding::kBlocked, Tile_96x128_S4_1x2>(c);  // uses: 71
    // add<Format, kColMajor, Striding::kBlocked, Tile_128x128_S4_1x2>(c);  // uses: 54
    // add<Format, kColMajor, Striding::kBlocked, Tile_192x128_S4_2x1, true>(c);  // uses: 15
    add<Format, kColMajor, Striding::kBlocked, Tile_192x128_S4_1x2>(c);  // uses: 135
    // add<Format, kColMajor, Striding::kBlocked, Tile_192x128_S3_1x2>(c);  // uses: 2
    // add<Format, kColMajor, Striding::kBlocked, Tile_224x128_S4_2x1, true>(c);  // uses: 22
    // add<Format, kColMajor, Striding::kBlocked, Tile_224x128_S4_1x2>(c);  // uses: 29
    // add<Format, kColMajor, Striding::kBlocked, Tile_224x128_S3_2x1, true>(c);  // uses: 1
    // add<Format, kColMajor, Striding::kBlocked, Tile_256x128_S4_2x1, true>(c);  // uses: 22
    // add<Format, kColMajor, Striding::kBlocked, Tile_256x128_S4_1x2>(c);  // uses: 21
    // add<Format, kColMajor, Striding::kBlocked, Tile_256x128_S3_2x1, true>(c);  // uses: 3
    // Cluster (2,1): multicast prepacked weight / WGMMA operand A across batch CTAs.
    add<Format, kColMajor, Striding::kBlocked, Tile_8x256_S3_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_8x256_S3_1x2, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_8x256_S3_1x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_16x256_S3_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_16x256_S3_1x2, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_32x256_S3_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_64x256_S3_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_96x256_S3_2x1, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_128x256_S3_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_128x256_S3_1x2, true, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_160x256_S3_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_8x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_16x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_32x128_S4_1x2, false, 1, 2>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_192x128_S4_1x2, false, 1, 2>(c);

    // Cluster (1,2): multicast BF16 input / WGMMA operand B across output CTAs.
    add<Format, kColMajor, Striding::kBlocked, Tile_8x256_S3_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_8x256_S3_1x2, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_8x256_S3_1x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_16x256_S3_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_16x256_S3_1x2, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_32x256_S3_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_64x256_S3_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_96x256_S3_2x1, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_128x256_S3_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_128x256_S3_1x2, true, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_160x256_S3_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_8x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_16x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_32x128_S4_1x2, false, 2, 1>(c);
    add<Format, kColMajor, Striding::kBlocked, Tile_192x128_S4_1x2, false, 2, 1>(c);
#endif
    (void)c;
});

}  // namespace
}  // namespace turbomind::gemm

#endif
