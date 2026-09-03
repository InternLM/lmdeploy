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
    add<Format, kRowMajor, Striding::kFlat, Tile_8x256_S3_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x256_S3_1x2, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x256_S3_1x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_16x256_S3_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_16x256_S3_1x2, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_32x256_S3_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_64x256_S3_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_96x256_S3_2x1, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x256_S3_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x256_S3_1x2, true>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x256_S4_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_160x256_S3_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_192x256_S3_1x2_N96>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_192x192_S3_1x3>(c);

    // Candidates selected by the cluster-1 full-linear-shape sweep.  `uses`
    // counts tuner dispatch records for this exact path/raster/tile.
    add<Format, kRowMajor, Striding::kFlat, Tile_8x128_S4_1x1, true>(c);  // uses: 225
    add<Format, kRowMajor, Striding::kFlat, Tile_8x128_S4_1x2>(c);  // uses: 249
    // add<Format, kRowMajor, Striding::kFlat, Tile_16x128_S4_2x1, true>(c);  // uses: 35
    add<Format, kRowMajor, Striding::kFlat, Tile_16x128_S4_1x2>(c);  // uses: 829
    // add<Format, kRowMajor, Striding::kFlat, Tile_32x128_S4_2x1, true>(c);  // uses: 43
    add<Format, kRowMajor, Striding::kFlat, Tile_32x128_S4_1x2>(c);  // uses: 218
    // add<Format, kRowMajor, Striding::kFlat, Tile_64x128_S4_2x1, true>(c);  // uses: 40
    add<Format, kRowMajor, Striding::kFlat, Tile_64x128_S4_1x2>(c);  // uses: 215
    // add<Format, kRowMajor, Striding::kFlat, Tile_96x128_S4_2x1, true>(c);  // uses: 17
    // add<Format, kRowMajor, Striding::kFlat, Tile_96x128_S4_1x2>(c);  // uses: 107
    // add<Format, kRowMajor, Striding::kFlat, Tile_128x128_S4_2x1, true>(c);  // uses: 46
    add<Format, kRowMajor, Striding::kFlat, Tile_128x128_S4_1x2>(c);  // uses: 176
    // add<Format, kRowMajor, Striding::kFlat, Tile_192x128_S4_2x1, true>(c);  // uses: 90
    add<Format, kRowMajor, Striding::kFlat, Tile_192x128_S4_1x2>(c);  // uses: 704
    // add<Format, kRowMajor, Striding::kFlat, Tile_192x128_S3_2x1, true>(c);  // uses: 1
    // add<Format, kRowMajor, Striding::kFlat, Tile_192x128_S3_1x2>(c);  // uses: 4
    // add<Format, kRowMajor, Striding::kFlat, Tile_224x128_S4_2x1, true>(c);  // uses: 36
    add<Format, kRowMajor, Striding::kFlat, Tile_224x128_S4_1x2>(c);  // uses: 640
    // add<Format, kRowMajor, Striding::kFlat, Tile_224x128_S3_2x1, true>(c);  // uses: 4
    add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S4_2x1, true>(c);  // uses: 616
    add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S4_1x2>(c);  // uses: 1879
    add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S4_1x2_M64>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S4_1x2_M128>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S4_1x2_M256>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_320x128_S4_2x1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_384x128_S4_2x1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_320x128_S4_1x2_N160>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_320x128_S3_1x2_N160>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_384x128_S4_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_384x128_S3_1x2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_384x128_S2_1x2>(c);
    // add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S3_2x1, true>(c);  // uses: 21
    // add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S3_1x2>(c);  // uses: 12
    // Cluster (2,1): multicast prepacked weight / WGMMA operand A across batch CTAs.
    add<Format, kRowMajor, Striding::kFlat, Tile_8x256_S3_1x2, false, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x256_S3_1x2, true, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x256_S3_1x1, true, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_16x256_S3_1x2, false, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_16x256_S3_1x2, true, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_32x256_S3_2x1, true, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_64x256_S3_2x1, true, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_96x256_S3_2x1, true, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x256_S3_1x2, false, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x256_S3_1x2, true, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_160x256_S3_1x2, false, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x128_S4_1x1, true, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x128_S4_1x2, false, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_16x128_S4_1x2, false, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_32x128_S4_1x2, false, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_64x128_S4_1x2, false, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x128_S4_1x2, false, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_192x128_S4_1x2, false, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_224x128_S4_1x2, false, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S4_2x1, true, 1, 2>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S4_1x2, false, 1, 2>(c);

    // Cluster (1,2): multicast BF16 input / WGMMA operand B across output CTAs.
    add<Format, kRowMajor, Striding::kFlat, Tile_8x256_S3_1x2, false, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x256_S3_1x2, true, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x256_S3_1x1, true, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_16x256_S3_1x2, false, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_16x256_S3_1x2, true, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_32x256_S3_2x1, true, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_64x256_S3_2x1, true, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_96x256_S3_2x1, true, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x256_S3_1x2, false, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x256_S3_1x2, true, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_160x256_S3_1x2, false, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x128_S4_1x1, true, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_8x128_S4_1x2, false, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_16x128_S4_1x2, false, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_32x128_S4_1x2, false, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_64x128_S4_1x2, false, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_128x128_S4_1x2, false, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_192x128_S4_1x2, false, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_224x128_S4_1x2, false, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S4_2x1, true, 2, 1>(c);
    add<Format, kRowMajor, Striding::kFlat, Tile_256x128_S4_1x2, false, 2, 1>(c);
#endif
    add<Format, kRowMajor, Striding::kFlat, Tile_384x128_S4_2x1>(c);
});

}  // namespace
}  // namespace turbomind::gemm

#endif
