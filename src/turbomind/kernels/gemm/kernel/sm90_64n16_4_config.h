// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/sm90_mixed_traits.h"

namespace turbomind::gemm {

namespace detail {

template<int Batch,
         int Out,
         int Stages_,
         class WGLayout_,
         int ProducerRegsTma,
         int MathRegsTma,
         int ProducerRegsIndexed,
         int MathRegsIndexed>
struct TileBase {
    static constexpr int TILE_BATCH = Batch;
    static constexpr int TILE_OUT   = Out;
    static constexpr int Stages     = Stages_;
    using WGLayout                  = WGLayout_;

    static constexpr int kProducerRegsTma     = ProducerRegsTma;
    static constexpr int kMathRegsTma         = MathRegsTma;
    static constexpr int kProducerRegsIndexed = ProducerRegsIndexed;
    static constexpr int kMathRegsIndexed     = MathRegsIndexed;
};

}  // namespace detail

// OUT128 policies for the live production catalog. Small tiles cap every
// warpgroup at the two-CTA threshold; large tiles transfer registers from the
// producer to the math warpgroups.
using Tile_8x128_S4_1x1 = detail::TileBase<8, 128, 4, WG_1x1, 80, 128, 120, 128>;

using Tile_8x128_S4_1x2   = detail::TileBase<8, 128, 4, WG_1x2, 80, 80, 80, 80>;
using Tile_16x128_S4_1x2  = detail::TileBase<16, 128, 4, WG_1x2, 80, 80, 80, 80>;
using Tile_32x128_S4_1x2  = detail::TileBase<32, 128, 4, WG_1x2, 80, 80, 80, 80>;
using Tile_64x128_S4_1x2  = detail::TileBase<64, 128, 4, WG_1x2, 80, 80, 120, 192>;
using Tile_96x128_S4_1x2  = detail::TileBase<96, 128, 4, WG_1x2, 80, 96, 120, 192>;
using Tile_128x128_S4_1x2 = detail::TileBase<128, 128, 4, WG_1x2, 80, 112, 120, 192>;

using Tile_16x128_S4_2x1  = detail::TileBase<16, 128, 4, WG_2x1, 80, 80, 80, 80>;
using Tile_32x128_S4_2x1  = detail::TileBase<32, 128, 4, WG_2x1, 80, 80, 80, 80>;
using Tile_64x128_S4_2x1  = detail::TileBase<64, 128, 4, WG_2x1, 80, 96, 120, 192>;
using Tile_96x128_S4_2x1  = detail::TileBase<96, 128, 4, WG_2x1, 80, 112, 120, 192>;
using Tile_128x128_S4_2x1 = detail::TileBase<128, 128, 4, WG_2x1, 80, 128, 120, 192>;

using Tile_192x128_S4_1x2 = detail::TileBase<192, 128, 4, WG_1x2, 80, 208, 120, 192>;
using Tile_192x128_S3_1x2 = detail::TileBase<192, 128, 3, WG_1x2, 80, 208, 120, 192>;
using Tile_224x128_S4_1x2 = detail::TileBase<224, 128, 4, WG_1x2, 80, 208, 120, 192>;
using Tile_256x128_S4_1x2 = detail::TileBase<256, 128, 4, WG_1x2, 80, 208, 120, 192>;
using Tile_256x128_S5_1x2 = detail::TileBase<256, 128, 5, WG_1x2, 80, 208, 120, 192>;
struct Tile_256x128_S4_1x2_M64: detail::TileBase<256, 128, 4, WG_1x2, 80, 208, 120, 192> {
    static constexpr int kEpiM = 64;
};
struct Tile_256x128_S4_1x2_M128: detail::TileBase<256, 128, 4, WG_1x2, 80, 208, 120, 192> {
    static constexpr int kEpiM = 128;
};
struct Tile_256x128_S4_1x2_M256: detail::TileBase<256, 128, 4, WG_1x2, 80, 208, 120, 192> {
    static constexpr int kEpiM = 256;
};
using Tile_256x128_S3_1x2 = detail::TileBase<256, 128, 3, WG_1x2, 80, 208, 120, 192>;

using Tile_192x128_S4_2x1 = detail::TileBase<192, 128, 4, WG_2x1, 80, 208, 120, 192>;
using Tile_256x128_S4_2x1 = detail::TileBase<256, 128, 4, WG_2x1, 80, 208, 120, 192>;
using Tile_320x128_S4_2x1 = detail::TileBase<320, 128, 4, WG_2x1, 40, 232, 56, 224>;
struct Tile_384x128_S4_2x1: detail::TileBase<384, 128, 4, WG_2x1, 24, 240, 56, 224> {
    static constexpr bool kSeparateMmaAtoms = true;
    static constexpr int  kEpiM             = 64;
};
struct Tile_320x128_S4_1x2_N160: detail::TileBase<320, 128, 4, WG_1x2, 40, 232, 56, 224> {
    static constexpr int kMmaN = 160;
};
struct Tile_320x128_S3_1x2_N160: detail::TileBase<320, 128, 3, WG_1x2, 40, 232, 56, 224> {
    static constexpr int kMmaN = 160;
};
struct Tile_384x128_S4_1x2: detail::TileBase<384, 128, 4, WG_1x2, 40, 232, 56, 224> {
    static constexpr int kMmaN = 192;
};
struct Tile_384x128_S3_1x2: detail::TileBase<384, 128, 3, WG_1x2, 40, 232, 56, 224> {
    static constexpr int kMmaN = 192;
};
struct Tile_384x128_S2_1x2: detail::TileBase<384, 128, 2, WG_1x2, 24, 240, 56, 224> {
    static constexpr int kMmaN = 192;
};
using Tile_192x192_S3_1x3 = detail::TileBase<192, 192, 3, WG_1x3, 40, 152, 56, 144>;

// OUT256 uses stage 3. BATCH 8 provides one-WG and output-split forms;
// BATCH 16, 128, and 160 use output-split forms to avoid WGMMA serialization,
// while BATCH 32..96 use activation-split forms.
using Tile_8x256_S3_1x2   = detail::TileBase<8, 256, 3, WG_1x2, 80, 80, 80, 80>;
using Tile_8x256_S3_1x1   = detail::TileBase<8, 256, 3, WG_1x1, 80, 128, 120, 128>;
using Tile_16x256_S3_1x2  = detail::TileBase<16, 256, 3, WG_1x2, 64, 96, 80, 88>;
using Tile_32x256_S3_1x2 = detail::TileBase<32, 256, 3, WG_1x2, 80, 208, 120, 192>;
using Tile_64x256_S3_1x2  = detail::TileBase<64, 256, 3, WG_1x2, 80, 208, 120, 192>;
using Tile_96x256_S3_1x2  = detail::TileBase<96, 256, 3, WG_1x2, 80, 208, 120, 192>;
using Tile_128x256_S3_1x2 = detail::TileBase<128, 256, 3, WG_1x2, 72, 216, 120, 192>;
using Tile_128x256_S4_1x2 = detail::TileBase<128, 256, 4, WG_1x2, 72, 216, 120, 192>;
using Tile_128x256_S5_1x2 = detail::TileBase<128, 256, 5, WG_1x2, 72, 216, 120, 192>;
using Tile_160x256_S3_1x2 = detail::TileBase<160, 256, 3, WG_1x2, 40, 232, 56, 224>;
struct Tile_192x256_S3_1x2_N96: detail::TileBase<192, 256, 3, WG_1x2, 40, 232, 56, 224> {
    static constexpr int kMmaN = 96;
};

}  // namespace turbomind::gemm
