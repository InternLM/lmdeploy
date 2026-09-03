#pragma once

#include "src/turbomind/kernels/gemm/gmma_bf16_sm90.h"

namespace turbomind::gemm {

// Tile / pipeline knobs for GemmUniversalSm90_Bf16. One struct = one registered
// kernel shape. WGLayout follows the public tile's (M, N) order:
//   Layout<Shape<_1,_2>> — 2 math WGs along tile N (OUT)
//   Layout<Shape<_2,_1>> — 2 math WGs along tile M (BATCH)
// Striding / grouped TMA policy is NOT here — that comes from the kernel's
// (is_grouped, StridingA) template args.
//
// Register budgets are per A-path (TMA vs indexed) and per tile. Pack: 2 math WGs
// → producer + 2*math ≤ 504 (regfile 64K/SM, 128 thr/WG, 8 regs slack). Budgets are
// only tightened where the lower static allocation buys occupancy:
//   Small tiles (8–32x128, 16x256): every region ≤ 80 → 2 CTAs/SM (15–33% faster
//     at small m; 64x128 loses from co-residency, so it stays at 1 CTA/SM).
//     3 CTAs/SM (56-reg regions, stages 3 for 16/32x128) measured 2–17% WORSE
//     (co-residency contention; these kernels are memory-bound) — 2 is the
//     sweet spot, 4 CTAs is not worth pursuing.
//   All other tiles: (120,192,192) = 1 CTA/SM — deviating from the proven math=192
//     without an occupancy gain only risks spills/codegen regressions. Exceptions
//     are measured winners noted per tile below.

namespace detail {

template<int TM,
         int TN,
         int TK,
         class WGLayout_,
         int Stages_,
         int kProdTma,
         int kMathTma,
         int kProdIdx,
         int kMathIdx,
         int MmaN_             = 0,
         bool SeparateMmaAtoms = false,
         int EpiM_             = 0,
         int EpiPipeStages_    = 0>
struct Sm90Bf16TileBase {
    static constexpr int TILE_M = TM;
    static constexpr int TILE_N = TN;
    static constexpr int TILE_K = TK;
    using WGLayout              = WGLayout_;
    static constexpr int Stages = Stages_;

    static constexpr int kProducerRegsTma     = kProdTma;
    static constexpr int kMathRegsTma         = kMathTma;
    static constexpr int kProducerRegsIndexed = kProdIdx;
    static constexpr int kMathRegsIndexed     = kMathIdx;
    static constexpr int kMmaN                = MmaN_;
    static constexpr bool kSeparateMmaAtoms   = SeparateMmaAtoms;
    static constexpr int kEpiM                = EpiM_;
    static constexpr int kEpiPipeStages       = EpiPipeStages_;
};

}  // namespace detail

// N128 × WG_1x2 (WGs along OUT). 8–32: all region caps ≤ 80 so the
// static allocation allows 2 CTAs/SM (15–33% faster at small m). 64+: 1 CTA/SM
// (64x128 loses 6–11% from 2-CTA co-residency), sum ≤ 504.
using Sm90Bf16Tile_8x128_1x2   = detail::Sm90Bf16TileBase<8, 128, 64, WG_1x2, 4, 24, 80, 40, 80>;
using Sm90Bf16Tile_16x128_1x2  = detail::Sm90Bf16TileBase<16, 128, 64, WG_1x2, 4, 24, 80, 40, 80>;
using Sm90Bf16Tile_32x128_1x2  = detail::Sm90Bf16TileBase<32, 128, 64, WG_1x2, 4, 24, 80, 48, 80>;
using Sm90Bf16Tile_64x128_1x2  = detail::Sm90Bf16TileBase<64, 128, 64, WG_1x2, 4, 120, 192, 120, 192>;
using Sm90Bf16Tile_96x128_1x2  = detail::Sm90Bf16TileBase<96, 128, 64, WG_1x2, 4, 120, 192, 120, 192>;
using Sm90Bf16Tile_128x128_1x2 = detail::Sm90Bf16TileBase<128, 128, 64, WG_1x2, 4, 120, 192, 120, 192>;
// Measured winners: indexed 192x128 (104,200,200) is ~8% faster, 256x128
// (128,184,184) is ~18% faster than the (120,192,192) baseline.
using Sm90Bf16Tile_192x128_1x2 = detail::Sm90Bf16TileBase<192, 128, 64, WG_1x2, 4, 120, 192, 104, 200>;
using Sm90Bf16Tile_224x128_1x2 = detail::Sm90Bf16TileBase<224, 128, 64, WG_1x2, 4, 120, 192, 120, 192>;
using Sm90Bf16Tile_256x128_1x2 = detail::Sm90Bf16TileBase<256, 128, 64, WG_1x2, 4, 120, 192, 128, 184>;
using Sm90Bf16Tile_320x128_1x2 = detail::Sm90Bf16TileBase<320, 128, 64, WG_1x2, 3, 40, 232, 56, 224, 160>;
using Sm90Bf16Tile_384x128_1x2 = detail::Sm90Bf16TileBase<384, 128, 64, WG_1x2, 3, 40, 232, 56, 224, 192>;

// N128 × WG_2x1 (WGs along BATCH; TILE_M must be legal for 2 WGs)
using Sm90Bf16Tile_64x128_2x1  = detail::Sm90Bf16TileBase<64, 128, 64, WG_2x1, 4, 120, 192, 120, 192>;
using Sm90Bf16Tile_128x128_2x1 = detail::Sm90Bf16TileBase<128, 128, 64, WG_2x1, 4, 120, 192, 120, 192>;
using Sm90Bf16Tile_320x128_2x1 = detail::Sm90Bf16TileBase<320, 128, 64, WG_2x1, 3, 40, 232, 56, 224>;
using Sm90Bf16Tile_384x128_2x1 = detail::Sm90Bf16TileBase<384, 128, 64, WG_2x1, 3, 24, 240, 56, 224, 0, true, 64>;

// N256: Stages=3 (SMEM). Fused WG_1x2 tiles rebase the operand-A descriptor view so
// each WG owns a contiguous [gate64|up64] range and fuses in registers. 16x256 is
// WG_2x1, while 8x256 also has a single-WG WG_1x1 form (on WG_2x1 its per-WG GMMA-N is 4,
// below the GMMA atom minimum of 8); the epilogue STSM atom tiers
// (gemm_universal_sm90_bf16.h EpiStsmAtoms) cover the narrow per-WG GMMA-N
// (8/16x256: 4 vals/thread).
// 8/16x256: caps ≤ 80 → 2 CTAs/SM; 32x256 is just over the 2-CTA smem budget → ≤ 504.
using Sm90Bf16Tile_8x256_1x2   = detail::Sm90Bf16TileBase<8, 256, 64, WG_1x2, 3, 24, 80, 40, 80>;
using Sm90Bf16Tile_8x256_1x1   = detail::Sm90Bf16TileBase<8, 256, 64, WG_1x1, 3, 24, 80, 40, 80>;
using Sm90Bf16Tile_16x256_2x1  = detail::Sm90Bf16TileBase<16, 256, 64, WG_2x1, 3, 24, 80, 40, 80>;
using Sm90Bf16Tile_64x256_1x2  = detail::Sm90Bf16TileBase<64, 256, 64, WG_1x2, 3, 120, 192, 120, 192>;
using Sm90Bf16Tile_128x256_1x2 = detail::Sm90Bf16TileBase<128, 256, 64, WG_1x2, 3, 120, 192, 120, 192>;
using Sm90Bf16Tile_160x256_1x2 = detail::Sm90Bf16TileBase<160, 256, 64, WG_1x2, 3, 40, 232, 104, 200>;
// One 192-row epilogue buffer reduces the large-tile store path from twelve
// passes to two and measured slightly faster than the default 32-row pipeline.
using Sm90Bf16Tile_192x256_1x2 = detail::Sm90Bf16TileBase<192, 256, 64, WG_1x2, 3, 40, 232, 120, 192, 0, false, 192>;
using Sm90Bf16Tile_32x256_2x1  = detail::Sm90Bf16TileBase<32, 256, 64, WG_2x1, 3, 120, 192, 120, 192>;
using Sm90Bf16Tile_64x256_2x1  = detail::Sm90Bf16TileBase<64, 256, 64, WG_2x1, 3, 120, 192, 120, 192>;
using Sm90Bf16Tile_96x256_2x1  = detail::Sm90Bf16TileBase<96, 256, 64, WG_2x1, 3, 120, 192, 120, 192>;
using Sm90Bf16Tile_128x256_2x1 = detail::Sm90Bf16TileBase<128, 256, 64, WG_2x1, 3, 120, 192, 120, 192>;

}  // namespace turbomind::gemm
