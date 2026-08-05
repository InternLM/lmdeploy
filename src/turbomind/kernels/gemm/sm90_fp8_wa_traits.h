#pragma once

namespace turbomind::gemm {

// Tile / pipeline knobs for GemmUniversalSm90_Fp8Wa (weight-as-A FP8).
// Problem TILE_M = BATCH (act rows) → GMMA-N; TILE_N = OUT (weight) → GMMA-M.
//
// RF policy: dense act scales live on GMMA-N (= TILE_M). Prefer smaller TILE_M
// when math WG RF exceeds setmaxnreg pack (2 WGs → 504).

namespace detail {

template<int TM, int TN, int MaxOpN, int Stages_, int WG_M_ = 1, int WG_N_ = 1>
struct Sm90Fp8WaTileBase {
    static constexpr int TILE_M  = TM;  // BATCH
    static constexpr int TILE_N  = TN;  // OUT
    static constexpr int TILE_K  = 128;
    static constexpr int WG_M    = WG_M_;  // split BATCH across math WGs
    static constexpr int WG_N    = WG_N_;  // split OUT across math WGs
    static constexpr int Stages  = Stages_;
    static constexpr int kMaxOpN = MaxOpN;

    // Small-BATCH tiles need less math RF; 2 math WGs pack to 504 (producer + 2*math).
    static constexpr int kProducerRegsTma     = 40;
    static constexpr int kMathRegsTma         = (WG_M_ * WG_N_ == 2) ? 232 : ((TM <= 32) ? 168 : 208);
    static constexpr int kProducerRegsIndexed = 88;
    static constexpr int kMathRegsIndexed     = (WG_M_ * WG_N_ == 2) ? 208 : ((TM <= 32) ? 168 : 208);
};

}  // namespace detail

// Small BATCH (RF-friendly / decode-friendly)
using Sm90Fp8WaTile_8x128  = detail::Sm90Fp8WaTileBase<8, 128, 8, 4>;
using Sm90Fp8WaTile_16x128 = detail::Sm90Fp8WaTileBase<16, 128, 16, 4>;
using Sm90Fp8WaTile_32x128 = detail::Sm90Fp8WaTileBase<32, 128, 32, 4>;

// Mid BATCH
using Sm90Fp8WaTile_64x128 = detail::Sm90Fp8WaTileBase<64, 128, 64, 4>;

// OUT=256 (gate/up width); Stages=3 for weight SMEM
using Sm90Fp8WaTile_8x256  = detail::Sm90Fp8WaTileBase<8, 256, 8, 3>;
using Sm90Fp8WaTile_16x256 = detail::Sm90Fp8WaTileBase<16, 256, 16, 3>;
using Sm90Fp8WaTile_32x256 = detail::Sm90Fp8WaTileBase<32, 256, 32, 3>;
using Sm90Fp8WaTile_64x256 = detail::Sm90Fp8WaTileBase<64, 256, 64, 3>;

// 2 math WGs (per-WG BATCH=64). 128x* split BATCH (WG_M=2); 64x256_n2 splits OUT
// (WG_N=2, gate/up staged through smem in the fused epilogue).
using Sm90Fp8WaTile_128x128   = detail::Sm90Fp8WaTileBase<128, 128, 64, 3, 2, 1>;
using Sm90Fp8WaTile_128x256   = detail::Sm90Fp8WaTileBase<128, 256, 64, 3, 2, 1>;
using Sm90Fp8WaTile_64x256_n2 = detail::Sm90Fp8WaTileBase<64, 256, 64, 3, 1, 2>;

}  // namespace turbomind::gemm
