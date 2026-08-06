#pragma once

namespace turbomind::gemm {

// Tile / pipeline knobs for GemmUniversalSm90_v3. One struct = one registered
// kernel shape. Striding / grouped TMA policy is NOT here — that comes from
// the kernel's (is_grouped, StridingA) template args.
//
// Register budgets are per A-path (TMA vs indexed). Kernel selects via
// kIndexedGather. setmaxnreg (PTX):
//   producer: .dec → imm ≤ current max (release down to imm)
//   math:     .inc → imm ≥ current max (request up to imm)
//   imm ∈ [24,256], multiple of 8
// Pack: 2 math WGs → producer + 2*math = 504; 1 math WG → producer + math ≤ 512.

struct Sm90V3Tile_128x192 {
    static constexpr int TILE_M  = 128;
    static constexpr int TILE_N  = 192;
    static constexpr int TILE_K  = 128;
    static constexpr int WG_M    = 2;
    static constexpr int WG_N    = 1;
    static constexpr int Stages  = 4;
    static constexpr int kMaxOpN = 192;

    static constexpr int kProducerRegsTma     = 40;
    static constexpr int kMathRegsTma         = 232;  // (504-40)/2
    static constexpr int kProducerRegsIndexed = 88;
    static constexpr int kMathRegsIndexed     = 208;  // (504-88)/2
};

struct Sm90V3Tile_128x256 {
    static constexpr int TILE_M  = 128;
    static constexpr int TILE_N  = 256;
    static constexpr int TILE_K  = 128;
    static constexpr int WG_M    = 2;
    static constexpr int WG_N    = 1;
    static constexpr int Stages  = 3;  // SMEM: N=256 needs Stages≤3
    static constexpr int kMaxOpN = 128;

    static constexpr int kProducerRegsTma     = 40;
    static constexpr int kMathRegsTma         = 232;
    static constexpr int kProducerRegsIndexed = 88;
    static constexpr int kMathRegsIndexed     = 208;
};

// CTA 64×256: single math WG (WG_TILE = CTA tile). Stages=4 fits (smaller A/C).
struct Sm90V3Tile_64x256 {
    static constexpr int TILE_M  = 64;
    static constexpr int TILE_N  = 256;
    static constexpr int TILE_K  = 128;
    static constexpr int WG_M    = 1;
    static constexpr int WG_N    = 1;
    static constexpr int Stages  = 4;
    static constexpr int kMaxOpN = 128;

    // Math RF model (kMaxOpN=128 → atom 64×128, ITER_N=2):
    //   AccumC 128 + FragC 64 peak ≈ 192, +U/V/ctrl ≈ 208.
    // Producer .dec must be ≤ ptxas temporal (~192); 256 is illegal.
    static constexpr int kProducerRegsTma     = 40;
    static constexpr int kMathRegsTma         = 208;
    static constexpr int kProducerRegsIndexed = 88;
    static constexpr int kMathRegsIndexed     = 208;
};

}  // namespace turbomind::gemm
