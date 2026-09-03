#pragma once

#include <numeric>

#include "cute/algorithm/gemm.hpp"
#include "cute/arch/mma_sm90.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/gmma_bf16_sm90.h"
#include "src/turbomind/kernels/gemm/sm90_utils.h"

namespace turbomind::gemm {

// Act-as-A FP8 GMMA traits used by the V3 kernel.  The CuTe tile is local to
// one math warp-group; MmaN caps the selected WGMMA N while the remaining N
// atoms stay visible as the TiledMMA fragment's rest-N mode.
template<int TILE_M, int TILE_N, int TILE_K, class AtomLayoutMNK_, int MmaN_ = 0>
struct GmmaFP8V3Traits {
    using ElementA = cutlass::float_e4m3_t;  // activation
    using ElementB = cutlass::float_e4m3_t;  // weight
    using ElementC = float;

    using TileShape              = cute::Shape<cute::Int<TILE_M>, cute::Int<TILE_N>, cute::Int<TILE_K>>;
    static constexpr auto MajorA = cute::GMMA::Major::K;
    static constexpr auto MajorB = cute::GMMA::Major::K;

    using AtomLayoutMNK = AtomLayoutMNK_;

    static constexpr int kAtomM = cute::size<0>(AtomLayoutMNK{});
    static constexpr int kAtomN = cute::size<1>(AtomLayoutMNK{});
    static_assert(kAtomM * kAtomN >= 1);
    static_assert(TILE_M % (64 * kAtomM) == 0, "TILE_M vs AtomLayout M");
    static_assert(TILE_N % kAtomN == 0, "TILE_N vs AtomLayout N");
    static_assert(TILE_K % 32 == 0, "FP8 GMMA K divisibility");

    static constexpr int kMmaN = MmaN_ ? MmaN_ : TILE_N / kAtomN;
    static_assert(TILE_N % (kAtomN * kMmaN) == 0, "TILE_N vs WGMMA N");
    using WgTileShape =
        cute::Shape<cute::Int<TILE_M / kAtomM>, cute::Int<kMmaN>, cute::Int<TILE_K>>;

    using MmaAtom =
        decltype(cute::GMMA::ss_op_selector<ElementA, ElementB, ElementC, WgTileShape, MajorA, MajorB>());
    using TiledMma = decltype(cute::make_tiled_mma(MmaAtom{}, AtomLayoutMNK{}));

    static constexpr typename cute::MMA_Traits<MmaAtom>::Shape_MNK OpShape{};
    static constexpr int kOpM = cute::get<0>(OpShape);
    static constexpr int kOpN = cute::get<1>(OpShape);
    static constexpr int kOpK = cute::get<2>(OpShape);

    static constexpr int kRestM   = TILE_M / (kAtomM * kOpM);
    static constexpr int kRestN   = TILE_N / (kAtomN * kOpN);
    static constexpr int kKBlocks = TILE_K / kOpK;
    static_assert(kOpM == 64 && kOpK == 32);
    static_assert(kRestM >= 1 && kRestN >= 1 && kKBlocks >= 1);
    static_assert(cute::tile_size<0>(TiledMma{}) == kAtomM * kOpM);
    static_assert(cute::tile_size<1>(TiledMma{}) == kAtomN * kOpN);

    using SmemLayoutAtomA =
        decltype(gmma_ss_smem_selector<MajorA, ElementA, cute::Int<TILE_M>, cute::Int<TILE_K>>());
    using SmemLayoutAtomB =
        decltype(gmma_ss_smem_selector<MajorB, ElementB, cute::Int<TILE_N>, cute::Int<TILE_K>>());
};

/*
 * Weight-as-A (WA) blockscaled FP8 GMMA — scale TV contract
 * =========================================================
 *
 * GMMA CLayout_64xN (MMA.md): thread t = t0 + 4*t1 + 32*t2, value (v0,v1,v2):
 *   m_gmma = t1 + 16*t2 + 8*v1          // OUT / weight axis after WA
 *   n_gmma = 2*t0 + v0 + 8*v2           // BATCH / act axis after WA
 *
 * Per 8-col stripe (v2), CRegisters pack as float[4]:
 *   [0]=(m0,n0), [1]=(m0,n1), [2]=(m0+8,n0), [3]=(m0+8,n1)
 *
 * Problem scales (SMEM shapes unchanged vs v3):
 *   U[row]: act kK scale, dense along problem M (= GMMA-N / BATCH)
 *   V[0|1]: weight kB scale, sparse along problem N/128 (= GMMA-M / OUT)
 *
 * WA apply (invert of ScaledGmmaFP8_TN::scale_batch_to_accum):
 *   sw0 = weight_scale_for(m0)     // from V, 128-block along OUT
 *   sw1 = weight_scale_for(m0+8)   // == sw0 when OUT atom is 64-aligned
 *   sa0 = act_scale[n0]            // from U, dense along BATCH
 *   sa1 = act_scale[n1]
 *   accum[i] += sw * sa * frag[i]
 *
 * Load invert vs v3:
 *   v3 Load_U: dense along GMMA-M (act rows)
 *   v3 Load_V: sparse 2-wide along GMMA-N (weight 128-blocks)
 *   WA: sparse weight on GMMA-M; dense act on GMMA-N (per owned columns)
 *
 * RF: dense act scales grow with atom N (= BATCH). Prefer TILE_M 128→64→32.
 */

// CuTe traits: weight=GMMA-A (OUT×K), act=GMMA-B (BATCH×K).  The
// TiledMma, TMA producer, and GMMA descriptor consumer all share the same
// canonical SMEM layout atoms.
template<int TILE_OUT, int TILE_BATCH, int TILE_K, class AtomLayoutMNK_>
struct GmmaFP8WaTraits {
    using ElementA = cutlass::float_e4m3_t;  // GMMA-A = weight
    using ElementB = cutlass::float_e4m3_t;  // GMMA-B = activation
    using ElementC = float;

    using TileShape              = cute::Shape<cute::Int<TILE_OUT>, cute::Int<TILE_BATCH>, cute::Int<TILE_K>>;
    static constexpr auto MajorA = cute::GMMA::Major::K;
    static constexpr auto MajorB = cute::GMMA::Major::K;

    using AtomLayoutMNK = AtomLayoutMNK_;

    static constexpr int kAtomM = cute::size<0>(AtomLayoutMNK{});
    static constexpr int kAtomN = cute::size<1>(AtomLayoutMNK{});
    static_assert(kAtomM * kAtomN >= 1);
    static_assert(TILE_OUT % (64 * kAtomM) == 0, "TILE_OUT vs AtomLayout M");
    static_assert(TILE_BATCH % kAtomN == 0, "TILE_BATCH vs AtomLayout N");
    static_assert(TILE_K % 32 == 0, "FP8 GMMA K divisibility");

    // Atom N comes from the per-WG BATCH extent (not weight OUT) — the same
    // operand-axis reversal as GmmaBF16Traits.
    using WgTileShape =
        cute::Shape<cute::Int<TILE_OUT / kAtomM>, cute::Int<TILE_BATCH / kAtomN>, cute::Int<TILE_K>>;

    using MmaAtom =
        decltype(cute::GMMA::ss_op_selector<ElementA, ElementB, ElementC, WgTileShape, MajorA, MajorB>());
    using TiledMma = decltype(cute::make_tiled_mma(MmaAtom{}, AtomLayoutMNK{}));

    static constexpr typename cute::MMA_Traits<MmaAtom>::Shape_MNK OpShape{};
    static constexpr int kOpM = cute::get<0>(OpShape);
    static constexpr int kOpN = cute::get<1>(OpShape);
    static constexpr int kOpK = cute::get<2>(OpShape);

    static constexpr int kRestM = TILE_OUT / (kAtomM * kOpM);
    static constexpr int kRestN = TILE_BATCH / (kAtomN * kOpN);
    static constexpr int kKBlocks = TILE_K / kOpK;
    static_assert(kOpM == 64 && kOpK == 32);
    static_assert(kRestM >= 1 && kRestN >= 1 && kKBlocks >= 1);
    static_assert(cute::tile_size<0>(TiledMma{}) == kAtomM * kOpM);
    static_assert(cute::tile_size<1>(TiledMma{}) == kAtomN * kOpN);

    // Weight scales are 128-wide along GMMA-M (OUT).  Dense activation
    // scales follow the two GMMA-N columns owned by each thread per stripe.
    static constexpr int kOuterM = std::gcd(TILE_OUT / kAtomM, 128);
    static constexpr int kActScalesPerThread = kOpN / 4;

    using SmemLayoutAtomA = decltype(gmma_ss_smem_selector<MajorA, ElementA, cute::Int<TILE_OUT>, cute::Int<TILE_K>>());
    using SmemLayoutAtomB =
        decltype(gmma_ss_smem_selector<MajorB, ElementB, cute::Int<TILE_BATCH>, cute::Int<TILE_K>>());
};

}  // namespace turbomind::gemm
