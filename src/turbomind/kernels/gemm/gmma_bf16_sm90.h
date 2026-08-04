#pragma once

#include "cute/arch/mma_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"

namespace turbomind::gemm {

// Identical to cutlass::gemm::collective::detail::ss_smem_selector
// (sm90_common.inl:273-319). Including that .inl from TurboMind headers is
// awkward (it is meant to be pulled inside cutlass::gemm::collective::detail).
// Compose the same GMMA Layout_*_Atom SoT here — MMA.md §6 / SHARED_MEMORY.md.
template<cute::GMMA::Major major, class ElementType, class BLK_MN, class BLK_K>
CUTE_HOST_DEVICE constexpr auto gmma_ss_smem_selector()
{
    using namespace cute;

    auto BLK_MN0 = size<0>(BLK_MN{});
    auto BLK_K0  = size<0>(BLK_K{});

    static_assert(BLK_MN0 % 8 == 0, "BLK_MN0 must be a multiple of 8.");
    static_assert(BLK_K0 % 8 == 0, "BLK_K0 must be a multiple of 8.");

    if constexpr (major == GMMA::Major::MN) {
        if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW128_Atom<ElementType>{}) == 0) {
            return GMMA::Layout_MN_SW128_Atom<ElementType>{};
        }
        else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW64_Atom<ElementType>{}) == 0) {
            return GMMA::Layout_MN_SW64_Atom<ElementType>{};
        }
        else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW32_Atom<ElementType>{}) == 0) {
            return GMMA::Layout_MN_SW32_Atom<ElementType>{};
        }
        else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{}) == 0) {
            return GMMA::Layout_MN_INTER_Atom<ElementType>{};
        }
        else {
            static_assert(BLK_MN0 % size<0>(GMMA::Layout_MN_INTER_Atom<ElementType>{}) == 0,
                          "BLK_MN0 must be a multiple of size<0>(GMMA::Layout_MN_INTER_Atom)");
        }
    }
    else if constexpr (major == GMMA::Major::K) {
        // Prefer largest legal Swizzle<B,4,3>: SW128 → SW64 → SW32 → INTER.
        if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW128_Atom<ElementType>{}) == 0) {
            return GMMA::Layout_K_SW128_Atom<ElementType>{};
        }
        else if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW64_Atom<ElementType>{}) == 0) {
            return GMMA::Layout_K_SW64_Atom<ElementType>{};
        }
        else if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_SW32_Atom<ElementType>{}) == 0) {
            return GMMA::Layout_K_SW32_Atom<ElementType>{};
        }
        else if constexpr (BLK_K0 % size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{}) == 0) {
            return GMMA::Layout_K_INTER_Atom<ElementType>{};
        }
        else {
            static_assert(BLK_K0 % size<1>(GMMA::Layout_K_INTER_Atom<ElementType>{}) == 0,
                          "BLK_K0 must be a multiple of size<1>(GMMA::Layout_K_INTER_Atom)");
        }
    }
}

// TILE_OUT  = GMMA-M extent = problem N_out (weight output)  → GMMA-A = weight
// TILE_BATCH = GMMA-N extent = problem M_batch (tokens)       → GMMA-B = activation
// LlamaLinear API A/B stay act/weight; MMA operands are swapped relative to that API.
// AtomLayoutMNK comes from the Tile_ (CuTe Layout<Shape<...>>).
template<int TILE_OUT, int TILE_BATCH, int TILE_K, class AtomLayoutMNK_>
struct GmmaBF16Traits {
    using ElementA = cutlass::bfloat16_t;  // GMMA-A = weight
    using ElementB = cutlass::bfloat16_t;  // GMMA-B = activation
    using ElementC = float;

    using TileShape              = cute::Shape<cute::Int<TILE_OUT>, cute::Int<TILE_BATCH>, cute::Int<TILE_K>>;
    static constexpr auto MajorA = cute::GMMA::Major::K;
    static constexpr auto MajorB = cute::GMMA::Major::K;

    using AtomLayoutMNK = AtomLayoutMNK_;

    static constexpr int kAtomM = cute::size<0>(AtomLayoutMNK{});
    static constexpr int kAtomN = cute::size<1>(AtomLayoutMNK{});
    static_assert(kAtomM * kAtomN >= 1);
    // Layout<_2,_1,_1>: GMMA-M (OUT) must be multiple of 128 (2×64 atom).
    // Layout<_1,_2,_1>: GMMA-M (OUT) multiple of 64; BATCH split across 2 WGs.
    static_assert(TILE_OUT % (64 * kAtomM) == 0, "TILE_OUT vs AtomLayout M");
    static_assert(TILE_BATCH % kAtomN == 0, "TILE_BATCH vs AtomLayout N");
    static_assert(TILE_K % 8 == 0, "GMMA K divisibility (MMA.md)");

    // ss_op_selector picks atom N from Tile_N only — it does NOT see AtomLayout.
    // AtomLayout<_1,_2,_1> splits BATCH across WGs, so each WG's N is
    // TILE_BATCH/kAtomN. Select the op from that per-WG tile; otherwise
    // MMA_64x128 on TILE_BATCH=128 with 2 WGs along N OOBs SMEM (atom N=128
    // but each WG owns only 64). Same for M with AtomLayout<_2,_1,_1>.
    using WgTileShape = cute::Shape<cute::Int<TILE_OUT / kAtomM>, cute::Int<TILE_BATCH / kAtomN>, cute::Int<TILE_K>>;

    using TiledMma = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<ElementA, ElementB, ElementC, WgTileShape, MajorA, MajorB>(), AtomLayoutMNK{}));

    // One SoT SMEM atom for TMA + GMMA (MMA.md §6). OUT×K and BATCH×K, K-major.
    using SmemLayoutAtomA = decltype(gmma_ss_smem_selector<MajorA, ElementA, cute::Int<TILE_OUT>, cute::Int<TILE_K>>());
    using SmemLayoutAtomB =
        decltype(gmma_ss_smem_selector<MajorB, ElementB, cute::Int<TILE_BATCH>, cute::Int<TILE_K>>());
};

}  // namespace turbomind::gemm
