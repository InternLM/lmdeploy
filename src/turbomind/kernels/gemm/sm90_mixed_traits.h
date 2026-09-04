// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <type_traits>

#include "cute/arch/mma_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/layout.hpp"

#include "src/turbomind/kernels/gemm/gmma_bf16_sm90.h"

namespace turbomind::gemm {

// The physical pack is deliberately independent of the production tile
// catalog.  The converter always writes the two contiguous 64-output
// warpgroup fragments of this tile-order 1x2, N256 reference MMA.
struct GmmaMixedPackTraits {
    static constexpr int TILE_OUT   = 128;
    static constexpr int TILE_BATCH = 256;
    static constexpr int TILE_K     = 64;

    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementC = float;

    using WGLayout      = WG_1x2;
    using AtomLayoutMNK = GmmaAtomLayoutMNK<WGLayout>;
    using MmaAtom       = cute::SM90_64x256x16_F32BF16BF16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
    using TiledMma      = decltype(cute::make_tiled_mma(MmaAtom{}, AtomLayoutMNK{}));

    static constexpr int kKBlocksPerStage  = TILE_K / 16;
    static constexpr int kPackedWordsStage = 2 * kKBlocksPerStage * 128;

    static_assert(cute::size(TiledMma{}) == 256);
    static_assert(kPackedWordsStage == 1024);
};

// Production GEMM traits. The public tile is BATCH x OUT x K64 while the
// hardware tile is OUT x BATCH x K64 (packed weight is RS operand A). The
// persistent weight ABI is [K/16][OUT/64][RS fragment].
template<int Out, int Batch, int Stages_, class WGLayout_, int MmaN_ = 0, class Element_ = cutlass::bfloat16_t>
struct GmmaMixedTraits {
    static constexpr int TILE_OUT   = Out;
    static constexpr int TILE_BATCH = Batch;
    static constexpr int TILE_K     = 64;
    static constexpr int Stages     = Stages_;

    using ElementA = Element_;
    using ElementB = Element_;
    using ElementC = float;

    using WGLayout      = WGLayout_;
    using AtomLayoutMNK = GmmaAtomLayoutMNK<WGLayout>;
    using TileShape     = cute::Shape<cute::Int<TILE_OUT>, cute::Int<TILE_BATCH>, cute::Int<TILE_K>>;

    static constexpr auto MajorA = cute::GMMA::Major::K;
    static constexpr auto MajorB = cute::GMMA::Major::K;

    static constexpr int kAtomM = cute::size<0>(AtomLayoutMNK{});
    static constexpr int kAtomN = cute::size<1>(AtomLayoutMNK{});
    static_assert(kAtomM * kAtomN >= 1 && kAtomM * kAtomN <= 3);
    static_assert(TILE_OUT % (64 * kAtomM) == 0);
    static_assert(TILE_BATCH % kAtomN == 0);
    static_assert(TILE_BATCH / kAtomN >= 8);

    static constexpr int kMmaN = MmaN_ ? MmaN_ : TILE_BATCH / kAtomN;
    static_assert(TILE_BATCH % (kAtomN * kMmaN) == 0);
    static constexpr int kMmaNSlices = TILE_BATCH / (kAtomN * kMmaN);
    using WgTileShape = cute::Shape<cute::Int<TILE_OUT / kAtomM>, cute::Int<kMmaN>, cute::Int<TILE_K>>;
    using MmaAtom  = decltype(cute::GMMA::rs_op_selector<ElementA, ElementB, ElementC, WgTileShape, MajorA, MajorB>());
    using TiledMma = decltype(cute::make_tiled_mma(MmaAtom{}, AtomLayoutMNK{}));
    using WgTiledMma = decltype(cute::make_tiled_mma(MmaAtom{}, cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>{}));

    // Candidate atoms must consume the exact same per-thread operand-A
    // fragment as the immutable pack reference. WGLayout only changes
    // whether contiguous ranges of pack segments are assigned to separate WGs
    // (tile 1x2) or traversed as RestM by each batch-split WG (tile 2x1).
    static_assert(
        std::is_same_v<typename TiledMma::AtomLayoutA_TV, typename GmmaMixedPackTraits::TiledMma::AtomLayoutA_TV>);
    static_assert(cute::tile_size<1>(TiledMma{}) == kAtomN * kMmaN);

    using DummySmemLayoutA =
        decltype(cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<ElementA>{},
                                     cute::make_shape(cute::Int<TILE_OUT>{}, cute::Int<TILE_K>{}, cute::_1{}),
                                     cute::Step<cute::_1, cute::_2, cute::_3>{}));

    using SmemLayoutAtomB =
        decltype(gmma_ss_smem_selector<MajorB, ElementB, cute::Int<TILE_BATCH>, cute::Int<TILE_K>>());
    using SmemLayoutB = decltype(
        cute::tile_to_shape(SmemLayoutAtomB{},
                            cute::make_shape(cute::Int<TILE_BATCH>{}, cute::Int<TILE_K>{}, cute::Int<Stages>{}),
                            cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutB_2D = decltype(cute::tile_to_shape(SmemLayoutAtomB{},
                                                        cute::make_shape(cute::Int<TILE_BATCH>{}, cute::Int<TILE_K>{}),
                                                        cute::Step<cute::_1, cute::_2>{}));

    static constexpr int kMathWarpgroups   = cute::size(AtomLayoutMNK{});
    static constexpr int kMathThreads      = 128 * kMathWarpgroups;
    static constexpr int kProducerThreads  = 128;
    static constexpr int kCtaThreads       = kMathThreads + kProducerThreads;
    static constexpr int kKBlocksPerStage  = TILE_K / 16;
    static constexpr int kPackedWordsStage = TILE_OUT * TILE_K / 8;
    static constexpr int kPackedBytesStage = kPackedWordsStage * sizeof(uint32_t);
    static constexpr int kQparamWordsStage = TILE_OUT;
    static constexpr int kQparamBytesStage = kQparamWordsStage * sizeof(uint32_t);

    static_assert(kMathWarpgroups >= 1 && kMathWarpgroups <= 3);
    static_assert(kPackedWordsStage == TILE_OUT * TILE_K / 8);
    static_assert(kPackedBytesStage == TILE_OUT * TILE_K / 2);
    static_assert(kQparamBytesStage == TILE_OUT * 4);
};

}  // namespace turbomind::gemm
