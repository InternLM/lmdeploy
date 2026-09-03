// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <type_traits>

#include "cute/arch/mma_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/layout.hpp"

#include "src/turbomind/kernels/gemm/gmma_bf16_sm90.h"

namespace turbomind::gemm {

template<int Out, int Batch, int Stages_, class WGLayout_, int MmaN_>
struct GmmaMxFp4Fp8TraitsBase {
    static constexpr int TILE_OUT = Out;
    static constexpr int TILE_BATCH = Batch;
    static constexpr int TILE_K = 128;
    static constexpr int Stages = Stages_;
    static constexpr int kMmaN = MmaN_;

    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = float;
    using WGLayout = WGLayout_;
    using AtomLayoutMNK = GmmaAtomLayoutMNK<WGLayout>;
    static constexpr int kAtomM = cute::size<0>(AtomLayoutMNK{});
    static constexpr int kAtomN = cute::size<1>(AtomLayoutMNK{});
    static constexpr int kMathWarpgroups = cute::size(AtomLayoutMNK{});
    static_assert(1 <= kMathWarpgroups && kMathWarpgroups <= 3);
    // TILE_BATCH is the public GEMM M tile.  Splitting that dimension across
    // two math warpgroups is reserved for tiles larger than M128; M64/M128
    // two-WG kernels split the output/N dimension with WG_1x2 instead.
    static_assert(!std::is_same_v<WGLayout, WG_2x1> || TILE_BATCH > 128);
    static_assert(TILE_OUT % (64 * kAtomM) == 0);
    static_assert(TILE_BATCH % (kAtomN * kMmaN) == 0);

    using WgTileShape = cute::Shape<cute::Int<TILE_OUT / kAtomM>, cute::Int<kMmaN>, cute::_32>;
    using MmaAtom = decltype(cute::GMMA::rs_op_selector<ElementA,
                                                        ElementB,
                                                        ElementC,
                                                        WgTileShape,
                                                        cute::GMMA::Major::K,
                                                        cute::GMMA::Major::K>());
    using WgAtomLayoutMNK = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
    using WgTiledMma = decltype(cute::make_tiled_mma(MmaAtom{}, WgAtomLayoutMNK{}));
    using TiledMma = decltype(cute::make_tiled_mma(MmaAtom{}, AtomLayoutMNK{}));
    using TileShape = cute::Shape<cute::Int<TILE_OUT>, cute::Int<TILE_BATCH>, cute::Int<TILE_K>>;

    CUTE_HOST_DEVICE static constexpr auto packed_layout_a_mk()
    {
        WgTiledMma mma;
        auto layout_a_tv = mma.get_layoutA_TV();
        auto layout_a_mk = cute::get<0>(mma.get_layoutA_MK());
        auto value_shape = cute::shape<1>(layout_a_tv);
        auto mma_value_shape = cute::get<0>(value_shape);
        static_assert(cute::size<0>(decltype(mma_value_shape){}) == 4);
        static_assert(cute::size<1>(decltype(mma_value_shape){}) == 2);
        static_assert(cute::size<2>(decltype(mma_value_shape){}) == 2);

        // Native FP8 A ownership is value=(k4,row2,k16). Map it to two E2M1
        // words without changing thread ownership: row2 selects the nibble
        // within a byte, k4 selects the byte, and k16 selects the u32.
        auto packed_value = cute::make_layout(
            value_shape,
            cute::make_stride(cute::make_stride(cute::_2{}, cute::_1{}, cute::_8{}),
                              cute::make_stride(cute::_0{}, cute::_0{})));
        auto packed_value_linear = cute::composition(
            packed_value, cute::right_inverse(cute::make_layout(value_shape)));
        auto packed_tv = cute::make_layout(
            cute::make_layout(cute::size<0>(layout_a_tv), cute::size<1>(layout_a_tv)),
            packed_value_linear);
        return cute::composition(packed_tv, layout_a_mk);
    }

    // Thread-major native E4M3 RS image used by the one-time folded pack.
    CUTE_HOST_DEVICE static constexpr auto folded_layout_a_mk()
    {
        WgTiledMma mma;
        auto layout_a_tv = mma.get_layoutA_TV();
        auto layout_a_mk = cute::get<0>(mma.get_layoutA_MK());
        auto thread_major_tv = cute::make_layout(
            cute::make_shape(cute::size<0>(layout_a_tv), cute::size<1>(layout_a_tv)),
            cute::make_stride(cute::size<1>(layout_a_tv), cute::_1{}));
        return cute::composition(thread_major_tv, layout_a_mk);
    }

    static constexpr typename cute::MMA_Traits<MmaAtom>::Shape_MNK OpShape{};
    static constexpr int kOpM = cute::get<0>(OpShape);
    static constexpr int kOpN = cute::get<1>(OpShape);
    static constexpr int kOpK = cute::get<2>(OpShape);
    static constexpr int kRestM = TILE_OUT / (kAtomM * kOpM);
    static constexpr int kRestN = TILE_BATCH / (kAtomN * kOpN);
    static constexpr int kKBlocksPerStage = TILE_K / kOpK;
    static_assert(kOpM == 64 && kOpK == 32);
    static_assert(kRestM >= 1 && kRestM <= 4);
    static_assert(kRestN >= 1);
    static_assert(kKBlocksPerStage == 4);
    static_assert(cute::size(TiledMma{}) == kMathWarpgroups * 128);
    static_assert(cute::size(WgTiledMma{}) == 128);
    // Keep the inexpensive structural checks in every instantiation.  The
    // full 128-thread oracle remains available for host-side validation, but
    // repeating it here exceeds NVCC's default constexpr operation budget.
    static_assert(cute::size<1>(typename cute::MMA_Traits<MmaAtom>::ALayout{}) == 16);
    static_assert(cute::size<1>(typename cute::MMA_Traits<MmaAtom>::CLayout{}) == kOpN / 2);

    // RS WGMMA still needs a tensor carrying the complete A-fragment layout.
    // As in the mixed-input kernel, derive that register layout by
    // partitioning a canonical GMMA shared-memory tensor; the tensor is never
    // read as an SS operand.
    using DummySmemLayoutA = decltype(cute::tile_to_shape(
        cute::GMMA::Layout_K_SW128_Atom<ElementA>{},
        cute::make_shape(cute::Int<TILE_OUT>{}, cute::Int<TILE_K>{}, cute::_1{}),
        cute::Step<cute::_1, cute::_2, cute::_3>{}));

    using SmemLayoutAtomB = decltype(gmma_ss_smem_selector<cute::GMMA::Major::K,
                                                           ElementB,
                                                           cute::Int<TILE_BATCH>,
                                                           cute::Int<TILE_K>>());
    using SmemLayoutB = decltype(cute::tile_to_shape(
        SmemLayoutAtomB{},
        cute::make_shape(cute::Int<TILE_BATCH>{}, cute::Int<TILE_K>{}, cute::Int<Stages>{}),
        cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutB_2D = decltype(cute::tile_to_shape(
        SmemLayoutAtomB{},
        cute::make_shape(cute::Int<TILE_BATCH>{}, cute::Int<TILE_K>{}),
        cute::Step<cute::_1, cute::_2>{}));

    static constexpr int kCRegsPerAtom = kOpM * kOpN / 128;
    static constexpr int kActScalesPerAtom = kOpN / 4;
    static_assert(2 * kActScalesPerAtom == kCRegsPerAtom);
};

template<int Out, int Batch, int Stages, class WGLayout, int MmaN>
struct GmmaMxFp4Fp8FoldedTraits: GmmaMxFp4Fp8TraitsBase<Out, Batch, Stages, WGLayout, MmaN> {
    using Base = GmmaMxFp4Fp8TraitsBase<Out, Batch, Stages, WGLayout, MmaN>;

    // One native E4M3 RS fragment is loaded directly from shared memory and
    // reused after each committed batch.
    static constexpr int kARegs = 16 * Base::kRestM;
    static constexpr int kAccumRegs = Base::kCRegsPerAtom * Base::kRestM * Base::kRestN;
    static constexpr int kScratchRegs = Base::kCRegsPerAtom;
    static constexpr int kQparamRegs = Base::kRestM;
    // The uniform schedule materializes activation scales only after wait<0>,
    // when the MMA issue operands are dead.  Guard the simultaneously-live
    // accumulator pair; ptxas's zero-stack/zero-spill report is the final
    // acceptance criterion for the remaining short-lived state.
    static_assert(kAccumRegs + kScratchRegs <= 224);
};

template<int Out, int Batch, int Stages, class WGLayout, int MmaN>
struct GmmaMxFp4Fp8UnfoldedTraits: GmmaMxFp4Fp8TraitsBase<Out, Batch, Stages, WGLayout, MmaN> {
    using Base = GmmaMxFp4Fp8TraitsBase<Out, Batch, Stages, WGLayout, MmaN>;

    // The full residual-output RS fragment mirrors the mixed GEMM. Each M64
    // atom carries sixteen u32 registers across the four K32 groups; the
    // packed source adds two transient u32 registers.
    static constexpr int kARegs = 2 + 16 * Base::kRestM;
    static constexpr int kAccumRegs = Base::kCRegsPerAtom * Base::kRestM * Base::kRestN;
    static constexpr int kScratchRegs = Base::kCRegsPerAtom;
    static constexpr int kQparamRegs = 2 * Base::kKBlocksPerStage * Base::kRestM;
    static constexpr int kActivationScaleRegs =
        Base::kRestN > 1 ? 2 : Base::kActScalesPerAtom * Base::kRestN;
    static constexpr int kExplicitRegs =
        kARegs + kAccumRegs + kScratchRegs + kQparamRegs + kActivationScaleRegs + 2;
    // The exact schedule overlaps scale fragments with dead MMA operands.
    // Keep this conservative model below Hopper's per-thread architectural
    // ceiling; ptxas's zero-stack/zero-spill report remains the acceptance
    // criterion for every registered allocation.
    static_assert(kExplicitRegs <= 255);
};

namespace detail {

template<int Batch,
         int Out,
         int Stages,
         class WGLayout,
         int MmaN,
         int ProducerRegsTma,
         int MathRegsTma,
         int ProducerRegsIndexed,
         int MathRegsIndexed>
struct MxFp4Fp8TileBase {
    static constexpr int TILE_BATCH = Batch;
    static constexpr int TILE_OUT = Out;
    static constexpr int Stages_ = Stages;
    using WGLayout_ = WGLayout;
    static constexpr int kMmaN = MmaN;
    static constexpr int kProducerRegsTma = ProducerRegsTma;
    static constexpr int kMathRegsTma = MathRegsTma;
    static constexpr int kProducerRegsIndexed = ProducerRegsIndexed;
    static constexpr int kMathRegsIndexed = MathRegsIndexed;
};

}  // namespace detail

using MxFp4Fp8Tile_64x128 =
    detail::MxFp4Fp8TileBase<64, 128, 3, WG_1x2, 64, 40, 232, 72, 216>;

}  // namespace turbomind::gemm
