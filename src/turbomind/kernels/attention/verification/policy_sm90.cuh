// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <type_traits>

#include <cute/tensor.hpp>

#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

namespace turbomind::verification_attention {

template<int HeadDim_, int MTile_>
struct Sm90PolicyBase {
    using MmaTraits = cute::MMA_Traits<cute::SM80_16x8x16_F32F16F16F32_TN>;

    static constexpr int HeadDim          = HeadDim_;
    static constexpr int MTile            = MTile_;
    static constexpr int MinThreads       = 128;
    static constexpr int MmaM             = cute::size<0>(typename MmaTraits::Shape_MNK{});
    static constexpr int MmaThreads       = cute::size(typename MmaTraits::ThrID{});
    static constexpr int MmaAtomM         = MTile / MmaM;
    static constexpr int MmaMThreads      = MmaThreads * MmaAtomM;
    static constexpr int MmaAtomN         = (MinThreads + MmaMThreads - 1) / MmaMThreads;
    static constexpr int Threads          = MmaMThreads * MmaAtomN;
    static constexpr int Stages           = 3;
    static constexpr int PvNtile          = 32;

    static_assert(MTile % MmaM == 0);

    using QShape      = cute::Shape<cute::Int<MTile>, cute::Int<HeadDim>>;
    using OutputShape = QShape;
};

template<int HeadDim_, int MTile_>
struct Sm90Policy: Sm90PolicyBase<HeadDim_, MTile_> {
    using Base = Sm90PolicyBase<HeadDim_, MTile_>;
    static constexpr int HeadDim     = Base::HeadDim;
    static constexpr int MTile       = Base::MTile;
    static constexpr int PvNtile     = Base::PvNtile;
    static constexpr int KeyTile     = HeadDim == 128 ? 64 : 32;
    static constexpr int PvTileCount = HeadDim / PvNtile;

    using KvShape    = cute::Shape<cute::Int<KeyTile>, cute::Int<HeadDim>>;
    using ScoreShape = cute::Shape<cute::Int<MTile>, cute::Int<KeyTile>>;
    using PvShape    = cute::Shape<cute::Int<MTile>, cute::Int<PvNtile>>;
};

template<class Atom_, int Rows, int Columns, class Policy>
struct VectorTileCopy {
    using Atom = Atom_;
    using T    = typename Atom::ValType;

    static constexpr int ValuesPerAccess   = 16 / sizeof(T);
    static constexpr int VectorsPerRow     = Columns / ValuesPerAccess;
    static constexpr int RowsPerThreadTile = Policy::Threads / VectorsPerRow;
    static constexpr int AccessCount       = Rows / RowsPerThreadTile;

    using ThreadLayout = cute::Layout<
        cute::Shape<cute::Int<RowsPerThreadTile>, cute::Int<VectorsPerRow>>,
        cute::Stride<cute::Int<VectorsPerRow>, cute::_1>>;
    using ValueLayout = cute::Layout<cute::Shape<cute::_1, cute::Int<ValuesPerAccess>>>;
    using TiledCopy = decltype(cute::make_tiled_copy(Atom{}, ThreadLayout{}, ValueLayout{}));

    static_assert(Columns % ValuesPerAccess == 0);
    static_assert(Policy::Threads % VectorsPerRow == 0);
    static_assert(Rows % RowsPerThreadTile == 0);
    static_assert(cute::size(ThreadLayout{}) == Policy::Threads);
};

template<class T, class Policy>
using QTileCopy = VectorTileCopy<cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, T>,
                                 Policy::MTile,
                                 Policy::HeadDim,
                                 Policy>;

template<class T, class Policy>
using KvTileCopy = VectorTileCopy<
    cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, T>,
    Policy::KeyTile,
    Policy::HeadDim,
    Policy>;

template<class T, class Policy>
using OutputTileCopy = QTileCopy<T, Policy>;

template<class T>
struct MmaOperation;

template<>
struct MmaOperation<cutlass::half_t> {
    using Type = cute::SM80_16x8x16_F32F16F16F32_TN;
};

template<>
struct MmaOperation<cutlass::bfloat16_t> {
    using Type = cute::SM80_16x8x16_F32BF16BF16F32_TN;
};

template<class T, class Policy>
struct Sm90Mma {
    using AtomLayout = cute::Layout<
        cute::Shape<cute::Int<Policy::MmaAtomM>, cute::Int<Policy::MmaAtomN>, cute::_1>>;
    using MmaOp      = typename MmaOperation<T>::Type;

    using QK = decltype(cute::make_tiled_mma(
        MmaOp{},
        AtomLayout{},
        cute::Tile<cute::Int<Policy::MTile>, cute::Int<Policy::KeyTile>, cute::_16>{}));
    using PV = decltype(cute::make_tiled_mma(
        MmaOp{},
        AtomLayout{},
        cute::Tile<cute::Int<Policy::MTile>, cute::Int<Policy::PvNtile>, cute::_16>{}));

    using CopyQkA = decltype(cute::make_tiled_copy_A(
        cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, T>{}, QK{}));
    using CopyQkB = decltype(cute::make_tiled_copy_B(
        cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, T>{}, QK{}));
    using CopyPvA = decltype(cute::make_tiled_copy_A(
        cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, T>{}, PV{}));
    using CopyPvBAtom = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, T>;
    using CopyPvB = decltype(cute::make_tiled_copy_B(CopyPvBAtom{}, PV{}));
    using StoreProbability = decltype(cute::make_tiled_copy_C(
        cute::Copy_Atom<cute::SM90_U32x4_STSM_N, T>{}, QK{}));
};

using SmemAtom = decltype(cute::composition(
    cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_8, cute::Shape<cute::_8, cute::_8>>,
                 cute::Stride<cute::_8, cute::Stride<cute::_1, cute::_64>>>{}));

using SmemAtomNarrow = decltype(cute::composition(
    cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_8, cute::Shape<cute::_8, cute::_4>>,
                 cute::Stride<cute::_8, cute::Stride<cute::_1, cute::_64>>>{}));

template<int Columns>
using SmemAtomFor = std::conditional_t<Columns % 64 == 0, SmemAtom, SmemAtomNarrow>;

template<int Rows, int Columns>
using SmemLayout2D = decltype(cute::tile_to_shape(
    SmemAtomFor<Columns>{}, cute::Shape<cute::Int<Rows>, cute::Int<Columns>>{}));

template<int Rows, int Columns, int Stages>
using SmemLayout3D = decltype(cute::tile_to_shape(
    SmemAtom{}, cute::Shape<cute::Int<Rows>, cute::Int<Columns>, cute::Int<Stages>>{}));

template<class T, class Policy>
union SharedStorage {
    alignas(16) T q[Policy::MTile * Policy::HeadDim];
    struct {
        alignas(16) T k[Policy::Stages * Policy::KeyTile * Policy::HeadDim];
        alignas(16) T v[Policy::Stages * Policy::KeyTile * Policy::HeadDim];
        alignas(16) T probability[Policy::MTile * Policy::KeyTile];
    } body;
};

}  // namespace turbomind::verification_attention
