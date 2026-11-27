// Copyright (c) OpenMMLab. All rights reserved.
//
// SM75 (Turing) verification-attention policy.
//
// Reuses the SM80-family mma.sync kernels (`SM80_16x8x16`, valid on SM75) and
// the SM75 ldmatrix atoms, but replaces the two arch-specific primitives the
// generic mainloop uses on SM90:
//   * KV tile copy  : `cp.async` (SM80+)  -> synchronous `UniversalCopy<uint128_t>`
//   * probability   : `stmatrix` (SM90)    -> synchronous `UniversalCopy<uint32_t>`
// `AsyncKV = false` makes the shared mainloop skip the cp.async pipeline (the
// software double-buffer, Stages=2, still hides smem latency).
//
// Nothing here is used by any other architecture.
#pragma once

#include <cute/tensor.hpp>

#include "src/turbomind/kernels/attention/verification/policy_sm90.cuh"
#include "src/turbomind/kernels/core/mma.h"

namespace turbomind::verification_attention {

// m16n8k16 f16 is an SM80 instruction; Turing (SM75) only has m16n8k8. Reuse
// TurboMind's arch-guarded `mma_m16n8k16_row_col`, which emulates m16n8k16 with
// two m16n8k8 on SM75. Register traits mirror `SM80_16x8x16_F32F16F16F32_TN`
// so the cute tiling / ldmatrix atoms are reused unchanged.
struct TM_16x8x16_F32F16F16F32_TN {
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];
    using BRegisters = uint32_t[2];
    using CRegisters = float[4];

    CUTE_DEVICE static void fma(float&          d0,
                                float&          d1,
                                float&          d2,
                                float&          d3,
                                uint32_t const& a0,
                                uint32_t const& a1,
                                uint32_t const& a2,
                                uint32_t const& a3,
                                uint32_t const& b0,
                                uint32_t const& b1,
                                float const&    c0,
                                float const&    c1,
                                float const&    c2,
                                float const&    c3)
    {
        Array<half, 8>  A;
        Array<half, 4>  B;
        Array<float, 4> D;
        Array<float, 4> C;
        reinterpret_cast<uint32_t*>(&A[0])[0] = a0;
        reinterpret_cast<uint32_t*>(&A[0])[1] = a1;
        reinterpret_cast<uint32_t*>(&A[0])[2] = a2;
        reinterpret_cast<uint32_t*>(&A[0])[3] = a3;
        reinterpret_cast<uint32_t*>(&B[0])[0] = b0;
        reinterpret_cast<uint32_t*>(&B[0])[1] = b1;
        C[0] = c0, C[1] = c1, C[2] = c2, C[3] = c3;
        mma_m16n8k16_row_col(D, A, B, C);
        d0 = D[0], d1 = D[1], d2 = D[2], d3 = D[3];
    }
};

template<int HeadDim_, int MTile_>
struct Sm75Policy: Sm90PolicyBase<HeadDim_, MTile_> {
    using Base = Sm90PolicyBase<HeadDim_, MTile_>;

    static constexpr int HeadDim = Base::HeadDim;
    static constexpr int MTile   = Base::MTile;
    static constexpr int PvNtile = Base::PvNtile;
    // Single-stage sync KV (no cp.async): K+V+prob = 32*HeadDim*2 + MTile*32*2
    //   HeadDim 128: 16 KB, HeadDim 256: 32 KB  (<= 64 KB with the union'd Q).
    static constexpr int KeyTile     = 32;
    static constexpr int PvTileCount = HeadDim / PvNtile;

    static constexpr int  Stages  = 1;
    static constexpr bool AsyncKV = false;

    // SM75 has ldmatrix + m16n8k16 but no cp.async / stmatrix.
    template<class T>
    using KvCopyAtom = cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, T>;

    template<class T>
    using StoreAtom = cute::Copy_Atom<cute::UniversalCopy<cute::uint32_t>, T>;

    // SM75 has no f16 m16n8k16; use the emulated (two m16n8k8) atom.
    template<class T>
    using MmaOp = TM_16x8x16_F32F16F16F32_TN;

    using KvShape    = cute::Shape<cute::Int<KeyTile>, cute::Int<HeadDim>>;
    using ScoreShape = cute::Shape<cute::Int<MTile>, cute::Int<KeyTile>>;
    using PvShape    = cute::Shape<cute::Int<MTile>, cute::Int<PvNtile>>;
};

}  // namespace turbomind::verification_attention

namespace cute {

// The emulated atom shares the SM80 m16n8k16 fragment layouts.
template<>
struct MMA_Traits<turbomind::verification_attention::TM_16x8x16_F32F16F16F32_TN>:
    MMA_Traits<SM80_16x8x16_F32F16F16F32_TN> {
};

}  // namespace cute
