#pragma once

#include <numeric>

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

// CuTe traits: weight=GMMA-A (OUT×K), act=GMMA-B (BATCH×K). SMEM atoms via gmma_ss_smem_selector.
template<int TILE_OUT, int TILE_BATCH, int TILE_K, class AtomLayoutMNK_>
struct ScaledGmmaFP8Traits {
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

    // Atom N from per-WG BATCH (not weight OUT) — same pitfall as GmmaBF16Traits.
    using WgTileShape = cute::Shape<cute::Int<TILE_OUT / kAtomM>, cute::Int<TILE_BATCH / kAtomN>, cute::Int<TILE_K>>;

    using TiledMma = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<ElementA, ElementB, ElementC, WgTileShape, MajorA, MajorB>(), AtomLayoutMNK{}));

    using SmemLayoutAtomA = decltype(gmma_ss_smem_selector<MajorA, ElementA, cute::Int<TILE_OUT>, cute::Int<TILE_K>>());
    using SmemLayoutAtomB =
        decltype(gmma_ss_smem_selector<MajorB, ElementB, cute::Int<TILE_BATCH>, cute::Int<TILE_K>>());
};

// Hand-rolled SS-TN WGMMA + WA scale-to-accum (CRegisters-aware).
// TILE_OUT → GMMA-M (weight); TILE_BATCH → GMMA-N (act). MAX_OP_N caps atom N (BATCH).
template<int TILE_OUT, int TILE_BATCH, int TILE_K, int BATCH_M, int BATCH_N, int PIPE_M, int PIPE_N, int MAX_OP_N = 256>
struct ScaledGmmaFP8_WA {

    static constexpr auto select_gmma_operation()
    {
        static_assert(TILE_OUT % (BATCH_M * PIPE_M) == 0);
        static_assert(TILE_BATCH % (BATCH_N * PIPE_N) == 0);

        constexpr int M = TILE_OUT / (BATCH_M * PIPE_M);
        constexpr int N = TILE_BATCH / (BATCH_N * PIPE_N);

        static_assert(M % 64 == 0);

        using namespace cute::SM90::GMMA;

        if constexpr (N % 256 == 0 && 256 <= MAX_OP_N) {
            return type_c<MMA_64x256x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 224 == 0 && 224 <= MAX_OP_N) {
            return type_c<MMA_64x224x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 192 == 0 && 192 <= MAX_OP_N) {
            return type_c<MMA_64x192x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 160 == 0 && 160 <= MAX_OP_N) {
            return type_c<MMA_64x160x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 128 == 0 && 128 <= MAX_OP_N) {
            return type_c<MMA_64x128x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 96 == 0 && 96 <= MAX_OP_N) {
            return type_c<MMA_64x96x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 64 == 0 && 64 <= MAX_OP_N) {
            return type_c<MMA_64x64x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 32 == 0 && 32 <= MAX_OP_N) {
            return type_c<MMA_64x32x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 16 == 0 && 16 <= MAX_OP_N) {
            return type_c<MMA_64x16x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 8 == 0 && 8 <= MAX_OP_N) {
            return type_c<MMA_64x8x32_F32E4M3E4M3_SS_TN<>>;
        }
        else {
            static_assert(N == 0, "unsupported WA BATCH configuration");
        }
    }

    using Operation = typename decltype(select_gmma_operation())::type;

    static constexpr typename cute::MMA_Traits<Operation>::Shape_MNK OP_Shape{};

    static constexpr int OP_M = cute::get<0>(OP_Shape);
    static constexpr int OP_N = cute::get<1>(OP_Shape);
    static constexpr int OP_K = cute::get<2>(OP_Shape);

    static constexpr int ITER_M = TILE_OUT / OP_M / BATCH_M / PIPE_M;
    static constexpr int ITER_N = TILE_BATCH / OP_N / BATCH_N / PIPE_N;

    // Weight scales along GMMA-M (OUT): 2 M-halves × sparse V select.
    // Act scales along GMMA-N: OP_N/4 floats/thread (2 cols × OP_N/8 stripes).
    using FragW                              = float[2];
    static constexpr int kActScalesPerThread = OP_N / 4;
    using FragA                              = float[kActScalesPerThread];

    using FragC  = typename Operation::CRegisters[PIPE_M][PIPE_N][BATCH_M][BATCH_N];
    using AccumC = FragC[ITER_M][ITER_N];

    static constexpr int kStepMA = (OP_M * TILE_K) >> 4;
    static constexpr int kStepNB = (OP_N * TILE_K) >> 4;
    static constexpr int kStepKA = (OP_K) >> 4;
    static constexpr int kStepKB = (OP_K) >> 4;

    // Weight 128-block partitioning along OUT (GMMA-M / problem N).
    static constexpr int OUTER_M = std::gcd(TILE_OUT, 128);

    // scale_w[2]: V[0], V[1] for the tile's N-straddle.
    // pred_W[i]: which V to use for OUT subtile i (OUTER_M chunks).
    // act_scales[OP_N/4]: this thread's dense U scales for owned BATCH columns,
    //   laid out stripe-major: for stripe s, indices [2s]=sa(n0), [2s+1]=sa(n1).
    template<class FragC, class AccumC, class FragW, class FragA, class PredW>
    __device__ static void scale_batch_to_accum_wa(AccumC&      accum_C,
                                                   const FragC& frag_C,
                                                   const FragW& scale_w,
                                                   const FragA& act_scales,
                                                   const PredW& pred_W,
                                                   int          offset_W)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < BATCH_M; ++m) {
            // 64-aligned OUT atoms: both M-halves share one weight 128-block.
            // Still index pred by offset_W (atom OUT start) for TILE_OUT straddles.
            const int   wi  = offset_W / OUTER_M;
            const bool  pw  = pred_W[wi];
            const float sw0 = pw ? scale_w[1] : scale_w[0];
            const float sw1 = sw0;

            PRAGMA_UNROLL
            for (int n = 0; n < BATCH_N; ++n) {
                PRAGMA_UNROLL
                for (int c = 0, s = 0; c < OP_N; c += 8, ++s) {
                    const float sa0 = act_scales[2 * s + 0];
                    const float sa1 = act_scales[2 * s + 1];
                    accum_C[m][n][c / 2 + 0] += sw0 * sa0 * frag_C[m][n][c / 2 + 0];
                    accum_C[m][n][c / 2 + 1] += sw0 * sa1 * frag_C[m][n][c / 2 + 1];
                    accum_C[m][n][c / 2 + 2] += sw1 * sa0 * frag_C[m][n][c / 2 + 2];
                    accum_C[m][n][c / 2 + 3] += sw1 * sa1 * frag_C[m][n][c / 2 + 3];
                }
            }
        }
    }

    __device__ static void warpgroup_wait(int n)
    {
        if (n == 0) {
            cute::warpgroup_wait<0>();
        }
        else if (n == 1) {
            cute::warpgroup_wait<1>();
        }
        else if (n == 2) {
            cute::warpgroup_wait<2>();
        }
        else if (n == 3) {
            cute::warpgroup_wait<3>();
        }
        else if (n == 4) {
            cute::warpgroup_wait<4>();
        }
        else if (n == 5) {
            cute::warpgroup_wait<5>();
        }
        else if (n == 6) {
            cute::warpgroup_wait<6>();
        }
        else if (n == 7) {
            cute::warpgroup_wait<7>();
        }
    }

    template<class SmemIterA, class SmemIterB, class FragC>
    __device__ static void gmma_batch(SmemIterA& iter_A, SmemIterB& iter_B, FragC& frag_C)
    {
        constexpr int BATCH_K = TILE_K / OP_K;
        PRAGMA_UNROLL
        for (int k = 0; k < BATCH_K; ++k) {
            PRAGMA_UNROLL
            for (int m = 0; m < BATCH_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < BATCH_N; ++n) {
                    wgmma<Operation>(iter_A, iter_B, frag_C[m][n], k == 0);
                    iter_B += kStepNB;
                }
                iter_B -= kStepNB * BATCH_N;
                iter_A += kStepMA;
            }
            iter_A -= kStepMA * BATCH_M;
            iter_A += kStepKA;
            iter_B += kStepKB;
        }
        iter_A -= kStepKA * BATCH_K;
        iter_B -= kStepKB * BATCH_K;
        cute::warpgroup_commit_batch();
    }

    template<class SmemIterA, class SmemIterB, class FragC, class AccumC, class FragW, class FragA, class PredW>
    __device__ static void gmma_pipe(AccumC&      accum_C,
                                     SmemIterA&   iter_A,
                                     SmemIterB&   iter_B,
                                     FragC&       frag_C,
                                     const FragW& scale_w,
                                     const FragA& act_scales,
                                     const PredW& pred_W,
                                     int          offset_W)
    {
        cute::warpgroup_arrive();
        PRAGMA_UNROLL
        for (int m = 0; m < PIPE_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < PIPE_N; ++n) {
                gmma_batch(iter_A, iter_B, frag_C[m][n]);
                iter_B += kStepNB * BATCH_N;
            }
            iter_B -= kStepNB * BATCH_N * PIPE_N;
            iter_A += kStepMA * BATCH_M;
        }
        iter_A -= kStepMA * BATCH_M * PIPE_M;

        int i = 0;
        PRAGMA_UNROLL
        for (int m = 0; m < PIPE_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < PIPE_N; ++n, ++i) {
                warpgroup_wait(PIPE_M * PIPE_N - i - 1);
                int offset = offset_W + m * BATCH_M * OP_M;
                scale_batch_to_accum_wa(accum_C[m][n], frag_C[m][n], scale_w, act_scales, pred_W, offset);
            }
        }
    }

    template<class SmemIterA, class SmemIterB, class FragC, class AccumC, class FragW, class FragA, class PredW>
    __device__ static void apply(SmemIterA&   iter_A,
                                 SmemIterB&   iter_B,
                                 FragC&       frag_C,
                                 AccumC&      accum_C,
                                 const FragW& scale_w,
                                 const FragA& act_scales,
                                 const PredW& pred_W)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < ITER_N; ++n) {
                int offset_W = m * PIPE_M * BATCH_M * OP_M;
                gmma_pipe(accum_C[m][n], iter_A, iter_B, frag_C, scale_w, act_scales, pred_W, offset_W);
                iter_B += kStepNB * BATCH_N * PIPE_N;
            }
            iter_B -= kStepNB * BATCH_N * PIPE_N * ITER_N;
            iter_A += kStepMA * BATCH_M * PIPE_M;
        }
        iter_A -= kStepMA * BATCH_M * PIPE_M * ITER_M;
    }

    template<class Frag, class Func>
    __device__ static void foreach_C(Frag& frag, Func&& func)
    {
        PRAGMA_UNROLL
        for (int i_m = 0; i_m < ITER_M; ++i_m) {
            PRAGMA_UNROLL
            for (int i_n = 0; i_n < ITER_N; ++i_n) {
                PRAGMA_UNROLL
                for (int p_m = 0; p_m < PIPE_M; ++p_m) {
                    PRAGMA_UNROLL
                    for (int p_n = 0; p_n < PIPE_N; ++p_n) {
                        PRAGMA_UNROLL
                        for (int b_m = 0; b_m < BATCH_M; ++b_m) {
                            PRAGMA_UNROLL
                            for (int b_n = 0; b_n < BATCH_N; ++b_n) {
                                int m = ((i_m * PIPE_M) + p_m * BATCH_M) + b_m;
                                int n = ((i_n * PIPE_N) + p_n * BATCH_N) + b_n;
                                func(frag[i_m][i_n][p_m][p_n][b_m][b_n], m, n);
                            }
                        }
                    }
                }
            }
        }
    }
};

}  // namespace turbomind::gemm
