#pragma once

#include <numeric>

#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/atom/mma_traits.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/sm90_utils.h"

namespace turbomind::gemm {

template<int TILE_M, int TILE_N, int TILE_K, int BATCH_M, int BATCH_N, int PIPE_M, int PIPE_N>
struct ScaledGmmaFP8_TN {

    static constexpr auto select_gmma_operation()
    {
        static_assert(TILE_M % (BATCH_M * PIPE_M) == 0);
        static_assert(TILE_N % (BATCH_N * PIPE_N) == 0);

        constexpr int M = TILE_M / (BATCH_M * PIPE_M);
        constexpr int N = TILE_N / (BATCH_N * PIPE_N);

        static_assert(M % 64 == 0);

        using namespace cute::SM90::GMMA;

        if constexpr (N % 256 == 0) {
            return type_c<MMA_64x256x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 224 == 0) {
            return type_c<MMA_64x224x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 192 == 0) {
            return type_c<MMA_64x192x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 160 == 0) {
            return type_c<MMA_64x160x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 128 == 0) {
            return type_c<MMA_64x128x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 96 == 0) {
            return type_c<MMA_64x96x32_F32E4M3E4M3_SS_TN<>>;
        }
        else if constexpr (N % 64 == 0) {
            return type_c<MMA_64x64x32_F32E4M3E4M3_SS_TN<>>;
        }
        else {
            static_assert(N == 0, "unsupported configuration");
        }
    }

    using Operation = typename decltype(select_gmma_operation())::type;

    static constexpr typename cute::MMA_Traits<Operation>::Shape_MNK OP_Shape{};

    static constexpr int OP_M = cute::get<0>(OP_Shape);
    static constexpr int OP_N = cute::get<1>(OP_Shape);
    static constexpr int OP_K = cute::get<2>(OP_Shape);

    static constexpr int ITER_M = TILE_M / OP_M / BATCH_M / PIPE_M;
    static constexpr int ITER_N = TILE_N / OP_N / BATCH_N / PIPE_N;

    using FragU = float[ITER_M][PIPE_M][BATCH_M][2];
    using FragV = float[2];

    using FragC = typename Operation::CRegisters[PIPE_M][PIPE_N][BATCH_M][BATCH_N];

    using AccumC = FragC[ITER_M][ITER_N];

    static constexpr int kStepMA = (OP_M * TILE_K) >> 4;
    static constexpr int kStepNB = (OP_N * TILE_K) >> 4;
    static constexpr int kStepKA = (OP_K) >> 4;
    static constexpr int kStepKB = (OP_K) >> 4;

    static constexpr int OUTER_N = std::gcd(TILE_N, 128);

    template<class FragC, class AccumC, class FragU, class FragV, class PredV>
    __device__ static void scale_batch_to_accum(AccumC&      accum_C,
                                                const FragC& frag_C,
                                                const FragU& frag_U,
                                                const FragV& frag_V,
                                                const PredV& pred_V,
                                                int          offset_V)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < BATCH_M; ++m) {
            float scales[2][2];
            // TODO: check the compiler's ability to avoid re-computing this
            scales[0][0] = frag_U[m][0] * frag_V[0];
            scales[1][0] = frag_U[m][1] * frag_V[0];
            scales[0][1] = frag_U[m][0] * frag_V[1];
            scales[1][1] = frag_U[m][1] * frag_V[1];
            PRAGMA_UNROLL
            for (int n = 0; n < BATCH_N; ++n) {
                PRAGMA_UNROLL
                for (int c0 = 0; c0 < OP_N; c0 += OUTER_N) {
                    int  i = (offset_V + c0) / OUTER_N;
                    bool p = pred_V[i];
                    PRAGMA_UNROLL
                    for (int c1 = 0; c1 < OUTER_N; c1 += 8) {
                        int c = c0 + c1;
                        accum_C[m][n][c / 2 + 0] += (p ? scales[0][1] : scales[0][0]) * frag_C[m][n][c / 2 + 0];
                        accum_C[m][n][c / 2 + 1] += (p ? scales[0][1] : scales[0][0]) * frag_C[m][n][c / 2 + 1];
                        accum_C[m][n][c / 2 + 2] += (p ? scales[1][1] : scales[1][0]) * frag_C[m][n][c / 2 + 2];
                        accum_C[m][n][c / 2 + 3] += (p ? scales[1][1] : scales[1][0]) * frag_C[m][n][c / 2 + 3];
                    }
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

    template<class SmemIterA, class SmemIterB, class FragC, class AccumC, class FragU, class FragV, class PredV>
    __device__ static void gmma_pipe(AccumC&      accum_C,
                                     SmemIterA&   iter_A,
                                     SmemIterB&   iter_B,
                                     FragC&       frag_C,
                                     const FragU& frag_U,
                                     const FragV& frag_V,
                                     const PredV& pred_V,
                                     int          offset_V)
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
                int offset = offset_V + n * BATCH_N * OP_N;
                scale_batch_to_accum(accum_C[m][n], frag_C[m][n], frag_U[m], frag_V, pred_V, offset);
            }
        }
    }

    template<class SmemIterA, class SmemIterB, class FragC, class AccumC, class FragU, class FragV, class PredV>
    __device__ static void apply(SmemIterA&   iter_A,
                                 SmemIterB&   iter_B,
                                 FragC&       frag_C,
                                 AccumC&      accum_C,
                                 const FragU& frag_U,
                                 const FragV& frag_V,
                                 const PredV& pred_V)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < ITER_N; ++n) {
                int offset_V = n * PIPE_N * BATCH_N * OP_N;
                gmma_pipe(accum_C[m][n], iter_A, iter_B, frag_C, frag_U[m], frag_V, pred_V, offset_V);
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
                            }  // BATCH_N
                        }      // BATCH_M
                    }          // PIPE_N
                }              // PIPE_M
            }                  // ITER_N
        }                      // ITER_M
    }

    template<class Frag, class Func>
    __device__ static void foreach_m(Frag& frag, Func&& func)
    {
        PRAGMA_UNROLL
        for (int i_m = 0; i_m < ITER_M; ++i_m) {
            PRAGMA_UNROLL
            for (int p_m = 0; p_m < PIPE_M; ++p_m) {
                PRAGMA_UNROLL
                for (int b_m = 0; b_m < BATCH_M; ++b_m) {
                    int m = ((i_m * PIPE_M) + p_m * BATCH_M) + b_m;
                    func(frag[i_m][p_m][b_m], m);
                }
            }
        }
    }
};

}  // namespace turbomind::gemm
