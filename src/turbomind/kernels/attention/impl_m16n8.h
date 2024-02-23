// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm_s_f16/common.h"

namespace turbomind::attention {

template<class T, int WARP_H, int WARP_Q, int WARP_S, int HeadDim>
struct Impl_m16k8 {

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 8;

    static constexpr int K_M = WARP_Q / OP_M;  //  16 / 16 = 1
    static constexpr int K_N = WARP_S / OP_N;  //  64 /  8 = 8

    static constexpr int V_M = WARP_Q / OP_M;   //  16 / 16 = 1
    static constexpr int V_N = HeadDim / OP_N;  // 128 /  8 = 16 -> D16

    template<class S>
    using FragS_ = Array<S, 4>[K_M][K_N];     // ((q8, s4), (Qm, Sn), (q2, s2))
                                              //    1   2    16   8     8   1
    using FragO = Array<float, 4>[V_M][V_N];  // ((q8, d4), (Qm, Dn), (q2, d2))
                                              //    1   2    16   8     8   1
    using FragM = Array<float, 2>[V_M];       // ((q8, _4), Qm, q2) => FragS with all S dim reduced
                                              //    1   0   16   8
    using FragS = FragS_<float>;
    using FragL = FragM;

    static constexpr bool kDeferReduceL = false;

    template<class Func>
    __device__ static void ForeachML(FragM& frag_M, FragL& frag_L, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {  // Q
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                const int qi = lane_id / 4 * 1 + m * OP_M + q * 8 + warp_id * WARP_Q;
                const int ri = lane_id % 4 * 1;
                ((Func&&)func)(qi % WARP_H, qi / WARP_H, ri, frag_M[m][q], frag_L[m][q]);
            }
        }
    }

    template<class Fragment, class Func>
    __device__ static void ForeachS(Fragment& S, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {  // Q
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {  // KV
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        const int qi = lane_id / 4 * 1 + m * OP_M + q * 8 + warp_id * WARP_Q;
                        const int ki = lane_id % 4 * 2 + n * OP_N + s * 1;
                        ((Func&&)func)(qi % WARP_H, qi / WARP_H, ki, /*ri*/ 0, S[m][n][q * 2 + s]);
                    }
                }
            }
        }
    }

    template<bool is_residue>
    __device__ static void Softmax(FragS& frag_S, FragM& frag_M, FragM& frag_L, FragO& frag_O, float qk_scale)
    {
        FragM prev_M;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            prev_M[m] = frag_M[m];
        }

        // maximum
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {  // Q
            auto& row_M = frag_M[m];
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {  // KV
                auto& C = frag_S[m][n];
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    row_M[q] = fmaxf(row_M[q], fmaxf(C[q * 2 + 0], C[q * 2 + 1]));  // reduce over local pair
                }
            }
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {  // reduce over thread group within warp (within warp tiles)
                row_M[q] = fmaxf(row_M[q], __shfl_xor_sync(uint32_t(-1), row_M[q], 1));
                row_M[q] = fmaxf(row_M[q], __shfl_xor_sync(uint32_t(-1), row_M[q], 2));
            }
        }

        FragM expdiff_M;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                // exp(M - M'), isinf(frag_M) => isnan(expdiff_M)
                expdiff_M[m][q] = exp2f((prev_M[m][q] - frag_M[m][q]) * qk_scale);
                if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M[m][q] = 0.f;
                }
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                frag_L[m][q] *= expdiff_M[m][q];
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                float tmp_L{};
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        // unnormalized prob
                        float p = exp2f(frag_S[m][n][q * 2 + s] * qk_scale - frag_M[m][q] * qk_scale);
                        if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                            p = 0.f;
                        }
                        tmp_L += p;
                        frag_S[m][n][q * 2 + s] = p;
                    }
                }
                if constexpr (!kDeferReduceL) {
                    tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 1);
                    tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 2);
                }
                frag_L[m][q] += tmp_L;  // update L
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    PRAGMA_UNROLL
                    for (int d = 0; d < 2; ++d) {
                        frag_O[m][n][q * 2 + d] *= expdiff_M[m][q];  // Rescale previous output
                    }
                }
            }
        }
    }

    template<class FragP>
    __device__ static void ConvertStoP(FragS& frag_S, FragP& frag_P, T* smem_P)
    {
        FragS_<T>& frag_Ps = (FragS_<T>&)frag_P;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int q = 0; q < 2; ++q) {
                    PRAGMA_UNROLL
                    for (int s = 0; s < 2; ++s) {
                        frag_Ps[m][n][q * 2 + s] = static_cast<T>(frag_S[m][n][q * 2 + s]);
                    }
                }
            }
        }

#if 0
        if (!smem_P) {
            FragS_<T>& frag_Ps = (FragS_<T>&)frag_P;
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        PRAGMA_UNROLL
                        for (int s = 0; s < 2; ++s) {
                            frag_Ps[m][n][q * 2 + s] = static_cast<T>(frag_S[m][n][q * 2 + s]);
                        }
                    }
                }
            }
        }
        else {
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; n += 2) {
                    Array<T, 8> tmp_P;
                    PRAGMA_UNROLL
                    for (int s1 = 0; s1 < 2; ++s1) {
                        PRAGMA_UNROLL
                        for (int q = 0; q < 2; ++q) {
                            PRAGMA_UNROLL
                            for (int s = 0; s < 2; ++s) {
                                tmp_P[s1 * 4 + q * 2 + s] = static_cast<T>(frag_S[m][n + s1][q * 2 + s]);
                            }
                        }
                    }
                    const int     k        = n / 2;
                    constexpr int kThreads = kWarpCntQ * WARP_SIZE;
                    Store(&smem_P[(k * V_M * kThreads + m * kThreads + threadIdx.x) * 8], tmp_P);
                }
            }
            __syncwarp();  // really?
        }
#endif
    }

    template<class Storage>
    __device__ static void Merge(FragO& frag_O, FragM& frag_M, FragL& frag_L, float qk_scale, Storage& storage) {}

    template<bool is_norm, class Func, class Storage>
    __device__ static void StoreO(FragO& frag_O, FragL& frag_L, Storage& storage, Func&& func)
    {
        FragL inv_L;
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                if constexpr (kDeferReduceL) {
                    frag_L[m][q] += __shfl_xor_sync(uint32_t(-1), frag_L[m][q], 1);
                    frag_L[m][q] += __shfl_xor_sync(uint32_t(-1), frag_L[m][q], 2);
                }
                inv_L[m][q] = fdividef(1.f, frag_L[m][q]);
            }
        }

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                const int qi = lane_id / 4 * 1 + m * OP_M + q * 8 + warp_id * WARP_Q;
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    if constexpr (is_norm) {
                        PRAGMA_UNROLL
                        for (int d = 0; d < 2; ++d) {
                            frag_O[m][n][q * 2 + d] *= inv_L[m][q];
                        }
                    }
                    const int di = n * 8 + lane_id % 4 * 2;
                    ((Func&&)func)(qi % WARP_H, qi / WARP_H, di, (Array<float, 2>&)frag_O[m][n][q * 2]);
                }
            }
        }
    }
};

}  // namespace turbomind::attention