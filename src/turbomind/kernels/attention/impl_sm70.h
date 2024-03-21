// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "array_ops.h"
#include "impl.h"
#include "src/turbomind/kernels/attention/iterator.h"
#include "src/turbomind/kernels/attention/thread_map.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include <cmath>

namespace turbomind::attention {

__inline__ __device__ void
mma_m8n8k4_row_col(Array<float, 8>& d, const Array<half, 4>& a, const Array<half, 4>& b, Array<float, 8>& c)
{
#if TURBOMIND_ARCH_SM70
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    // clang-format off
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11},"
        "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
// clang-format on
#endif
}

__inline__ __device__ void
mma_m8n8k4_row_row(Array<float, 8>& d, const Array<half, 4>& a, const Array<half, 4>& b, Array<float, 8>& c)
{
#if TURBOMIND_ARCH_SM70
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    // clang-format off
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32"
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11},"
        "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
// clang-format on
#endif
}

template<class T, class Layout, int M>
struct Sm70SmemIterQ: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;

    using Base::Base;
    using Base::ptr;

    __device__ void Load(Array<half, 4> (&frag_Q)[M], int k)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < M; ++m) {
            const int qi = m * 16 + (lane_id & 8) + lane_id % 4 + lane_id / 16 * 4 + warp_id * 16;
            const int di = k * 4;
            Lds(frag_Q[m], ptr(qi, di));
        }
    }
};

template<class T, class Layout, int N>
struct Sm70SmemIterK: BaseSmemIterator<T, Layout> {
    using Base = BaseSmemIterator<T, Layout>;

    using Base::Base;
    using Base::smem_;

    __device__ void Load(Array<half, 4> (&frag_K)[N], int k, int)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; ++n) {
            const int s = n * 16 + lane_id / 16 * 4 + (lane_id & 4) * 2 + lane_id % 4;
            const int c = k * 4;
            Lds(frag_K[n], &smem_[Layout::apply(s, c)]);
        }
    }
};

template<class T, class Layout, int N>
struct Sm70SmemIterV: BaseSmemIterator<T, Layout> {
    using Base = BaseSmemIterator<T, Layout>;
    using Base::smem_;

    static_assert(N % 2 == 0);

    Array<int, N / 2> idxs_;

    __device__ explicit Sm70SmemIterV(T* smem): Base{smem}
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < 8; n += 2) {
            const int s  = 0 * 4 + lane_id % 4;
            const int c  = n * 16 + lane_id / 16 * 4 + (lane_id & 4) * 2;
            idxs_[n / 2] = Layout::apply(s, c);
        }
    }

    __device__ void Load(Array<half, 4> (&frag_V)[N], int k, int)
    {
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 2) {
            const int idx = idxs_[n / 2] + k * 4 * Layout::C0;
            Lds((Array<half, 8>&)frag_V[n], &smem_[idx]);
        }
    }
};

template<class T, class Layout, int M>
struct Sm70SmemIterP: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;

    using Base::Base;
    using Base::ptr;

    __device__ void Load(Array<half, 4> (&frag_P)[M], int k)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < M; ++m) {
            const int qi = m * 16 + lane_id / 16 * 4 + (lane_id & 8) + lane_id % 4 + warp_id * 16;
            const int si = k * 4;
            Lds(frag_P[m], ptr(qi, si));
        }
    }
};

template<class T_, int CTA_H_, int CTA_Q_, int CTA_S_, int WARP_H_, int WARP_Q, int WARP_S, int HeadDim>
struct Impl<Sm70_884, T_, T_, CTA_H_, CTA_Q_, CTA_S_, WARP_H_, WARP_Q, WARP_S, HeadDim> {

    using T   = T_;
    using Tkv = T_;

    using Arch = Sm70_884;

    static constexpr int CTA_H    = CTA_H_;
    static constexpr int CTA_Q    = CTA_Q_;
    static constexpr int CTA_S    = CTA_S_;
    static constexpr int kHeadDim = HeadDim;

    static constexpr int kWarpCntQ  = CTA_Q / WARP_Q;
    static constexpr int kWarpCntS  = CTA_S / WARP_S;
    static constexpr int kWarpCount = kWarpCntQ * kWarpCntS;

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 4;

    static constexpr int K_M = WARP_Q / OP_M;   // 1
    static constexpr int K_N = WARP_S / OP_N;   // 4
    static constexpr int K_K = HeadDim / OP_K;  // 32

    static constexpr int V_M = WARP_Q / OP_M;   // 1
    static constexpr int V_N = HeadDim / OP_N;  // 8
    static constexpr int V_K = WARP_S / OP_K;   // 16

    //  +---+---+
    //  | 0 | 1 |
    //  +---+---+
    //  | 2 | 3 |
    //  +---+---+
    using FragQ = Array<half, 4>[K_K][K_M];   //    (q2,q2,x2,q4) (Dk,Qm) (d4)
                                              //      4  8  0  1    4 16    1
    using FragK = Array<half, 4>[K_K][K_N];   //    (s2,x2,s2,s4) (Dk,Sn) (d4)
                                              //      4  0  8  1    4 16    1
    using FragS = Array<float, 8>[K_M][K_N];  // (q2,q2,s2,s2,q2) (Qm,Sn) (s2,q2,s2)
                                              //   4  8  8  2  1   16 16    4  2  1
    using FragP = Array<half, 4>[V_K][V_M];   //    (q2,q2,x2,q4) (Sk,Qm) (s4)
                                              //      4  8  0  1    4 16    1
    using FragV = Array<half, 4>[V_K][V_N];   //    (d2,x2,d2,s4) (Sk,Dn) (d4)       [row major]
                                              //      4  0  8  1    4 16    1
    using FragO = Array<float, 8>[V_M][V_N];  // (q2,q2,d2,d2,q2) (Qm,Dn) (d2,q2,d2)
                                              //   4  8  8  2  1   16 16    4  2  1
    using FragM = Array<float, 2>[V_M];       // (q2,q2,_2,_2,q2) (Qm)    (q2))
    using FragL = FragM;

    // using Swizzle = Identity;

    struct SwizzleV {

        __device__ static int apply(int offset)
        {
            // Rearrange for LDS.128 (also avoid bank-conflict along C)
            // 6543210
            // dDDDDdd
            offset = ((offset & 8) << 2) ^ offset;                                     // x[5] ^= x[3]
            offset = ((offset & ~20) | (((offset & 16) >> 2) | ((offset & 4) << 2)));  // swap(x[4], x[2])

            // Shuffle C according S to avoid bank-conflict
            // ssssSSdDDddd
            offset = ((offset & (0x3 << 6)) >> 3) ^ offset;
            return offset;
        }

        __device__ int operator()(int offset)
        {
            return apply(offset);
        }
    };

    using SmemLayoutQ = SmemLayoutV2<CTA_Q, HeadDim + 4, 1, 1, Identity>;
    using SmemLayoutP = SmemLayoutV2<CTA_Q, CTA_S + 4, 1, 1, Identity>;
    using SmemLayoutK = SmemLayoutV2<CTA_S, HeadDim + 4, 1, 1, Identity>;
    using SmemLayoutV = SmemLayoutV2<CTA_S, HeadDim, CTA_S, 64, SwizzleV>;

    struct SharedStorage {
        union {
            __align__(16) T Q[SmemLayoutQ::kSize];
            struct {
                __align__(16) T K[SmemLayoutK::kSize];
                __align__(16) T V[SmemLayoutV::kSize];
                __align__(16) T P[SmemLayoutP::kSize];
            };
        };
    };

    using SmemIterQ = T*;
    using SmemIterP = T*;

    using SmemIterK = Sm70SmemIterK<T, SmemLayoutK, K_N>;
    using SmemIterV = Sm70SmemIterV<T, SmemLayoutV, V_N>;

    static constexpr bool kUseSmemQ = false;
    static constexpr bool kUseSmemP = false;

    using ThreadMapQ  = RakedThreadMap<HeadDim, CTA_Q, 4, kWarpCount>;
    using ThreadMapKV = RakedThreadMap<HeadDim, CTA_S, 4, kWarpCount>;

    using TransformK = float2;
    using TransformV = float2;

    static constexpr bool kDeferReduceL = true;

    __device__ static void Sync()
    {
        __syncthreads();
    }

    __device__ static T* GetSmemK(SharedStorage& storage)
    {
        return storage.K;
    }

    __device__ static T* GetSmemV(SharedStorage& storage)
    {
        return storage.V;
    }

    template<class Fragment, class Func>
    __device__ static void ForeachS(Fragment& S, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            const int qi = m * OP_M + (lane_id & 8) + (lane_id & 1) + lane_id / 16 * 4 + q * 2;
                            const int si = n * OP_N + (lane_id & 4) * 2 + (lane_id & 2) + s1 * 4 + s0;
                            ((Func &&) func)(0, warp_id * WARP_Q + qi, si, /*ri*/ 0, S[m][n][s1 * 4 + q * 2 + s0]);
                        }
                    }
                }
            }
        }
    }

    __device__ static void TransformQ(const T* smem_Q, FragQ& frag_Q)
    {
        if constexpr (!kUseSmemQ) {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < K_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
                    const int qi = m * OP_M + (lane_id & 8) + lane_id % 4 + lane_id / 16 * 4 + warp_id * WARP_Q;
                    const int di = k * 4;
                    Lds(frag_Q[k][m], &smem_Q[SmemLayoutQ::apply(qi, di)]);
                }
            }
        }
    }

    template<class SmemQ, class SmemK, class Prefetch, class Preload>
    __device__ static void ComputeQK(SmemQ&      smem_Q,
                                     SmemK&      smem_K,
                                     FragQ&      frag_Q,
                                     FragK&      frag_K,
                                     FragS&      frag_S,
                                     TransformV& transform,
                                     int         offset,
                                     Prefetch&&  prefetch,
                                     Preload&&   preload)
    {
        // smem_K.Load(frag_K[0], 0);
        // if constexpr (kUseSmemQ) {
        //     smem_Q.Load(frag_Q[0], 0);
        // }

        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                smem_K.Load(frag_K[k + 1], k + 1, offset);
                // if constexpr (kUseSmemQ) {
                //     smem_Q.Load(frag_Q[k + 1], k + 1);
                // }
            }
            else {
                ((Preload &&) preload)();
            }
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    const int nn = n ^ 1;
                    mma_m8n8k4_row_col(frag_S[m][nn], frag_Q[k][m], frag_K[k][nn], frag_S[m][nn]);
                }
            }
        }
    }

    template<class SmemP, class SmemV, class Prefetch, class Preload>
    __device__ static void ComputePV(SmemP&      smem_P,
                                     SmemV&      smem_V,
                                     FragP&      frag_P,
                                     FragV&      frag_V,
                                     FragO&      frag_O,
                                     TransformV& transform,
                                     int         offset,
                                     Prefetch&&  prefetch,
                                     Preload&&   preload)
    {
        // FragV frag_V;

        // smem_V.Load(frag_V[0], 0);
        // if constexpr (kUseSmemP) {
        //     smem_P.Load(frag_P[0], 0);
        // }

        PRAGMA_UNROLL
        for (int k = 0; k < V_K; ++k) {
            if (k < V_K - 1) {
                smem_V.Load(frag_V[k + 1], k + 1, offset);
                // if constexpr (kUseSmemP) {
                //     smem_P.Load(frag_P[k + 1], k + 1);
                // }
            }
            else {
                ((Preload &&) preload)();
            }
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    mma_m8n8k4_row_row(frag_O[m][n], frag_P[k][m], frag_V[k][n], frag_O[m][n]);
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

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; ++n) {
                PRAGMA_UNROLL
                for (int s1 = 0; s1 < 2; ++s1) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            frag_M[m][q] =
                                fmaxf(frag_M[m][q], frag_S[m][n][s1 * 4 + q * 2 + s0]);  // reduce over local quad
                        }
                    }
                }
            }
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {  // reduce over thread group within warp (within warp tiles)
                frag_M[m][q] = fmaxf(frag_M[m][q], __shfl_xor_sync(uint32_t(-1), frag_M[m][q], 2));
                frag_M[m][q] = fmaxf(frag_M[m][q], __shfl_xor_sync(uint32_t(-1), frag_M[m][q], 4));
            }
        }

        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                // exp(M - M'), isinf(frag_M) => isnan(expdiff_M)
                float expdiff_M = exp2f((prev_M[m][q] - frag_M[m][q]) * qk_scale);
                if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                    expdiff_M = 0.f;
                }
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    PRAGMA_UNROLL
                    for (int s1 = 0; s1 < 2; ++s1) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            frag_O[m][n][s1 * 4 + q * 2 + s0] *= expdiff_M;  // Rescale previous output
                        }
                    }
                }
                frag_L[m][q] *= expdiff_M;
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
                    for (int s1 = 0; s1 < 2; ++s1) {
                        PRAGMA_UNROLL
                        for (int s0 = 0; s0 < 2; ++s0) {
                            // unnormalized prob, optimized to FFMA
                            float p = exp2f(frag_S[m][n][s1 * 4 + q * 2 + s0] * qk_scale - frag_M[m][q] * qk_scale);
                            if (is_residue && frag_M[m][q] == -std::numeric_limits<float>::infinity()) {
                                p = 0.f;
                            }
                            tmp_L += p;
                            frag_S[m][n][s1 * 4 + q * 2 + s0] = p;
                        }
                    }
                }
                if constexpr (!kDeferReduceL) {
                    tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 2);
                    tmp_L += __shfl_xor_sync(uint32_t(-1), tmp_L, 4);
                }
                frag_L[m][q] = frag_L[m][q] + tmp_L;  // update L
            }
        }
    }

    __device__ static void ConvertStoP(FragS& frag_S, FragP& frag_P, T* smem_P)
    {
        ForeachS(frag_S, [&](int, int qi, int si, int ri, float p) { smem_P[SmemLayoutP::apply(qi, si)] = half(p); });

        if constexpr (!kUseSmemP) {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < V_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < V_M; ++m) {
                    const int qi = m * OP_M + lane_id / 16 * 4 + (lane_id & 8) + lane_id % 4 + warp_id * WARP_Q;
                    const int si = k * OP_K;
                    Lds(frag_P[k][m], &smem_P[SmemLayoutP::apply(qi, si)]);
                }
            }
        }
    }

    template<class Func>
    __device__ static void ForeachML(FragM& frag_M, FragL& frag_L, Func&& func){};

    template<bool is_norm, class Func>
    __device__ static void StoreO(FragO& frag_O, FragL& frag_L, SharedStorage& storage, Func&& func)
    {
        FragL inv_L;
        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int q = 0; q < 2; ++q) {
                if constexpr (kDeferReduceL) {
                    frag_L[m][q] += __shfl_xor_sync(uint32_t(-1), frag_L[m][q], 2);
                    frag_L[m][q] += __shfl_xor_sync(uint32_t(-1), frag_L[m][q], 4);
                }
                inv_L[m][q] = fdividef(1.f, frag_L[m][q] + 1e-8f);
            }
        }

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int mm = lane_id / 16 * 4 + (lane_id & 8) + (lane_id & 1);
        const int nn = (lane_id & 4) * 2 + (lane_id & 2);

        PRAGMA_UNROLL
        for (int m = 0; m < V_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; ++n) {
                PRAGMA_UNROLL
                for (int d1 = 0; d1 < 2; ++d1) {
                    PRAGMA_UNROLL
                    for (int q = 0; q < 2; ++q) {
                        const int qi = m * OP_M + mm + q * 2 + warp_id * WARP_Q;
                        const int di = n * OP_N + nn + d1 * 4;
                        if constexpr (is_norm) {
                            PRAGMA_UNROLL
                            for (int d0 = 0; d0 < 2; ++d0) {
                                frag_O[m][n][d1 * 4 + q * 2 + d0] *= inv_L[m][q];
                            }
                        }
                        ((Func &&) func)(0, qi, di, (Array<float, 2>&)frag_O[m][n][d1 * 4 + q * 2]);
                    }
                }
            }
        }
    }
};

}  // namespace turbomind::attention
