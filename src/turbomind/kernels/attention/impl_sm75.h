// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "impl.h"
#include "impl_m16n8.h"
#include "iterator.h"
#include "src/turbomind/kernels/attention/thread_map.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

namespace turbomind::attention {

template<class T, class Layout, int N>
struct Sm75SmemIterK: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;

    using Base::Base;
    using Base::swizzle_uint_ptr;

    static_assert(N % 4 == 0);

    __device__ void Load(Array<T, 2> (&frag_K)[N], int k)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {  // Load (s32,d8) tiles
            const int s = n * 8 + lane_id;
            const int c = k * 8;
            ldsm_x4((Array<uint32_t, 4>&)frag_K[n], swizzle_uint_ptr(s, c));
        }
    }
};

template<class T, class Layout, int N>
struct Sm75SmemIterV0: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;

    using Base::Base;
    using Base::swizzle_uint_ptr;

    static_assert(N % 4 == 0);

    __device__ void Load(Array<T, 2> (&frag_V)[N], int k)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {  // Load (d32,s8) tiles
            const int si = k * 8 + lane_id % 8;
            const int di = n * 8 + lane_id / 8 * 8;
            ldsm_x4_trans((Array<uint32_t, 4>&)frag_V[n], swizzle_uint_ptr(si, di));
        }
    }
};

template<class T, class Layout, int N>
struct Sm75SmemIterV: BaseSmemIterator<T, Layout> {

    using Base = BaseSmemIterator<T, Layout>;
    using Base::Base;
    using Base::swizzle_uint_ptr;

    static_assert(N % 4 == 0);

    Array<int, N / 4> idxs_;

    __device__ Sm75SmemIterV(T* smem): Base{smem}
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {  // Load (d32,s8) tiles
            const int si = 0 * 8 + lane_id % 8;
            const int di = n * 8 + lane_id / 8 * 8;
            idxs_[n / 4] = swizzle_uint_ptr(si, di);
        }
    }

    __device__ void Load(Array<T, 2> (&frag_V)[N], int k)
    {
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {
            const int idx = idxs_[n / 4] + sizeof(T) * k * 8 * Layout::kStride;
            ldsm_x4_trans((Array<uint32_t, 4>&)frag_V[n], idx);
        }
    }
};

template<class T_, int CTA_H_, int CTA_Q_, int CTA_S_, int WARP_H, int WARP_Q, int WARP_S, int HeadDim>
struct Impl<Sm75_1688, T_, T_, CTA_H_, CTA_Q_, CTA_S_, WARP_H, WARP_Q, WARP_S, HeadDim>:
    Impl_m16k8<T_, WARP_Q, WARP_S, HeadDim> {

    using Base = Impl_m16k8<T_, WARP_Q, WARP_S, HeadDim>;

    using Base::OP_M;
    using Base::OP_N;
    using Base::K_M;
    using Base::K_N;
    using Base::V_M;
    using Base::V_N;

    using typename Base::FragS;
    using typename Base::FragO;
    using typename Base::FragM;
    using typename Base::FragL;

    using Base::ForeachS;
    using Base::Softmax;
    using Base::ConvertStoP;
    using Base::StoreO;

    using T   = T_;
    using Tkv = T_;

    static constexpr int kHeadDim = HeadDim;

    static constexpr int CTA_Q = CTA_Q_;
    static constexpr int CTA_S = CTA_S_;

    static constexpr int kWarpCntQ  = CTA_Q / WARP_Q;
    static constexpr int kWarpCntS  = CTA_S / WARP_S;
    static constexpr int kWarpCount = kWarpCntQ * kWarpCntS;

    static constexpr int OP_K = 8;

    static constexpr int K_K = HeadDim / OP_K;  // 128 / 16 = 8
    static constexpr int V_K = WARP_S / OP_K;   //  64 / 16 = 4  -> S4

    using FragQ = Array<T, 4>[K_K][K_M];  // ((q8, d4), (Dk, Qm), (q2, d2))
                                          //    1   2     8  16     8   1
    using FragK = Array<T, 2>[K_K][K_N];  // ((s8, d4), (Dk, Sn), (d2))
                                          //    1   2     8   8     1
    using FragP = Array<T, 4>[V_M][V_K];  // ((q8, s4), (Qm, Sk), (q2, s2))
                                          //    1   2    16   8     8   1
    using FragV = Array<T, 2>[V_K][V_N];  // ((d8, s4), (Sk, Dn), (s2))
                                          //    1   2     8   8     1

    struct Swizzle {
        __device__ int operator()(int index)
        {
            // sssSSSdDDDddd
            // DDD ^= SSS
            constexpr int mask = 0x7 << 7;
            return index ^ ((index & mask) >> 4);
        }
    };

    using SmemLayoutQ = SmemLayout<HeadDim, Swizzle>;
    using SmemLayoutK = SmemLayout<HeadDim, Swizzle>;
    using SmemLayoutP = Identity;
    using SmemLayoutV = SmemLayout<HeadDim, Swizzle>;

    struct SharedStorage {
        union {
            __align__(16) T Q[CTA_Q * SmemLayoutQ::kStride];
            struct {
                __align__(16) T K[CTA_S * SmemLayoutK::kStride];
                __align__(16) T V[CTA_S * SmemLayoutV::kStride];
            };
            struct {
                T P[1];
            };
        };
    };

    using SmemIterQ = NullSmemIter<T>;
    using SmemIterP = NullSmemIter<T>;

    using SmemIterK = Sm75SmemIterK<T, SmemLayoutK, K_N>;
    using SmemIterV = Sm75SmemIterV<T, SmemLayoutV, V_N>;

    static constexpr bool kUseSmemQ = false;

    using ThreadMapQ  = RakedThreadMap<HeadDim, CTA_Q, 8, kWarpCount>;
    using ThreadMapKV = RakedThreadMap<HeadDim, CTA_S, 8, kWarpCount>;

    __device__ static void Sync()
    {
        __syncthreads();
    }

    __device__ static void TransformQ(T* smem_Q, FragQ& frag_Q)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        __syncwarp();

        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_Q);
        if constexpr (!kUseSmemQ) {
            // Load from shared memory using LDSM, rearrange to m16n8k16 atom layout
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_K; k += 2) {
                    const int qi     = m * 16 + lane_id % 16 + warp_id * WARP_Q;
                    const int di     = k * +8 + lane_id / 16 * 8;
                    const int offset = sizeof(T) * SmemLayoutQ::swizzle(qi, di);
                    ldsm_x4((Array<uint32_t, 4>&)frag_Q[k][m], smem_int_ptr + offset);
                }
            }
        }
        else {
            // Rearrange Q in smem so that swizzling is not needed for later LDSMs
            const int group_id      = lane_id / 16;
            const int group_lane_id = lane_id % 16;
            PRAGMA_UNROLL
            for (int k = 0; k < K_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
                    const int qi     = m * 16 + group_lane_id % 8 + group_id * 8 + warp_id * WARP_Q;
                    const int di     = k * 16 + group_lane_id / 8 * 8;
                    const int offset = sizeof(T) * SmemLayoutQ::swizzle(qi, di);
                    ldsm_x4((Array<uint32_t, 4>&)frag_Q[k][m], smem_int_ptr + offset);
                }
            }

            __syncthreads();

            constexpr int THREADS = kWarpCount * WARP_SIZE;
            PRAGMA_UNROLL
            for (int k = 0; k < K_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
                    constexpr int kVecSize = 8;
                    Store(&smem_Q[(k * K_M * THREADS + m * THREADS + threadIdx.x) * kVecSize], frag_Q[k][m]);
                }
            }
        }
    }

    template<class SmemQ, class SmemK>
    __device__ static void ComputeQK(SmemQ& smem_Q, SmemK& smem_K, FragQ& frag_Q, FragS& frag_S)
    {
        FragK frag_K;

        smem_K.Load(frag_K[0], 0);
        if constexpr (kUseSmemQ) {
            smem_Q.Load(frag_Q[0], 0);
        }

        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                smem_K.Load(frag_K[k + 1], k + 1);
                if constexpr (kUseSmemQ) {
                    smem_Q.Load(frag_Q[k + 1], k + 1);
                }
            }
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    const int nn = n ^ 2;
                    mma_m16n8k8_row_col(frag_S[m][nn], frag_Q[k][m], frag_K[k][nn], frag_S[m][nn]);
                }
            }
        }
    }

    template<class SmemP, class SmemV>
    __device__ static void ComputePV(SmemP&, SmemV& smem_V, const FragP& frag_P, FragO& frag_O)
    {
        FragV frag_V;

        smem_V.Load(frag_V[0], 0);

        PRAGMA_UNROLL
        for (int k = 0; k < V_K; ++k) {
            if (k < V_K - 1) {
                smem_V.Load(frag_V[k + 1], k + 1);
            }
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    mma_m16n8k8_row_col(frag_O[m][n], frag_P[m][k], frag_V[k][n], frag_O[m][n]);
                }
            }
        }
    }
};

}  // namespace turbomind::attention