// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_m16n8.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/core/thread_map.h"

namespace turbomind::attention {

template<class T_, int CTA_H_, int CTA_Q_, int CTA_S_, int WARP_H, int WARP_Q, int WARP_S, int HeadDim, int Stages>
struct Impl<MMA_16816, T_, T_, CTA_H_, CTA_Q_, CTA_S_, WARP_H, WARP_Q, WARP_S, HeadDim, Stages>:
    Impl_m16k8<T_, WARP_H, WARP_Q, WARP_S, HeadDim> {

    using Base = Impl_m16k8<T_, WARP_H, WARP_Q, WARP_S, HeadDim>;

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

    static constexpr int CTA_H = CTA_H_;
    static constexpr int CTA_Q = CTA_Q_;
    static constexpr int CTA_S = CTA_S_;

    static constexpr int kWarpCntQ  = CTA_Q * CTA_H / WARP_Q;
    static constexpr int kWarpCntS  = CTA_S / WARP_S;
    static constexpr int kWarpCount = kWarpCntQ * kWarpCntS;

    static constexpr int OP_K = 16;

    static constexpr int K_K = HeadDim / OP_K;  // 128 / 16 = 8
    static constexpr int V_K = WARP_S / OP_K;   //  64 / 16 = 4  -> S4

    using FragQ = Array<T, 8>[K_K][K_M];  // ((q8, d4), (Dk, Qm), (d2, q2, d2))
                                          //    1   2    16  16     8   8   1
    using FragK = Array<T, 4>[K_K][K_N];  // ((s8, d4), (Dk, Sn), (d2, d2))
                                          //    1   2    16   8     8   1
    using FragP = Array<T, 8>[V_M][V_K];  // ((q8, s4), (Qm, Sk), (s2, q2, s2))
                                          //    1   2    16  16     8   8   1
    using FragV = Array<T, 4>[V_K][V_N];  // ((d8, s4), (Sk, Dn), (s2, s2))
                                          //    1   2    16   8     8   1

    static_assert(sizeof(FragS) / 2 == sizeof(FragP));

    using SmemLayoutQ = SmemLayoutV2<CTA_Q * CTA_H, HeadDim, 64, 128, Swizzle<3, 3, 4>>;
#if 0
    using SmemLayoutK = SmemLayoutV2<CTA_S, HeadDim, 16, 64, Swizzle<3, 3, 3>>;
    using SmemLayoutV = SmemLayoutV2<CTA_S, HeadDim, 16, 64, Swizzle<3, 3, 3>>;
#else
    using SmemLayoutK = SmemLayoutV2<CTA_S, HeadDim, 16, 128, Swizzle<3, 3, 4>>;
    using SmemLayoutV = SmemLayoutV2<CTA_S, HeadDim, 16, 128, Swizzle<3, 3, 4>>;
#endif

    using SmemLayoutKVp = void;

    union SharedStorage {
        __align__(16) T KV[Stages * (SmemLayoutK::kSize + SmemLayoutV::kSize) / 2];
        __align__(16) T Q[SmemLayoutQ::kSize];
    };

    static constexpr bool kUseSmemQ = false;
    static constexpr bool kUseSmemP = false;

    using ThreadMapQ  = RakedThreadMap<HeadDim, CTA_Q * CTA_H, 8, kWarpCount>;
    using ThreadMapKV = RakedThreadMap<HeadDim, CTA_S, 8, kWarpCount>;

    using ThreadMapKVp = void;

    static constexpr int kBatchK = std::min(4, ThreadMapKV::kIterS);
    static constexpr int kBatchV = kBatchK;

    __device__ static void Sync()
    {
        __syncthreads();
    }

    template<class GmemIterK, class GmemIterV>
    __device__ static void SetSmemKV(GmemIterK& gmem_K, GmemIterV& gmem_V, SharedStorage& storage, bool offset_kv)
    {
        int pred = offset_kv;
        gmem_K.SetSmem(storage.KV);
        gmem_V.SetSmem(storage.KV + pred * SmemLayoutK::kSize);
    }

    __device__ static void TransformQ(T* smem_Q, FragQ& frag_Q)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        __syncwarp();

        SmemAccessor<T, SmemLayoutQ> sQ{smem_Q};

        // Load from shared memory using LDSM, rearrange to m16n8k16 atom layout
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            PRAGMA_UNROLL
            for (int k = 0; k < K_K; ++k) {
                const int qi = lane_id % 16 * 1 + m * 16 + warp_id * WARP_Q;
                const int di = lane_id / 16 * 8 + k * 16;
                ldsm_x4((Array<uint32_t, 4>&)frag_Q[k][m], cast_smem_ptr_to_uint(&sQ(qi, di)));
            }
        }

        if constexpr (kUseSmemQ) {
            __syncthreads();

            // Rearrange Q in smem so that swizzling is not needed for later LDSMs
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

    struct StateQK {
        SmemAccessor<T, SmemLayoutK> smem_K;

        FragQ frag_Q;
        FragK frag_K;

        __device__ StateQK(SharedStorage& storage, FragQ frag_Q_): smem_K{storage.KV}
        {
            static_assert(!kUseSmemQ, "not implemented");
            PRAGMA_UNROLL
            for (int k = 0; k < K_K; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < K_M; ++m) {
                    frag_Q[k][m] = frag_Q_[k][m];
                }
            }
        }

        __device__ void Load(int k, int pipe_iter)
        {
            const int lane_id       = threadIdx.x % WARP_SIZE;
            const int group_id      = lane_id / 16;
            const int group_lane_id = lane_id % 16;
            const int offset_s      = group_lane_id % 8 + group_id * 8;
            const int offset_c      = group_lane_id / 8 * 8;
            const int offset        = pipe_iter * SmemLayoutK::kSize;
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; n += 2) {  // Load (s16,d16) tiles
                const int s = n * 8 + offset_s;
                const int c = k * 16 + offset_c;
                ldsm_x4((Array<uint32_t, 4>&)frag_K[k][n], cast_smem_ptr_to_uint(&smem_K(s, c, offset)));
            }
        }

        __device__ void Transform(int k) {}
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputeQK(StateQK state_QK, FragS& frag_S, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                state_QK.Load(k + 1, offset);
            }
            else {
                ((Preload &&) preload)();
            }
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    const int nn = (Stages == 2) ? (n ^ 1) : (n ^ 2);
                    mma_m16n8k16_row_col(frag_S[m][nn], state_QK.frag_Q[k][m], state_QK.frag_K[k][nn], frag_S[m][nn]);
                }
            }
            if (k < K_K - 1) {
                ((Prefetch &&) prefetch)(k);
            }
            if (k == K_K - 2) {
                ((Prefetch &&) prefetch)(K_K - 1);
            }
        }
    }

    struct StatePV {
        SmemAccessor<T, SmemLayoutV> smem_V;

        FragP frag_P;
        FragV frag_V;

        __device__ StatePV(SharedStorage& storage, bool offset = false):
            smem_V{storage.KV + (offset ? SmemLayoutK::kSize : 0)}
        {
        }

        __device__ void Load(int k, int pipe_iter)
        {
            const int lane_id  = threadIdx.x % WARP_SIZE;
            const int offset_s = lane_id % 16;
            const int offset_c = lane_id / 16 * 8;
            const int offset   = pipe_iter * SmemLayoutV::kSize;
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; n += 2) {  // Load (d16,s16) tiles
                const int s = k * 16 + offset_s;
                const int c = n * 8 + offset_c;
                ldsm_x4_trans((Array<uint32_t, 4>&)frag_V[k][n], cast_smem_ptr_to_uint(&smem_V(s, c, offset)));
            }
        }

        __device__ void Transform(int k) {}
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputePV(StatePV state_PV, FragO& frag_O, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        PRAGMA_UNROLL
        for (int k = 0; k < V_K; ++k) {
            if (k < V_K - 1) {
                state_PV.Load(k + 1, offset);
            }
            else {
                ((Preload &&) preload)();
            }
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    const int nn = n ^ 0;
                    mma_m16n8k16_row_col(frag_O[m][nn], state_PV.frag_P[m][k], state_PV.frag_V[k][nn], frag_O[m][nn]);
                }
            }
            if (k < V_K - 1) {
                ((Prefetch &&) prefetch)(k);
            }
            if (k == V_K - 2) {
                ((Prefetch &&) prefetch)(V_K - 1);
            }
        }
    }
};

}  // namespace turbomind::attention
