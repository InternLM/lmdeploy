// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "impl.h"
#include "impl_m16n8.h"
#include "iterator.h"
#include "src/turbomind/kernels/attention/thread_map.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

namespace turbomind::attention {

template<class T_, int CTA_H_, int CTA_Q_, int CTA_S_, int WARP_H, int WARP_Q, int WARP_S, int HeadDim>
struct Impl<MMA_1688, T_, T_, CTA_H_, CTA_Q_, CTA_S_, WARP_H, WARP_Q, WARP_S, HeadDim, 2>:
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

    using SmemLayoutQ = SmemLayoutV2<CTA_Q * CTA_H, HeadDim, 64, 128, Swizzle<3, 3, 4>>;
    using SmemLayoutK = SmemLayoutV2<CTA_S, HeadDim, 32, 128, Swizzle<3, 3, 4>>;
    using SmemLayoutV = SmemLayoutV2<CTA_S, HeadDim, 16, 128, Swizzle<3, 3, 4>>;

    using SmemLayoutKVp = void;

    union SharedStorage {
        __align__(16) T Q[SmemLayoutQ::kSize];
        struct {
            __align__(16) Tkv K[SmemLayoutK::kSize];
            __align__(16) Tkv V[SmemLayoutV::kSize];
        };
    };

    static constexpr bool kUseSmemQ = false;

    using ThreadMapQ  = RakedThreadMap<HeadDim, CTA_Q * CTA_H, 8, kWarpCount>;
    using ThreadMapKV = RakedThreadMap<HeadDim, CTA_S, 8, kWarpCount>;

    using ThreadMapKVp = void;

    __device__ static void Sync()
    {
        __syncthreads();
    }

    template<class GmemIterK, class GmemIterV>
    __device__ static void SetSmemKV(GmemIterK& gmem_K, GmemIterV& gmem_V, SharedStorage& storage, bool offset_kv)
    {
        gmem_K.SetSmem(storage.K);
        gmem_V.SetSmem(storage.V);
    }

    __device__ static void TransformQ(T* smem_Q, FragQ& frag_Q)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        __syncwarp();

        SmemAccessor<T, SmemLayoutQ> sQ{smem_Q};
        if constexpr (!kUseSmemQ) {
            // Load from shared memory using LDSM, rearrange to m16n8k16 atom layout
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int k = 0; k < K_K; k += 2) {
                    const int qi = lane_id % 16 * 1 + m * 16 + warp_id * WARP_Q;
                    const int di = lane_id / 16 * 8 + k * 8;
                    ldsm_x4((Array<uint32_t, 4>&)frag_Q[k][m], cast_smem_ptr_to_uint(&sQ(qi, di)));
                }
            }
        }
        else {
            static_assert(!std::is_same_v<T, T>, "not supported");
        }
    }

    struct StateQK {
        SmemAccessor<T, SmemLayoutK> smem_K;

        FragQ frag_Q;
        FragK frag_K;

        __device__ StateQK(SharedStorage& storage, FragQ frag_Q_): smem_K{storage.K}
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
            const int lane_id = threadIdx.x % WARP_SIZE;
            const int offset  = pipe_iter * SmemLayoutK::kSize;
            PRAGMA_UNROLL
            for (int n = 0; n < K_N; n += 4) {  // Load (s32,d8) tiles
                const int s = n * 8 + lane_id;
                const int c = k * 8;
                ldsm_x4((Array<uint32_t, 4>&)frag_K[k][n], cast_smem_ptr_to_uint(&smem_K(s, c, offset)));
            }
        }

        __device__ void Transform(int k) {}
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputeQK(StateQK& state_QK, FragS& frag_S, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        PRAGMA_UNROLL
        for (int k = 0; k < K_K; ++k) {
            if (k < K_K - 1) {
                state_QK.Load(k + 1, offset);
            }
            else {
                ((Preload&&)preload)();
            }
            PRAGMA_UNROLL
            for (int m = 0; m < K_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < K_N; ++n) {
                    const int nn = n ^ 2;
                    mma_m16n8k8_row_col(frag_S[m][nn], state_QK.frag_Q[k][m], state_QK.frag_K[k][nn], frag_S[m][nn]);
                }
            }
        }
    }

    struct StatePV {
        SmemAccessor<T, SmemLayoutV> smem_V;

        FragP frag_P;
        FragV frag_V;

        __device__ StatePV(SharedStorage& storage): smem_V{storage.V} {}

        __device__ void Load(int k, int pipe_iter)
        {
            const int lane_id = threadIdx.x % WARP_SIZE;
            const int offset  = pipe_iter * SmemLayoutV::kSize;
            PRAGMA_UNROLL
            for (int n = 0; n < V_N; n += 4) {  // Load (d32,s8) tiles
                const int si = k * 8 + lane_id % 8;
                const int di = n * 8 + lane_id / 8 * 8;
                ldsm_x4_trans((Array<uint32_t, 4>&)frag_V[k][n], cast_smem_ptr_to_uint(&smem_V(si, di, offset)));
            }
        }

        __device__ void Transform(int k) {}
    };

    template<class Prefetch, class Preload>
    __device__ static void
    ComputePV(StatePV& state_PV, FragO& frag_O, int offset, Prefetch&& prefetch, Preload&& preload)
    {
        PRAGMA_UNROLL
        for (int k = 0; k < V_K; ++k) {
            if (k < V_K - 1) {
                state_PV.Load(k + 1, offset);
            }
            else {
                ((Preload&&)preload)();
            }
            PRAGMA_UNROLL
            for (int m = 0; m < V_M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < V_N; ++n) {
                    mma_m16n8k8_row_col(frag_O[m][n], state_PV.frag_P[m][k], state_PV.frag_V[k][n], frag_O[m][n]);
                }
            }
        }
    }
};

}  // namespace turbomind::attention
