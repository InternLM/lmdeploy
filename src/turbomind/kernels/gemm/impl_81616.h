// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/mma.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/impl.h"
#include "src/turbomind/kernels/gemm/thread_map.h"

namespace turbomind::gemm {

template<class T_,
         class Tx_,
         class Tw_,
         int CTA_M_,
         int CTA_N_,
         int CTA_K_,
         int WARP_M_,
         int WARP_N_,
         int WARP_K_,
         int Stages_>
struct Impl<MMA_81616, T_, Tx_, Tw_, CTA_M_, CTA_N_, CTA_K_, WARP_M_, WARP_N_, WARP_K_, Stages_> {

    using T  = T_;
    using Tx = Tx_;
    using Tw = Tw_;

    static constexpr int CTA_M = CTA_M_;
    static constexpr int CTA_N = CTA_N_;
    static constexpr int CTA_K = CTA_K_;

    static constexpr int WARP_M = WARP_M_;
    static constexpr int WARP_N = WARP_N_;
    static constexpr int WARP_K = WARP_K_;

    static constexpr int Stages = Stages_;

    static constexpr int WARP_CNT_M = CTA_M / WARP_M;
    static constexpr int WARP_CNT_N = CTA_N / WARP_N;
    static constexpr int WARP_CNT_K = CTA_K / WARP_K;

    static constexpr int WARP_CNT = WARP_CNT_M * WARP_CNT_N * WARP_CNT_K;

    // - M batch size
    // - N output dims
    // - K input dims

    static constexpr int OP_M = 8;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 16;

    static constexpr int ITER_M = WARP_M / OP_M;
    static constexpr int ITER_N = WARP_N / OP_N;
    static constexpr int ITER_K = WARP_K / OP_K;

    using FragA = Array<T, 4>[ITER_K][ITER_M];      // {m8,k4}, [iK,iM], (k2,k2)
                                                    //   1  2    16  8     8  1
    using FragB = Array<T, 8>[ITER_K][ITER_N];      // {n8,k4}, [iK,iN], (k2,n2,k2)
                                                    //   1  2    16 16     8  8  1
    using FragC = Array<float, 4>[ITER_M][ITER_N];  // {n8,m4}, [iM,iN], (n2,m2)
                                                    //   1  2     8 16     8  1

    using SmemLayoutA = SmemLayoutV2<CTA_M, CTA_K, 32, 32, Swizzle<3, 3, 2>>;
    using SmemLayoutB = SmemLayoutV2<CTA_N, CTA_K, 32, 32, Swizzle<3, 3, 2>>;

    // using SmemLayoutA = SmemLayoutV2<CTA_M, CTA_K + 8, CTA_M, CTA_K + 8, Identity>;
    // using SmemLayoutB = SmemLayoutV2<CTA_N, CTA_K + 8, CTA_M, CTA_K + 8, Identity>;

    using ThreadMapA = gemm::ThreadMap<CTA_K, CTA_M, 8, WARP_CNT>;
    using ThreadMapB = gemm::ThreadMap<CTA_K, CTA_N, 8, WARP_CNT>;

    union SharedStorage {
        struct {
            __align__(16) T A[Stages_ * SmemLayoutA::kSize];
            __align__(16) T B[Stages_ * SmemLayoutB::kSize];
        };
    };

    template<class GmemIterA, class GmemIterB>
    __device__ static void SetSmem(GmemIterA& gmem_A, GmemIterB& gmem_B, SharedStorage& storage)
    {
        gmem_A.SetSmem(storage.A);
        gmem_B.SetSmem(storage.B);
    }

    struct StateA {
        SmemAccessor<T, SmemLayoutA> smem_A;
        FragA                        frag_A;

        __device__ StateA(SharedStorage& storage): smem_A{storage.A} {}

        __device__ void Load(int k, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;

            const int warp_offset_m = warp_id_m(warp_id) * WARP_M;

            const int offset = pipe_iter * SmemLayoutA::kSize;

            if constexpr (ITER_M == 1) {
                const int offset_s = lane_id % 8 + warp_offset_m;
                const int offset_c = lane_id / 8 * 8;
                if constexpr (ITER_K % 2 == 0) {
                    if (k % 2 == 0) {
                        const int s = offset_s;
                        const int c = offset_c + k * 16;
                        ldsm_x4((Array<uint32_t, 4>&)frag_A[k][0], cast_smem_ptr_to_uint(&smem_A(s, c, offset)));
                    }
                }
                else {
                    const int s = offset_s;
                    const int c = offset_c % 16 + k * 16;
                    ldsm_x2((Array<uint32_t, 2>&)frag_A[k][0], cast_smem_ptr_to_uint(&smem_A(s, c, offset)));
                }
            }
            else {
                const int offset_s = lane_id % 8 + lane_id / 16 * 8 + warp_offset_m;
                const int offset_c = lane_id / 8 * 8 % 16;
                static_assert(ITER_M % 2 == 0);
                PRAGMA_UNROLL
                for (int m = 0; m < ITER_M; m += 2) {
                    const int s = m * 8 + offset_s;
                    const int c = k * 16 + offset_c;
                    ldsm_x4((Array<uint32_t, 4>&)frag_A[k][m], cast_smem_ptr_to_uint(&smem_A(s, c, offset)));
                }
            }
        }
    };

    struct StateB {
        SmemAccessor<T, SmemLayoutB> smem_B;
        FragB                        frag_B;

        __device__ StateB(SharedStorage& storage): smem_B{storage.B} {}

        __device__ void Load(int k, int pipe_iter)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;

            const int warp_offset_n = warp_id_n(warp_id) * WARP_N;

            const int offset_s = lane_id % 16 + warp_offset_n;
            const int offset_c = lane_id / 16 * 8;
            const int offset   = pipe_iter * SmemLayoutB::kSize;
            PRAGMA_UNROLL
            for (int n = 0; n < ITER_N; ++n) {
                const int s = n * 16 + offset_s;
                const int c = k * 16 + offset_c;
                ldsm_x4((Array<uint32_t, 4>&)frag_B[k][n], cast_smem_ptr_to_uint(&smem_B(s, c, offset)));
            }
        }
    };

    template<class Prefetch, class Preload>
    __device__ static void
    Compute(StateA& state_A, StateB& state_B, FragC& frag_C, int pipe_iter, Prefetch&& prefetch, Preload&& preload)
    {
        static_assert(ITER_K > 1);

        PRAGMA_UNROLL
        for (int k = 0; k < ITER_K; ++k) {
            if (k < ITER_K - 1) {
                state_A.Load(k + 1, pipe_iter);
                state_B.Load(k + 1, pipe_iter);
            }
            else {
                ((Preload&&)preload)();
            }
            PRAGMA_UNROLL
            for (int n = 0; n < ITER_N; ++n) {
                PRAGMA_UNROLL
                for (int m = 0; m < ITER_M; ++m) {
                    mma_m16n8k16_row_col(frag_C[m][n], state_B.frag_B[k][n], state_A.frag_A[k][m], frag_C[m][n]);
                }
            }
            if (k < ITER_K - 1) {
                ((Prefetch&&)prefetch)(k);
            }
            if (k == ITER_K - 2) {
                ((Prefetch&&)prefetch)(ITER_K - 1);
            }
        }
    }

    template<class Func>
    __device__ static void StoreC(FragC& frag_C, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_offset_m = warp_id_m(warp_id) * WARP_M;
        const int warp_offset_n = warp_id_n(warp_id) * WARP_N;

        PRAGMA_UNROLL
        for (int m = 0; m < ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < ITER_N; ++n) {
                PRAGMA_UNROLL
                for (int nn = 0; nn < 2; ++nn) {
                    auto tmp = cast<T>((Array<float, 2>&)frag_C[m][n][nn * 2]);
                    // {n8,m4},[iM,iN],(n2,m2) -> {m8,n4},[iM,iN],(n2,n2)
                    //   1  2    8 16    8  1       1  2    8 16    8  1
                    (uint32_t&)tmp = transpose_m8n8_b16((uint32_t&)tmp);
                    const int mi   = m * OP_M + lane_id / 4 * 1;
                    const int ni   = n * OP_N + lane_id % 4 * 2 + nn * 8;
                    ((Func&&)func)(warp_offset_m + mi, warp_offset_n + ni, tmp);
                }
            }
        }
    }

    __device__ static int warp_id_m(int warp_id)
    {
        return warp_id % WARP_CNT_M;
    }

    __device__ static int warp_id_n(int warp_id)
    {
        return warp_id / WARP_CNT_M % WARP_CNT_N;
    }

    __device__ static int warp_id_k(int warp_id)
    {
        return warp_id / WARP_CNT_M / WARP_CNT_N;
    }
};

}  // namespace turbomind::gemm