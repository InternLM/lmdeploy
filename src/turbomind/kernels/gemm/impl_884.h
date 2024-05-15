// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/mma.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/impl.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/thread_map.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <type_traits>

namespace turbomind::gemm {

struct MMA_884 {};

template<class T_,
         class Tb_,
         class Tq_,
         int CTA_M_,
         int CTA_N_,
         int CTA_K_,
         int WARP_M_,
         int WARP_N_,
         int WARP_K_,
         int Stages_,
         int Flag_>
struct Impl<MMA_884, T_, Tb_, Tq_, CTA_M_, CTA_N_, CTA_K_, WARP_M_, WARP_N_, WARP_K_, Stages_, Flag_> {
    using T  = T_;
    using Tb = Tb_;
    using Tq = Tq_;

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

    static constexpr int WARP_CNT_MN = WARP_CNT_M * WARP_CNT_N;

    static constexpr int WARP_CNT = WARP_CNT_MN * WARP_CNT_K;

    static constexpr int OP_M = 16;
    static constexpr int OP_N = 16;
    static constexpr int OP_K = 4;

    static constexpr int ITER_M = WARP_M / OP_M;
    static constexpr int ITER_N = WARP_N / OP_N;
    static constexpr int ITER_K = WARP_K / OP_K;

    using FragA = Array<T, 4>[ITER_K][ITER_M];  // m1,k4
    using FragB = Array<T, 4>[ITER_K][ITER_N];  // n1,k4
    using FragC = Array<float, 8>[ITER_M][ITER_N];

    using SmemLayoutA = SmemLayoutV2<CTA_M, CTA_K, CTA_M, 8>;
    using ThreadMapA  = gemm::ThreadMap<CTA_K, CTA_M, 4, WARP_CNT>;

    using SmemLayoutB = SmemLayoutV2<CTA_N, CTA_K, CTA_N, 8>;
    using ThreadMapB  = gemm::ThreadMap<CTA_K, CTA_N, 4, WARP_CNT>;

    using SmemLayoutQ             = SmemLayoutB;
    using ThreadMapQ              = gemm::ThreadMap<CTA_N, CTA_K, 4, WARP_CNT>;
    static constexpr int G        = 128;
    static constexpr int CTA_G    = ceil_div(CTA_K, G);
    static constexpr int G_CTA    = ceil_div(G, CTA_K);
    static constexpr int kPackedN = 1;

    static constexpr auto LayoutA = LayoutType::kRowMajor;  // (m, k)
    static constexpr auto LayoutB = LayoutType::kColMajor;  // (n, k)
    static constexpr auto LayoutQ = LayoutType::kRowMajor;
    static constexpr auto LayoutC = LayoutType::kRowMajor;  // (m, n)

    struct SharedStorage {
        __align__(16) T A[Stages_ * SmemLayoutA::kSize];
        __align__(16) Array<T, Stages_ * SmemLayoutB::kSize> B;
    };

    struct StateA {

        SmemAccessor<T, SmemLayoutA> smem_A;
        T*                           data;
        int                          offset = 0;
        FragA                        frag_A;

        __device__ StateA(SharedStorage& storage): smem_A{storage.A}
        {
            data = storage.A;
        }

        __device__ void Load(int k, int)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;

            const int offset_s = warp_id_m(warp_id) * WARP_M;
            const int offset_c = warp_id_k(warp_id) * WARP_K;
            //                        4                3               01
            const int thr_s = lane_id / 16 * 4 + (lane_id & 8) + lane_id % 4;
            const int thr_c = 0;

            SmemAccessor<T, SmemLayoutA> smem{data};

            PRAGMA_UNROLL
            for (int m = 0; m < ITER_M; ++m) {
                const int s = offset_s + thr_s + m * OP_M;
                const int c = offset_c + thr_c + k * OP_K;
                Lds(frag_A[k][m], &smem(s, c));
            }
        }

        __device__ void Advance()
        {
            offset += SmemLayoutA::kSize;
            data += SmemLayoutA::kSize;
            if (offset == Stages * SmemLayoutA::kSize) {
                offset = 0;
                data -= Stages * SmemLayoutA::kSize;
            }
        }
    };

    struct StateQ {
        Tq*             data{};
        bool            counting{};
        __device__      StateQ(SharedStorage& storage) {}
        __device__ void Load(int k, int) {}
        __device__ void Advance() {}
    };

    struct StateB {

        T*      data;
        int     offset = 0;
        FragB   frag_B;
        StateQ* state_Q;

        __device__ StateB(SharedStorage& storage)  //: smem_B{storage.B.data()}
        {
            data = storage.B.data();
        }

        __device__ void Load(int k, int)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;

            const int offset_s = warp_id_n(warp_id) * WARP_N;
            const int offset_c = warp_id_k(warp_id) * WARP_K;
            //                     4                     2                 01
            const int thr_s = lane_id / 16 * 4 + (lane_id & 4) * 2 + lane_id % 4;
            const int thr_c = 0;

            SmemAccessor<T, SmemLayoutB> smem{data};

            PRAGMA_UNROLL
            for (int n = 0; n < ITER_N; ++n) {
                const int s = offset_s + thr_s + n * OP_N;
                const int c = offset_c + thr_c + k * OP_K;
                Lds(frag_B[k][n], &smem(s, c));
            }
        }

        __device__ void Transform(int) {}

        __device__ void Advance()
        {
            offset += SmemLayoutB::kSize;
            data += SmemLayoutB::kSize;
            if (offset == Stages * SmemLayoutB::kSize) {
                offset = 0;
                data -= Stages * SmemLayoutB::kSize;
            }
        }
    };

    template<class Prefetch, class Advance>
    __device__ static void
    Compute(StateA& state_A, StateB& state_B, FragC& frag_C, int pipe_iter, Prefetch&& prefetch, Advance&& advance)
    {
        static_assert(ITER_K > 1);

        PRAGMA_UNROLL
        for (int k = 0; k < ITER_K; ++k) {
            state_A.Load((k + 1) % ITER_K, pipe_iter);
            state_B.Load((k + 1) % ITER_K, pipe_iter);

            PRAGMA_UNROLL
            for (int n = 0; n < ITER_N; ++n) {
                PRAGMA_UNROLL
                for (int m = 0; m < ITER_M; ++m) {
                    mma_m8n8k4_row_col(frag_C[m][n], state_A.frag_A[k][m], state_B.frag_B[k][n], frag_C[m][n]);
                }
            }
            ((Prefetch&&)prefetch)((k + 1) % ITER_K);
            if (k + 1 == ITER_K - 1) {
                ((Advance&&)advance)();
            }

            // ! Transform of k must come before prefetching k + 1
            // state_B.Transform((k + 1) % ITER_K);
        }
    }

    template<class Tc, class Func>
    __device__ static void StoreC(FragC& frag_C, SharedStorage& storage, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_idx_m = warp_id_m(warp_id);
        const int warp_idx_n = warp_id_n(warp_id);
        // const int warp_idx_k = warp_id_k(warp_id);

        const int warp_offset_m = warp_idx_m * WARP_M;
        const int warp_offset_n = warp_idx_n * WARP_N;

        static_assert(WARP_CNT_K == 1);

        PRAGMA_UNROLL
        for (int m = 0; m < ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < ITER_N; ++n) {
                PRAGMA_UNROLL
                for (int nn = 0; nn < 2; ++nn) {
                    PRAGMA_UNROLL
                    for (int mm = 0; mm < 2; ++mm) {
                        const int mi = m * OP_M + (lane_id & 8) + (lane_id & 1) + lane_id / 16 * 4 + mm * 2;
                        const int ni = n * OP_N + (lane_id & 4) * 2 + (lane_id & 2) + nn * 4;
                        ((Func&&)func)(warp_offset_m + mi,
                                       warp_offset_n + ni,
                                       cast<Tc>((Array<float, 2>&)frag_C[m][n][nn * 4 + mm * 2]));
                    }
                }
            }
        }
    }

    __device__ static int warp_id_m(int warp_id)
    {
        if constexpr (WARP_CNT_M == 1) {
            return 0;
        }
        else {
            return warp_id % WARP_CNT_M;
        }
    }

    __device__ static int warp_id_n(int warp_id)
    {
        if constexpr (WARP_CNT_N == 1) {
            return 0;
        }
        else {
            return warp_id / WARP_CNT_M % WARP_CNT_N;
        }
    }

    __device__ static int warp_id_k(int warp_id)
    {
        if constexpr (WARP_CNT_K == 1) {
            return 0;
        }
        else {
            return warp_id / WARP_CNT_M / WARP_CNT_N;
        }
    }
};

}  // namespace turbomind::gemm