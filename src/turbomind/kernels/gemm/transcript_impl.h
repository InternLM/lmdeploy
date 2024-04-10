// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/sync.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include <cuda_pipeline_primitives.h>

namespace turbomind::gemm {

template<class Arch_, class Gemm, class DataIter, class CtaMap_>
struct Transcript {

    using T = typename Gemm::T;

    using Arch   = Arch_;
    using CtaMap = CtaMap_;

    static constexpr int CTA_M = Gemm::CTA_M;
    static constexpr int CTA_N = Gemm::CTA_N;
    static constexpr int CTA_K = Gemm::CTA_K;

    static constexpr int WARP_CNT = Gemm::WARP_CNT;

    using ThreadMapB  = typename Gemm::ThreadMapB;
    using SmemLayoutB = typename Gemm::SmemLayoutB;

    using GmemIterB = GmemIteratorSm80<T, ThreadMapB, SmemLayoutB, 1>;

    struct SharedStorage {
        __align__(16) T B[Gemm::SmemLayoutB::kSize];
    };

    // row.col.row
    struct Param {
        const T* A;  // x (m,k)
        const T* B;  // W (n,k)
        T*       C;
        int      m;
        int      n;
        int      k;
    };

    __device__ void operator()(const Param& param, char* smem_buf)
    {
        const auto [cta_idx_m, cta_idx_n, split_idx] =  //
            CtaMap::get_tile_offset(0);

        const auto [cta_cnt_m, cta_cnt_n, split_cnt] =
            CtaMap::get_tiled_shape(param.m, param.n, param.k, CTA_M, CTA_N, 1);

        // [n, k] -> [cta_cnt_k, cta_cnt_n, mma_cnt_k, mma_cnt_n, lane_id, x, fragment]

        constexpr int MMA_CNT_K = CTA_K / Gemm::OP_K;
        constexpr int MMA_CNT_N = CTA_N / Gemm::OP_N;
        constexpr int CTA_SIZE  = CTA_K * CTA_N;
        constexpr int FRAG_SIZE = 8;

        static_assert(CTA_SIZE == MMA_CNT_K * MMA_CNT_N * WARP_SIZE * FRAG_SIZE);

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        const int cta_cnt_k = (param.k + CTA_K - 1) / CTA_K;

        // printf("cta_cnt_k=%d cta_cnt_n=%d mma_cnt_k=%d mma_cnt_n=%d\n", cta_cnt_n, cta_cnt_k, MMA_CNT_K, MMA_CNT_N);

        DataIter data_iter{param, 0, cta_idx_n * CTA_N, 0, param.k};

        GmemIterB gmem_B{param.k};

        Gemm::SetSmemB(gmem_B, storage);

        gmem_B.ClearSmem(0);

        typename Gemm::StateB state_B{storage};

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_id_m = Gemm::warp_id_m(warp_id);
        const int warp_id_n = Gemm::warp_id_n(warp_id);

        int cta_idx_k = 0;

        // printf("warp_id=%d, warp_id_n=%d\n", warp_id, warp_id_n);

        PRAGMA_NO_UNROLL
        for (; cta_idx_k < cta_cnt_k; ++cta_idx_k) {
            gmem_B.Prefetch(data_iter, 0);
            __pipeline_commit();
            ++data_iter;

            __pipeline_wait_prior(0);
            __syncthreads();

            T* cta_C = &param.C[(cta_idx_k * cta_cnt_n + cta_idx_n) * CTA_N * CTA_K];
            PRAGMA_UNROLL
            for (int k = 0; k < Gemm::ITER_K; ++k) {
                state_B.Load(k, 0);
                PRAGMA_UNROLL
                for (int n = 0; n < Gemm::ITER_N; ++n) {
                    const int mma_idx_k = k;
                    const int mma_idx_n = n + warp_id_n * Gemm::ITER_N;
                    // mma fragment ptr for the warp
                    T* C = cta_C + (mma_idx_k * MMA_CNT_N + mma_idx_n) * WARP_SIZE * FRAG_SIZE;
                    if (warp_id_m == 0) {
                        const int frag_idx = (cta_idx_k * cta_cnt_n + cta_idx_n) * MMA_CNT_N * MMA_CNT_K
                                             + mma_idx_k * MMA_CNT_N + mma_idx_n;
                        // if (lane_id == 0) {
                        //     printf("frag_idx = %6d\n", frag_idx);
                        // }
                        Store(&C[lane_id * FRAG_SIZE], state_B.frag_B[k][n]);
                        // PRAGMA_UNROLL
                        // for (int c = 0; c < state_B.frag_B[k][n].size(); ++c) {
                        //     printf("%2d %2d %2d %2d %2d %f\n",
                        //            (int)cta_idx_k,
                        //            (int)cta_idx_n,
                        //            (int)mma_idx_k,
                        //            (int)mma_idx_n,
                        //            (int)lane_id,
                        //            (float)state_B.frag_B[k][n][c]);
                        // }
                    }
                }
            }
        }
    }
};

extern __shared__ char smem_buf[];

template<class Kernel>
__global__ void transcript_kernel(typename Kernel::Param params)
{
    Kernel kernel;
    kernel(params, smem_buf);
}

}  // namespace turbomind::gemm