// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/sync.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include <cuda_pipeline_primitives.h>

namespace turbomind::gemm {

template<class Arch_, class Gemm, class Gemm1, class Converter, class CtaMap_>
struct Transcript {

    using T  = typename Gemm::Tb;
    using Tq = typename Gemm::Tq;
    using T1 = typename Gemm1::Tb;

    using Arch   = Arch_;
    using CtaMap = CtaMap_;

    static constexpr int CTA_M = Gemm::CTA_M;
    static constexpr int CTA_N = Gemm::CTA_N;
    static constexpr int CTA_K = Gemm::CTA_K;
    static constexpr int CTA_G = Gemm::CTA_G;

    static constexpr int WARP_CNT = Gemm::WARP_CNT;

    using ThreadMapB = typename Gemm::ThreadMapB;
    using ThreadMapQ = typename Gemm::ThreadMapB;

    using SmemLayoutB = typename Gemm::SmemLayoutB;
    using SmemLayoutQ = typename Gemm::SmemLayoutQ;

    using GmemIterB = GmemIteratorSm80<T, ThreadMapB, SmemLayoutB, 100>;
    using GemmIterQ = GmemIteratorSm80<T, ThreadMapQ, SmemLayoutQ, 101>;

    struct SharedStorage {
        __align__(16) Array<T, Gemm::SmemLayoutB::kSize> B;
        __align__(16) Array<Tq, Gemm::SmemLayoutQ::kSize> Q;
    };

    static constexpr int MMA_CNT_K = CTA_K / Gemm::OP_K;
    static constexpr int MMA_CNT_N = CTA_N / Gemm::OP_N;

    static constexpr int P_N = Gemm1::P_N;
    static constexpr int P_K = Gemm1::P_K;

    static_assert(CTA_K * CTA_N == MMA_CNT_K * MMA_CNT_N * WARP_SIZE * 8);

    // static constexpr int P_Q_N = Gemm1::P_Q_N;
    // static constexpr int P_Q_K = Gemm1::P_Q_K;
    // static constexpr int G     = Gemm1::G;

    // row.col.row
    struct Param {
        const T*             A;
        const T*             B;
        const Tq*            Q;
        get_pointer_type<T1> C;
        T*                   D;
        int                  m;
        int                  n;
        int                  k;
    };

    __device__ void operator()(const Param& param, char* smem_buf)
    {
        const auto [cta_idx_m, cta_idx_n, split_idx] =  //
            CtaMap::get_tile_offset(0);

        const auto [cta_cnt_m, cta_cnt_n, split_cnt] =
            CtaMap::get_tiled_shape(param.m, param.n, param.k, CTA_M, CTA_N, 1);

        const int cta_cnt_k = (param.k + CTA_K - 1) / CTA_K;

        // [n, k] -> [packed_n, packed_k, warp_size, p_k, p_n, fragment]

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        const int                  packed_k = cta_cnt_k * MMA_CNT_K / P_K;
        [[maybe_unused]] const int packed_n = cta_cnt_n * MMA_CNT_N / P_N;

        GmemIterB gmem_B{(T*)param.B + cta_idx_n * CTA_N * param.k, param.k, CTA_K};

        typename Gemm::StateB state_B{storage};
        // typename Gemm::StateQ state_Q{storage};

        gmem_B.smem_data_ = state_B.data;
        gmem_B.ClearSmem();

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int warp_id_m = Gemm::warp_id_m(warp_id);
        const int warp_id_n = Gemm::warp_id_n(warp_id);

        int cta_idx_k = 0;

        PRAGMA_NO_UNROLL
        for (; cta_idx_k < cta_cnt_k; ++cta_idx_k) {
            gmem_B.Prefetch(true);
            gmem_B.Advance();
            __pipeline_commit();

            __pipeline_wait_prior(0);
            __syncthreads();  // wait for smem being written

            PRAGMA_UNROLL
            for (int k = 0; k < Gemm::ITER_K; k += P_K) {
                PRAGMA_UNROLL
                for (int p_k = 0; p_k < P_K; ++p_k) {
                    state_B.Load(k + p_k, 0);
                }
                PRAGMA_UNROLL
                for (int n = 0; n < Gemm::ITER_N; n += P_N) {
                    const int    frag_idx_k = cta_idx_k * MMA_CNT_K + k;
                    const int    frag_idx_n = cta_idx_n * MMA_CNT_N + n + warp_id_n * Gemm::ITER_N;
                    const int    pack_idx_k = frag_idx_k / P_K;
                    const int    pack_idx_n = frag_idx_n / P_N;
                    Array<T1, 8> data[P_K][P_N];
                    PRAGMA_UNROLL
                    for (int p_k = 0; p_k < P_K; ++p_k) {
                        PRAGMA_UNROLL
                        for (int p_n = 0; p_n < P_N; ++p_n) {
                            // transform to packed data (quantization & permutation)
                            Converter converter{};
                            data[p_k][p_n] = converter(state_B.frag_B[k + p_k][n + p_n]);
                        }
                    }
                    constexpr int kAccessSize = 8 * P_K * P_N;
                    static_assert(sizeof(data) == 16);
                    if (warp_id_m == 0) {
                        // mma fragment ptr for the warp
                        auto C = param.C + ((pack_idx_n * packed_k + pack_idx_k) * WARP_SIZE + lane_id) * kAccessSize;
                        Store((T1*)C, (Array<T1, kAccessSize>&)data);
                    }
                }
            }

            __syncthreads();  // wait for smem being read
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