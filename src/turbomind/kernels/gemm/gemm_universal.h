// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/sync.h"
#include "src/turbomind/kernels/gemm/thread_map.h"

namespace turbomind::gemm {

template<class Arch_, class Mainloop, class CtaMap_>
struct GemmUniversal {

    using Impl = typename Mainloop::Impl;

    using T  = typename Impl::T;
    using Tb = typename Impl::Tb;
    using Tq = typename Impl::Tq;

    using Arch   = Arch_;
    using CtaMap = CtaMap_;

    static constexpr int CTA_M = Impl::CTA_M;
    static constexpr int CTA_N = Impl::CTA_N;
    static constexpr int CTA_K = Impl::CTA_K;
    static constexpr int CTA_G = Impl::CTA_G;

    using FragC = typename Impl::FragC;

    static constexpr int WARP_CNT = Impl::WARP_CNT;

    using SharedStorage = typename Mainloop::SharedStorage;

    // row.col.row
    struct Param {
        T*                   A;  // x (m  ,k)
        get_pointer_type<Tb> B;  // W (n  ,k)
        Tq*                  Q;  //   (k/g,n)
        T*                   C;  //   (m  ,n)
        int                  m;
        int                  n;
        int                  k;
        int                  log_tile;
        int3                 tiled_shape;
        float*               partial_C;  // (k, m, n)
        int*                 locks;      // (m/cta_m, n/cta_n, k)
    };

    __device__ void operator()(const Param& param, const CtaMap& cta_map, char* smem_buf)
    {
        const auto tile_offset = CtaMap::get_tile_offset(param.log_tile);

        const auto& tiled_shape = param.tiled_shape;

        constexpr int chunk_size = std::max(Impl::G, CTA_K);
        const int     chunk_cnt  = (param.k + chunk_size - 1) / chunk_size;

        const int chunk_per_split = (chunk_cnt + tiled_shape.z - 1) / tiled_shape.z;

        const int cta_offset_k = chunk_per_split * chunk_size * tile_offset.z;
        const int gemm_k_size  = std::min(cta_offset_k + chunk_per_split * chunk_size, param.k) - cta_offset_k;

        // const int gemm_k_size = param.k / tiled_shape.z;

        const int m_idx = tile_offset.x * CTA_M;
        const int n_idx = tile_offset.y * CTA_N;
        // const int k_idx = tile_offset.z * gemm_k_size;

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        FragC frag_C{};

        int tile_iter = (gemm_k_size + CTA_K - 1) / CTA_K;

        typename Mainloop::GmemIterA gmem_A{param.A + m_idx * param.k + cta_offset_k,                   // ptr
                                            param.k,                                                    // stride s
                                            CTA_K};                                                     // stride k
        typename Mainloop::GmemIterB gmem_B{param.B + n_idx * param.k + cta_offset_k * Impl::kPackedN,  // ptr
                                            param.k * Impl::kPackedN,                                   // stride s
                                            CTA_K * Impl::kPackedN};                                    // stride k
        typename Mainloop::GmemIterQ gmem_Q{
            param.Q + n_idx + cta_offset_k / Impl::G * param.n,  // ptr
            param.n,                                             // stride s, not very useful
            CTA_G * param.n                                      // stride k
        };

        Mainloop mainloop{};

        mainloop(gmem_A, gmem_B, gmem_Q, frag_C, tile_iter, storage);

        if (tiled_shape.z == 1) {
            StoreC(frag_C, m_idx, n_idx, param, storage);
        }
        else {

            // store partial
            Impl::StoreC(frag_C, storage, [&](int mi, int ni, const auto& vec) {
                Store(&param.partial_C[tile_offset.z * param.m * param.n + (m_idx + mi) * param.n + n_idx + ni],
                      cast<float>(vec));
            });

            int* locks = &param.locks[(tile_offset.x * tiled_shape.y + tile_offset.y) * tiled_shape.z];

            const int thread_idx = threadIdx.x;

            // set flag if not last split
            if (tile_offset.z < tiled_shape.z - 1) {
                sem_post(&locks[tile_offset.z], 1, thread_idx == 0);
            }
            else {
                sem_wait_many(&locks[thread_idx], tiled_shape.z - 1, thread_idx < tiled_shape.z - 1);

                Reduce(tile_offset, param);

                if (thread_idx < tiled_shape.z) {
                    locks[thread_idx] = 0;
                }
            }
        }
    }

    __device__ void StoreC(FragC& frag_C, int offset_m, int offset_n, const Param& param, SharedStorage& storage)
    {
        Impl::StoreC(frag_C, storage, [&](int mi, int ni, const auto& vec) {
            Store(&param.C[(offset_m + mi) * param.n + offset_n + ni], cast<half>(vec));  //
        });
    }

    __device__ void Reduce(const int3& tile_offset, const Param& param)
    {
        constexpr int kVecSize = std::min(4, std::max(1, (CTA_K * CTA_M) / (WARP_CNT * WARP_SIZE)));

        using Map = gemm::ThreadMap<CTA_N, CTA_M, kVecSize, WARP_CNT>;
        using Vec = Array<float, kVecSize>;

        constexpr int S = Map::kIterS;
        constexpr int C = Map::kIterC;

        Vec accu_C[S][C]{};
        Vec frag_C[S][C];

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int cta_offset_m = tile_offset.x * CTA_M;
        const int cta_offset_n = tile_offset.y * CTA_N;

        const int2 d = Map::get_offset(warp_id, lane_id);

        for (int k = 0; k < param.tiled_shape.z; ++k) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                const int mi = d.y + s * Map::kDeltaS;
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    const int ni = d.x + c * Map::kDeltaC;
                    Ldg(frag_C[s][c],
                        &param.partial_C[k * param.m * param.n + (cta_offset_m + mi) * param.n + cta_offset_n + ni]);
                }
            }

            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    using namespace ops;
                    accu_C[s][c] = accu_C[s][c] + frag_C[s][c];
                }
            }
        }

        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            const int mi = d.y + s * Map::kDeltaS;
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                const int ni = d.x + c * Map::kDeltaC;
                Store(&param.C[(cta_offset_m + mi) * param.n + cta_offset_n + ni], cast<T>(accu_C[s][c]));
            }
        }
    }
};

extern __shared__ char smem_buf[];

template<class Kernel>
__global__ void gemm_kernel(typename Kernel::Param params, typename Kernel::CtaMap cta_map)
{
    Kernel kernel;
    kernel(params, cta_map, smem_buf);
}

}  // namespace turbomind::gemm