// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/sync.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/thread_map.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

template<Order order>
__host__ __device__ auto make_stride(int ld)
{
    if constexpr (order == Order::kColMajor) {
        return Stride{std::integral_constant<int, 1>{}, ld};
    }
    else if constexpr (order == Order::kRowMajor) {
        return Stride{ld, std::integral_constant<int, 1>{}};
    }
    else {
        return 0;
    }
}

template<class Arch_, class Mainloop, class CtaMap_, bool AlignedM_, bool AlignedN_, bool SplitK_>
struct GemmUniversal {

    // using Impl = typename Mainloop::Impl;
    using Impl = Mainloop;

    using T = typename Impl::Ta;

    using Ta = typename Impl::Ta;
    using Tb = typename Impl::Tb;
    using Tq = typename Impl::Tq;

    using Arch   = Arch_;
    using CtaMap = CtaMap_;

    static constexpr Order kOrderC = Order::kRowMajor;

    static constexpr int CTA_M = Impl::CTA_M;
    static constexpr int CTA_N = Impl::CTA_N;
    static constexpr int CTA_K = Impl::CTA_K;

    static constexpr int G     = 128;
    static constexpr int CTA_G = ceil_div(CTA_K, G);

    static constexpr bool AlignedM = AlignedM_;
    static constexpr bool AlignedN = AlignedN_;

    static constexpr bool SplitK = SplitK_;

    static constexpr int kChunkSizeK = std::max(Impl::G, CTA_K);

    using FragC = typename Impl::FragC;

    static constexpr int WARP_CNT = Impl::WARP_CNT;

    using IteratorA = typename Mainloop::GmemIterA;
    using IteratorB = typename Mainloop::GmemIterB;
    using IteratorQ = typename Mainloop::GmemIterQ;

    using SharedStorage = typename Mainloop::SharedStorage;

    using PtrA = get_pointer_type<Ta>;
    using PtrB = get_pointer_type<Tb>;

    struct Param {
        int    m;
        int    n;
        int    k;
        PtrA   A;
        int    lda;
        PtrB   B;
        int    ldb;
        Tq*    Q;
        int    ldq;
        T*     C;
        int    ldc;
        int    log_tile;
        int3   tiled_shape;
        float* partial_C;  // (k, m, n)
        int*   locks;      // (m/cta_m, n/cta_n, k)
    };

    __device__ void operator()(const Param& param, const CtaMap& cta_map, char* smem_buf)
    {
        const auto tile_offset = CtaMap::get_tile_offset(param.log_tile);

        const auto& tiled_shape = param.tiled_shape;

        const int chunk_cnt = (param.k + kChunkSizeK - 1) / kChunkSizeK;

        const int chunk_per_split = (chunk_cnt + tiled_shape.z - 1) / tiled_shape.z;

        const int offset_k = chunk_per_split * kChunkSizeK * tile_offset.z;

        const int gemm_k_size = std::min(offset_k + chunk_per_split * kChunkSizeK, param.k) - offset_k;

        const int offset_m = tile_offset.x * CTA_M;
        const int offset_n = tile_offset.y * CTA_N;

        if (offset_m >= param.m || offset_n >= param.n || offset_k >= param.k) {  // empty tile
            return;
        }

        const int end_m = min(CTA_M, param.m - offset_m);
        const int end_n = min(CTA_N, param.n - offset_n);

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        FragC frag_C{};

        int tile_iter = (gemm_k_size + CTA_K - 1) / CTA_K;

        auto layout_a = make_stride<Impl::OperandA::kOrder>(param.lda);
        auto layout_b = make_stride<Impl::OperandB::kOrder>(param.ldb);
        auto layout_q = make_stride<Impl::OperandQ::kOrder>(param.ldq);

        IteratorA gmem_A{
            param.A + layout_a(offset_m, offset_k),  // ptr
            param.lda,                               // stride s
            layout_a(0, CTA_K),                      // stride k
            mk2cs<Impl::OperandA::kOrder == Order::kRowMajor>(end_m, CTA_K),
        };

        IteratorB gmem_B{
            param.B + layout_b(offset_k, offset_n),  // ptr
            param.ldb,                               // stride s
            layout_b(CTA_K, 0),                      // stride k
            mk2cs<Impl::OperandB::kOrder == Order::kColMajor>(end_n, CTA_K),
        };

        IteratorQ gmem_Q{
            param.Q + layout_q(offset_k, offset_n),  // ptr
            param.ldq,                               // stride s
            layout_q(CTA_G, 0),                      // stride k
            int2{end_n, CTA_G},
        };

        Mainloop mainloop{};

        mainloop(gmem_A, gmem_B, gmem_Q, frag_C, tile_iter, storage);

        if (!SplitK || tiled_shape.z == 1) {
            StoreC(frag_C, offset_m, offset_n, end_m, end_n, param, storage);
        }
        else {
            // store partial
            Impl::template StoreC<float>(frag_C, storage, [&](int mi, int ni, const auto& vec) {
                const int idx = tile_offset.z * param.m * param.n + (offset_m + mi) * param.n + offset_n + ni;
                if (check_m(mi, end_m) && check_n(ni, end_n)) {
                    Store(&param.partial_C[idx], cast<float>(vec));
                }
            });

            int* locks = &param.locks[(tile_offset.x * tiled_shape.y + tile_offset.y) * tiled_shape.z];

            const int thread_idx = threadIdx.x;

            if (offset_k + gemm_k_size < param.k) {  // set flag if not last split
                sem_post(&locks[tile_offset.z], 1, thread_idx == 0);
            }
            else {  // offset_k + gemm_k_size == param.k => last split
                sem_wait_many(&locks[thread_idx], tile_offset.z, thread_idx < tile_offset.z);

                Reduce(tile_offset, param, end_m, end_n);

                if (thread_idx <= tile_offset.z) {
                    locks[thread_idx] = 0;
                }
            }
        }
    }

    template<bool k_major>
    __device__ static constexpr int2 mk2cs(int m, int k)
    {
        return k_major ? int2{k, m} : int2{m, k};
    }

    __device__ static constexpr bool check_m(int mi, int end_m)
    {
        if constexpr (AlignedM) {
            return 1;
        }
        else {
            return mi < end_m;
        }
    }

    __device__ static constexpr bool check_n(int ni, int end_n)
    {
        if constexpr (AlignedN) {
            return 1;
        }
        else {
            return ni < end_n;
        }
    }

    __device__ void
    StoreC(FragC& frag_C, int offset_m, int offset_n, int end_m, int end_n, const Param& param, SharedStorage& storage)
    {
        static_assert(kOrderC == Order::kRowMajor);

        const auto layout_c = make_stride<kOrderC>(param.ldc);

        Impl::template StoreC<T>(frag_C, storage, [&](int mi, int ni, const auto& vec) {
            if (check_m(mi, end_m) && check_n(ni, end_n)) {
                Store(param.C + layout_c(offset_m + mi, offset_n + ni), cast<T>(vec));
            }
        });
    }

    __device__ void Reduce(const int3& tile_offset, const Param& param, int end_m, int end_n)
    {
        constexpr int kVecSize = std::min(4, std::max(1, (CTA_K * CTA_M) / (WARP_CNT * WARP_SIZE)));

        using Map = gemm::ThreadMap<CTA_N, CTA_M, kVecSize, WARP_CNT>;
        using Vec = Array<float, kVecSize>;

        constexpr int S = Map::kIterS;
        constexpr int C = Map::kIterC;

        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int offset_m = tile_offset.x * CTA_M;
        const int offset_n = tile_offset.y * CTA_N;

        const int2 d = Map::get_offset(warp_id, lane_id);

        Vec accu_C[S][C]{};

        for (int k = 0; k < param.tiled_shape.z; ++k) {
            PRAGMA_UNROLL
            for (int s = 0; s < S; ++s) {
                Vec       frag_C[1][C];
                const int mi = d.y + s * Map::kDeltaS;
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    const int  ni    = d.x + c * Map::kDeltaC;
                    const int  index = k * param.m * param.n + (offset_m + mi) * param.n + offset_n + ni;
                    const bool mask  = check_m(mi, end_m) && check_n(ni, end_n);
                    if (mask) {
                        Ldg(frag_C[0][c], &param.partial_C[index]);
                    }
                }

                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    using namespace ops;
                    accu_C[s][c] = accu_C[s][c] + frag_C[0][c];
                }
            }
        }

        PRAGMA_UNROLL
        for (int s = 0; s < S; ++s) {
            const int mi = d.y + s * Map::kDeltaS;
            PRAGMA_UNROLL
            for (int c = 0; c < C; ++c) {
                const int  ni    = d.x + c * Map::kDeltaC;
                const int  index = (offset_m + mi) * param.n + offset_n + ni;
                const bool mask  = check_m(mi, end_m) && check_n(ni, end_n);
                if (mask)
                    Store(&param.C[index], cast<T>(accu_C[s][c]));
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