// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/core/mma.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/simt.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/thread_map.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

template<class MMA_Atom_, class MMA_Map_>
struct Tiled_MMA_v2 {
    using Atom = MMA_Atom_;
    using Map  = MMA_Map_;

    static constexpr int M = Map::M;
    static constexpr int N = Map::N;
    static constexpr int K = Map::K;

    static constexpr int kGroupCount  = Map::kGroupCount;
    static constexpr int kThreadCount = kGroupCount * Atom::kThreadCount;

    static constexpr int kTileIterM = Map::kIterM;
    static constexpr int kTileIterN = Map::kIterN;
    static constexpr int kTileIterK = Map::kIterK;

    static constexpr int kDeltaM = Map::kDeltaM;
    static constexpr int kDeltaN = Map::kDeltaN;
    static constexpr int kDeltaK = Map::kDeltaK;

    static constexpr int kAtomM = Map::TileM / Atom::M;
    static constexpr int kAtomN = Map::TileN / Atom::N;
    static constexpr int kAtomK = Map::TileK / Atom::K;

    static constexpr int kMmaIterM = kTileIterM * kAtomM;
    static constexpr int kMmaIterN = kTileIterN * kAtomN;
    static constexpr int kMmaIterK = kTileIterK * kAtomK;

    __device__ static int3 get_offset(int thread_idx)
    {
        return Map::get_offset(Atom::get_group_id(thread_idx));
    }
};

template<class MMA>
struct Rearrange {
    using Map  = typename MMA::Map;
    using Atom = typename MMA::Atom;

    template<class T, int V, int M, int N, class Layout, Order order, int TM, int TN>
    __device__ static void
    apply(Array<T, V> (&frag_C)[M][N], SmemAccessorV2<T, Layout, order>& smem_C, int2 offset_mn, pair<TM, TN>)
    {
        const int3 offset_mnk = MMA::get_offset(threadIdx.x);
        const int  group_id_k = offset_mnk.z / Map::kFootprintK;

        constexpr bool kRakedM = Map::kPartitionM == Partition::kRaked;
        constexpr bool kRakedN = Map::kPartitionN == Partition::kRaked;

        static constexpr int2 kMN0 = cs2mk<order>(Layout::C0, Layout::S0);

        constexpr int kPeriodM  = ceil_div(kMN0.x, Map::kDeltaM);
        constexpr int kPeriodN  = ceil_div(kMN0.y, Map::kDeltaN);
        constexpr int kPeriodM1 = ceil_div(kMN0.x, Atom::M);
        constexpr int kPeriodN1 = ceil_div(kMN0.y, Atom::N);

        constexpr auto offset_C = Atom::static_offset_C();
        const int2     thr      = Atom::thread_offset_C();

        // Contract: All these indices is not a part of swizzling
        int phases[kPeriodM][kPeriodN][kPeriodM1][kPeriodN1][offset_C.size()];
        PRAGMA_UNROLL
        for (int m = 0; m < kPeriodM; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < kPeriodN; ++n) {
                PRAGMA_UNROLL
                for (int m1 = 0; m1 < kPeriodM1; ++m1) {
                    PRAGMA_UNROLL
                    for (int n1 = 0; n1 < kPeriodN1; ++n1) {
                        const int mm = offset_mnk.x + m * Map::kDeltaM + m1 * Atom::M + thr.x;
                        const int nn = offset_mnk.y + n * Map::kDeltaN + n1 * Atom::N + thr.y;
                        PRAGMA_UNROLL
                        for (int i = 0; i < offset_C.size(); ++i) {
                            const int2 cs           = mk2cs<order>(mm + offset_C[i].x, nn + offset_C[i].y);
                            phases[m][n][m1][n1][i] = Layout::apply(cs.y, cs.x);
                        }
                    }
                }
            }
        }

        constexpr int K = Map::kGroupK;
        constexpr int C = offset_C.size();

        int offsets[K][M][N][C];
        int masks[K][M][N][C];

        PRAGMA_UNROLL
        for (int k = 0; k < K; ++k) {
            PRAGMA_UNROLL
            for (int m = 0; m < M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < N; ++n) {
                    int m0 = m / MMA::kAtomM, m1 = m % MMA::kAtomM, n0 = n / MMA::kAtomN, n1 = n % MMA::kAtomN;
                    int m01 =
                        m0 / kPeriodM * kPeriodM * Map::kDeltaM + m1 / kPeriodM1 * kPeriodM1 * Atom::M - offset_mn.x;
                    int n01 =
                        n0 / kPeriodN * kPeriodN * Map::kDeltaN + n1 / kPeriodN1 * kPeriodN1 * Atom::N - offset_mn.y;
                    const int2 cs       = mk2cs<order>(m01, n01);
                    int        offset_0 = Layout::apply(cs.y, cs.x);
                    PRAGMA_UNROLL
                    for (int i = 0; i < offset_C.size(); ++i) {
                        int offset_1        = phases[m0 % kPeriodM][n0 % kPeriodN][m1 % kPeriodM1][n1 % kPeriodN1][i];
                        offsets[k][m][n][i] = offset_0 + offset_1;
                        const int bm        = offset_mnk.x - offset_mn.x + m0 * Map::kDeltaM + m1 * Atom::M + thr.x;
                        const int bn        = offset_mnk.y - offset_mn.y + n0 * Map::kDeltaN + n1 * Atom::N + thr.y;
                        const int mm        = kRakedM ? m01 : bm;
                        const int nn        = kRakedN ? n01 : bn;
                        masks[k][m][n][i]   = (Map::kGroupK == 1 || group_id_k == k)
                                            && (TM >= Map::M || (0 <= mm && mm < TM))
                                            && (TN >= Map::N || (0 <= nn && nn < TN));
                    }
                }
            }
        }

        auto _store = [](auto ptr, auto offset, auto vec) {
            if constexpr (order == kRowMajor) {
                Store(&ptr[offset], vec);
            }
            else {
                for (int i = 0; i < vec.size(); ++i) {
                    ptr[offset + Layout::apply(i, 0)] = vec[i];
                }
            }
        };

        typename Atom::FragC_ reshape_C;

        auto ptr = &smem_C(0, 0);

        PRAGMA_UNROLL
        for (int m = 0; m < M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < N; ++n) {
                Atom::ReshapeC(frag_C[m][n], reshape_C);
                PRAGMA_UNROLL
                for (int c = 0; c < C; ++c) {
                    auto& vec    = reshape_C[c];
                    int   offset = offsets[0][m][n][c];
                    if (masks[0][m][n][c]) {
                        _store(ptr, offset, vec);
                    }
                }
            }
        }

        __syncthreads();

#if 1
        auto _load = [](auto ptr, auto offset, auto& vec) {
            if constexpr (order == kRowMajor) {
                Load(vec, &ptr[offset]);
            }
            else {
                for (int i = 0; i < vec.size(); ++i) {
                    vec[i] = ptr[offset + Layout::apply(i, 0)];
                }
            }
        };

        PRAGMA_UNROLL
        for (int k = 1; k < K; ++k) {
            PRAGMA_UNROLL
            for (int m = 0; m < M; ++m) {
                PRAGMA_UNROLL
                for (int n = 0; n < N; ++n) {
                    Atom::ReshapeC(frag_C[m][n], reshape_C);
                    PRAGMA_UNROLL
                    for (int c = 0; c < C; ++c) {
                        auto& vec    = reshape_C[c];
                        int   offset = offsets[k][m][n][c];
                        if (masks[k][m][n][c]) {
                            std::remove_reference_t<decltype(vec)> tmp;
                            _load(ptr, offset, tmp);
                            {
                                using namespace ops;
                                vec = vec + tmp;
                            }
                            _store(ptr, offset, vec);
                        }
                    }
                }
            }
            __syncthreads();
        }
#endif
    }
};

}  // namespace turbomind::gemm
