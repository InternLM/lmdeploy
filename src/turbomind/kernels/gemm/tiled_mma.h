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

    struct Rearrange {
        template<class T, int V, int M, int N, class Layout, Order order, int TM, int TN>
        __device__ static void
        apply(Array<T, V> (&frag_C)[M][N], SmemAccessorV2<T, Layout, order>& smem_C, int2 offset_mn, pair<TM, TN>)
        {
            const int3 offset_mnk = get_offset(threadIdx.x);
            const int  group_id_k = offset_mnk.z / Map::kFootprintK;

            constexpr bool kRakedM = Map::kPartitionM == Partition::kRaked;
            constexpr bool kRakedN = Map::kPartitionN == Partition::kRaked;

            const int2 thr = Atom::get_offset_C();

            static constexpr int2 kMN0 = cs2mk<order>(Layout::C0, Layout::S0);

            constexpr int kPeriodM  = ceil_div(kMN0.x, Map::kDeltaM);
            constexpr int kPeriodN  = ceil_div(kMN0.y, Map::kDeltaN);
            constexpr int kPeriodM1 = ceil_div(kMN0.x, Atom::M);
            constexpr int kPeriodN1 = ceil_div(kMN0.y, Atom::N);

            // Contract: All these indices is not a part of swizzling
            int phases[kPeriodM][kPeriodN][kPeriodM1][kPeriodN1][Atom::kPieceC];
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
                            Atom::foreach_C(frag_C[m][n], [&](auto, int idx, int mi, int ni) {
                                const int2 cs             = mk2cs<order>(mm + mi, nn + ni);
                                phases[m][n][m1][n1][idx] = Layout::apply(cs.y, cs.x);
                            });
                        }
                    }
                }
            }

            auto ptr = &smem_C(0, 0);

            auto apply = [&](int m, int n, int k) {
                const int m0 = m / kAtomM, m1 = m % kAtomM, n0 = n / kAtomN, n1 = n % kAtomN;
                int m01 = m0 / kPeriodM * kPeriodM * Map::kDeltaM + m1 / kPeriodM1 * kPeriodM1 * Atom::M - offset_mn.x;
                int n01 = n0 / kPeriodN * kPeriodN * Map::kDeltaN + n1 / kPeriodN1 * kPeriodN1 * Atom::N - offset_mn.y;
                const int2 cs       = mk2cs<order>(m01, n01);
                const int  offset_0 = Layout::apply(cs.y, cs.x);
                Atom::foreach_C(frag_C[m][n], [&](auto& vec, int idx, int, int) {
                    const int offset_1 = phases[m0 % kPeriodM][n0 % kPeriodN][m1 % kPeriodM1][n1 % kPeriodN1][idx];
                    const int bm       = offset_mnk.x - offset_mn.x + m0 * Map::kDeltaM + m1 * Atom::M + thr.x;
                    const int bn       = offset_mnk.y - offset_mn.y + n0 * Map::kDeltaN + n1 * Atom::N + thr.y;
                    const int mm       = kRakedM ? m01 : bm;
                    const int nn       = kRakedN ? n01 : bn;
                    const int mask = (TM >= Map::M || (0 <= mm && mm < TM)) && (TN >= Map::N || (0 <= nn && nn < TN));
                    if ((Map::kGroupK == 1 || group_id_k == k) && mask) {
                        if (k > 0) {
                            std::remove_reference_t<decltype(vec)> tmp;
                            if constexpr (order == kRowMajor) {
                                Load(tmp, &ptr[offset_0 + offset_1]);
                            }
                            else {
                                for (int i = 0; i < vec.size(); ++i) {
                                    tmp[i] = ptr[offset_0 + offset_1 + Layout::apply(i, 0)];
                                }
                            }
                            using namespace ops;
                            vec = vec + tmp;
                        }
                        if constexpr (order == kRowMajor) {
                            Store(&ptr[offset_0 + offset_1], vec);
                        }
                        else {
                            for (int i = 0; i < vec.size(); ++i) {
                                ptr[offset_0 + offset_1 + Layout::apply(i, 0)] = vec[i];
                            }
                        }
                    }
                });
            };

            PRAGMA_UNROLL
            for (int k = 0; k < Map::kGroupK; ++k) {
                PRAGMA_UNROLL
                for (int m = 0; m < M; ++m) {
                    PRAGMA_UNROLL
                    for (int n = 0; n < N; ++n) {
                        apply(m, n, k);
                    }
                }
                __syncthreads();
            }
        }
    };
};

}  // namespace turbomind::gemm
