// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/core/mma.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/simt.h"

namespace turbomind::gemm {

struct SM80_MMA_16x8x16_F32_F16_F16_F32_TN {
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 16;

    static constexpr int kThreadCount = 32;

    using FragA = Array<half, 8>;
    using FragB = Array<half, 4>;
    using FragC = Array<float, 4>;

    __device__ static void fma(FragC& d, const FragA& a, const FragB& b, const FragC& c)
    {
        mma_m16n8k16_row_col(d, a, b, (FragC&)c);
    }

    template<class Func>
    __device__ static void foreach_C(FragC& c, Func&& func)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int m = 0; m < 2; ++m) {
            const int mi = lane_id / 4 + m * 8;
            const int ni = lane_id % 4 * 2;
            ((Func&&)func)((Array<float, 2>&)c[m * 2], mi, ni);
        }
    }

    __device__ static int get_group_id(int thread_idx)
    {
        return thread_idx / WARP_SIZE;
    }
};

template<class T>
struct SM70_MMA_SIMT {
    static constexpr int M = sm70_mma_simt::OP_M;
    static constexpr int N = sm70_mma_simt::OP_N;
    static constexpr int K = sm70_mma_simt::OP_K;

    static constexpr int kThreadCount = 32;

    using FragA = Array<T, K>;
    using FragB = Array<T, K>;
    using FragC = Array<float, 1>;

    __device__ static void fma(FragC& d, const FragA& a, const FragB& b, const FragC& c)
    {
        PRAGMA_UNROLL
        for (int k = 0; k < K; ++k) {
            d[0] = c[0] + float(a[k]) * float(b[k]);
        }
    }

    template<class Func>
    __device__ static void foreach_C(FragC& c, Func&& func)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;

        const int mi = lane_id / N;
        const int ni = lane_id % N;

        ((Func&&)func)(c, mi, ni);
    }

    __device__ static int get_group_id(int thread_idx)
    {
        return thread_idx / WARP_SIZE;
    }
};

struct SM70_MMA_884 {
    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 4;

    static constexpr int kThreadCount = 32;

    using FragA = Array<half, 4>;
    using FragB = Array<half, 4>;
    using FragC = Array<float, 8>;

    __device__ static void fma(FragC& d, const FragA& a, const FragB& b, const FragC& c)
    {
        mma_m8n8k4_row_col(d, a, b, (FragC&)c);
    }

    template<class Func>
    __device__ static void foreach_C(FragC& c, Func&& func)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        PRAGMA_UNROLL
        for (int nn = 0; nn < 2; ++nn) {
            PRAGMA_UNROLL
            for (int mm = 0; mm < 2; ++mm) {
                const int mi = (lane_id & 8) + (lane_id & 1) + lane_id / 16 * 4 + mm * 2;
                const int ni = (lane_id & 4) * 2 + (lane_id & 2) + nn * 4;
                ((Func&&)func)((Array<float, 2>&)c[nn * 4 + mm * 2], mi, ni);
            }
        }
    }

    __device__ static int get_group_id(int thread_idx)
    {
        return thread_idx / WARP_SIZE;
    }
};

template<class MMA_Atom_, class ThreadGroupMap>
struct Tiled_MMA_v2 {
    using Atom = MMA_Atom_;
    using Map  = ThreadGroupMap;

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

    template<class T, int V, int M, int N, class Func>
    __device__ static void _foreach_C(Array<T, V> (&frag_C)[M][N], Func&& func)
    {
        const int3 offset_mnk = get_offset(threadIdx.x);
        const int  offset_m   = offset_mnk.x;
        const int  offset_n   = offset_mnk.y;

        PRAGMA_UNROLL
        for (int m = 0; m < M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < N; ++n) {
                const int mm = offset_m + m / kAtomM * Map::TileM + m % kAtomM * Atom::M;
                const int nn = offset_n + n / kAtomN * Map::TileN + n % kAtomN * Atom::N;
                Atom::foreach_C(frag_C[m][n], [&](auto& vec, int mi, int ni) {  //
                    ((Func&&)func)(vec, mm + mi, nn + ni);
                });
            }
        }
    }

    struct Rearrange {
        template<class FragC, class AccessorV2, int PM, int PN>
        __device__ static void apply(FragC& frag_C, AccessorV2& smem_C, int2 offset_mn, pair<PM, PN>)
        {
            const int3 offset_mnk = get_offset(threadIdx.x);
            const int  group_id_k = offset_mnk.z / Map::kFootprintK;

            PRAGMA_UNROLL
            for (int k = 0; k < Map::kGroupK; ++k) {
                // `vec` is a array in C's continguous dim
                _foreach_C(frag_C, [&](auto& vec, int mi, int ni) {
                    const int mm = mi - offset_mn.x;
                    const int nn = ni - offset_mn.y;
                    // const int mm       = mi;
                    // const int nn       = ni;
                    auto smem_ptr = &smem_C(mm, nn);

                    // Store(smem_ptr, vec);

                    //  *(uint2*)smem_ptr = (const uint2&)vec;

                    StShared(cast_smem_ptr_to_uint(smem_ptr), vec);
                    // *(uint2*)smem_ptr = (uint2&)vec;
                    // static_assert(sizeof(vec) == 8);
                    // if ((Map::kGroupK == 1 || group_id_k == k)     //
                    //     && (PM >= Map::M || (0 <= mm && mm < PM))  //
                    //     && (PN >= Map::N || (0 <= nn && nn < PN))) {
                    //     if (k > 0) {  // constant
                    //         std::remove_reference_t<decltype(vec)> tmp;
                    //         Load(tmp, smem_ptr);
                    //         {
                    //             using namespace ops;
                    //             vec = vec + tmp;
                    //         }
                    //     }

                    //     // for (const auto& x : vec) {
                    //     //     printf("%d %f %d %d\n", (int)threadIdx.x, x, mi, ni);
                    //     // }

                    //     Store(smem_ptr, vec);
                    // }
                });
                __syncthreads();
            }
        }
    };
};

}  // namespace turbomind::gemm