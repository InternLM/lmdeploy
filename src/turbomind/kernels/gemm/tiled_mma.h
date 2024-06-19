// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/mma.h"

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
};

}  // namespace turbomind::gemm