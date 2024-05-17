// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/mma.h"

namespace turbomind::gemm {

struct SM80_MMA_16x8x16_F32_F16_F16_F32_TN {
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 16;

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
};

template<class MMA_Atom_, int M_, int N_, int K_>
struct TiledMMA {
    using MMA_Atom = MMA_Atom_;

    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int K = K_;

    static constexpr int ITER_M = M / MMA_Atom::M;
    static constexpr int ITER_N = N / MMA_Atom::N;
    static constexpr int ITER_K = K / MMA_Atom::K;
};

}  // namespace turbomind::gemm