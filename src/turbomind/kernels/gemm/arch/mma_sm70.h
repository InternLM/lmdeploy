// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/mma.h"
#include "src/turbomind/kernels/gemm/desc.h"

namespace turbomind::gemm {

struct SM70_MMA_884 {
    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 4;

    static constexpr int kThreadCount = 32;

    static constexpr auto kOpClass = OpClass::kMMA_s884;

    using FragA = Array<half, 4>;
    using FragB = Array<half, 4>;
    using FragC = Array<float, 8>;

    static constexpr int kPieceC = 4;

    __device__ static void fma(FragC& d, const FragA& a, const FragB& b, const FragC& c)
    {
        mma_m8n8k4_row_col(d, a, b, (FragC&)c);
        // mma_m8n8k4_row_row(d, a, b, (FragC&)c);
    }

    template<class Func>
    __device__ static void foreach_C(FragC& c, Func&& func)
    {
        PRAGMA_UNROLL
        for (int n = 0; n < 2; ++n) {
            PRAGMA_UNROLL
            for (int m = 0; m < 2; ++m) {
                ((Func&&)func)((Array<float, 2>&)c[n * 4 + m * 2], n * 2 + m, m * 2, n * 4);
            }
        }
    }

    __device__ static int2 get_offset_C()  // -> (m,n)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        return {
            (lane_id & 8) * 1 + (lane_id & 1) + lane_id / 16 * 4,
            (lane_id & 4) * 2 + (lane_id & 2),
        };
    }

    __device__ static int get_group_id(int thread_idx)
    {
        return thread_idx / WARP_SIZE;
    }
};

}  // namespace turbomind::gemm