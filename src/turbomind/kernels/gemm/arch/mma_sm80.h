// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/mma.h"
#include "src/turbomind/kernels/gemm/desc.h"

namespace turbomind::gemm {

struct SM80_MMA_16x8x16_F32_F16_F16_F32_TN {
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 16;

    static constexpr int kThreadCount = 32;

    static constexpr auto kOpClass = OpClass::kMMA_s16816;

    using FragA = Array<half, 8>;
    using FragB = Array<half, 4>;
    using FragC = Array<float, 4>;

    static constexpr int kPieceC = 2;

    __device__ static void fma(FragC& d, const FragA& a, const FragB& b, const FragC& c)
    {
        mma_m16n8k16_row_col(d, a, b, (FragC&)c);
    }

    template<class Func>
    __device__ static void foreach_C(FragC& c, Func&& func)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < 2; ++m) {
            ((Func&&)func)((Array<float, 2>&)c[m * 2], m, m * 8, 0);
        }
    }

    __device__ static int2 get_offset_C()  // -> (m,n)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        return {lane_id / 4, lane_id % 4 * 2};
    }

    __device__ static int get_group_id(int thread_idx)
    {
        return thread_idx / WARP_SIZE;
    }
};

// This is not used yet
struct SM75_MMA_16x8x8_F32_F16_F16_F32_TN: SM80_MMA_16x8x16_F32_F16_F16_F32_TN {
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 8;

    using FragA = Array<half, 4>;
    using FragB = Array<half, 2>;
    using FragC = Array<float, 4>;

    __device__ static void fma(FragC& d, const FragA& a, const FragB& b, const FragC& c)
    {
        mma_m16n8k8_row_col(d, a, b, (FragC&)c);
    }
};

}  // namespace turbomind::gemm
