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

    using OffsetC = Array<int2, 2>;  // (m, n)
    using FragC_  = Array<float, 2>[2];

    __device__ static void fma(FragC& d, const FragA& a, const FragB& b, const FragC& c)
    {
        mma_m16n8k16_row_col(d, a, b, (FragC&)c);
    }

    __device__ static constexpr OffsetC static_offset_C()
    {
        return {int2{0, 0}, int2{8, 0}};
    }

    __device__ static int2 thread_offset_C()  // -> (m,n)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        return {lane_id / 4, lane_id % 4 * 2};
    }

    __device__ static void ReshapeC(const FragC& c, FragC_& c_)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < 2; ++m) {
            c_[m] = (Array<float, 2>&)c[m * 2];
        }
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
