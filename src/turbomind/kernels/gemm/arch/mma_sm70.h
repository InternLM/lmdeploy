// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/mma.h"
#include "src/turbomind/kernels/gemm/desc.h"

namespace turbomind::gemm {

struct SM70_MMA_884 {
    // static constexpr int M = 16;
    // static constexpr int N = 16;
    static constexpr int M = 8;
    static constexpr int N = 32;
    static constexpr int K = 8;

    static constexpr int kThreadCount = 32;

    static constexpr auto kOpClass = OpClass::kMMA_s884;

    using FragA = Array<half, K>;
    using FragB = Array<half, K>;
    using FragC = Array<float, 8>;

    using OffsetC = Array<int2, 4>;
    using FragC_  = Array<float, 2>[4];

    __device__ static void fma(FragC& d, const FragA& a, const FragB& b, const FragC& c)
    {
        mma_m8n8k4_row_col(d, (const Array<half, 4>&)a[0], (const Array<half, 4>&)b[0], (FragC&)c);
        if constexpr (K == 8) {
            mma_m8n8k4_row_col(d, (const Array<half, 4>&)a[4], (const Array<half, 4>&)b[4], (FragC&)d);
        }
    }

    __device__ static constexpr OffsetC static_offset_C()
    {
        OffsetC r{};
        PRAGMA_UNROLL
        for (int n = 0; n < 2; ++n) {
            PRAGMA_UNROLL
            for (int m = 0; m < 2; ++m) {
                r[n * 2 + m] = int2{m * 2, n * 4};
            }
        }
        return r;
    }

    __device__ static int2 thread_offset_C()  // -> (m,n)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        // return {
        //     (lane_id & 8) * 1 + (lane_id & 1) + lane_id / 16 * 4,
        //     (lane_id & 4) * 2 + (lane_id & 2),
        // };
        return {(lane_id & 1) + (lane_id / 16) * 4,  //
                (lane_id & 2) + (lane_id & 12) * 2};
    }

    __device__ static void ReshapeC(const FragC& c, FragC_& c_)
    {
        PRAGMA_UNROLL
        for (int m = 0; m < 4; ++m) {
            c_[m] = (Array<float, 2>&)c[m * 2];
        }
    }

    __device__ static int get_group_id(int thread_idx)
    {
        return thread_idx / WARP_SIZE;
    }
};

}  // namespace turbomind::gemm
