// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/simt.h"

namespace turbomind::gemm {

template<class T>
struct MMA_SIMT {
    static constexpr int M = simt::OP_M;
    static constexpr int N = simt::OP_N;
    static constexpr int K = simt::OP_K;

    static constexpr int kThreadCount = 32;

    static constexpr auto kOpClass = OpClass::kSIMT;

    using FragA = Array<T, K>;
    using FragB = Array<T, K>;
    using FragC = Array<float, 1>;

    using OffsetC = Array<int2, 1>;
    using FragC_  = FragC[1];

    __device__ static void fma(FragC& d, const FragA& a, const FragB& b, const FragC& c)
    {
        PRAGMA_UNROLL
        for (int k = 0; k < K; ++k) {
            d[0] = c[0] + float(a[k]) * float(b[k]);
        }

        // PRAGMA_UNROLL
        // for (int k = 0; k < K; ++k) {
        //     d[0] = c[0] + float(a[k] * b[k]);
        // }

        // T acc{};
        // PRAGMA_UNROLL
        // for (int k = 0; k < K; ++k) {
        //     acc += a[k] * b[k];
        // }
        // d[0] = c[0] + float(acc);
    }

    __device__ static constexpr OffsetC static_offset_C()
    {
        return {};
    }

    __device__ static int2 thread_offset_C()  // -> (m,n)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        return {lane_id / N, lane_id % N};
    }

    __device__ static void ReshapeC(const FragC& c, FragC_& c_)
    {
        c_[0] = c;
    }

    __device__ static int get_group_id(int thread_idx)
    {
        return thread_idx / WARP_SIZE;
    }
};

}  // namespace turbomind::gemm
