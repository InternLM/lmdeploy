// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"

namespace turbomind::gemm {

template<class T>
struct SmemCopy_MMA_884_A {
    // static constexpr int M = 16;
    // static constexpr int K = 8;
    static constexpr int M = 8;
    static constexpr int K = 8;

    static constexpr int kFragNum = 1;

    using Frag = Array<T, K>;

    __device__ static int2 unique(int thread_idx, int pack_idx)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        //                   4                3               01
        // const int m = lane_id / 16 * 4 + (lane_id & 8) + lane_id % 4;
        // return {pack_idx * M + m, (lane_id & 4) >> 2};

        //                   4                01
        const int m = lane_id / 16 * 4 + lane_id % 4;
        return {pack_idx * M + m, (lane_id & 12) >> 2};
    }

    __device__ static int2 get_offset(int thread_idx)
    {
        return int2{unique(thread_idx, 0).x, 0};
    }

    template<class S, class D>
    __device__ static void copy(S&& src_ptr, D&& dst_ptr, bool)
    {
        Lds(*(Frag*)dst_ptr, src_ptr);
    }
};

template<class T>
struct SmemCopy_MMA_884_B {
    // static constexpr int M = 16;
    // static constexpr int K = 8;
    static constexpr int M = 32;
    static constexpr int K = 8;

    static constexpr int kFragNum = 1;

    using Frag = Array<T, K>;

    __device__ static int2 unique(int thread_idx, int pack_idx)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        //                4                     2                 01
        // const int m = lane_id / 16 * 4 + (lane_id & 4) * 2 + lane_id % 4;
        // return {pack_idx * M + m, (lane_id & 8) >> 3};

        //                  4                  23                  01
        const int m = lane_id / 16 * 4 + (lane_id & 12) * 2 + lane_id % 4;
        return {pack_idx * M + m, 0};
    }

    __device__ static int2 get_offset(int thread_idx)
    {
        return int2{unique(thread_idx, 0).x, 0};
    }

    template<class S, class D>
    __device__ static void copy(S&& src_ptr, D&& dst_ptr, bool)
    {
        Lds(*(Frag*)dst_ptr, src_ptr);
    }
};

template<class T, int K_>
struct SmemCopy_MMA_884_V {
    // static constexpr int M = 16;
    static constexpr int M = 32;
    static constexpr int K = K_;

    static constexpr int kFragNum = 1;

    using Frag = Array<T, 1>;

    __device__ static int2 unique(int thread_idx, int pack_idx)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        //                4                     2                 01
        // const int m = lane_id / 16 * 4 + (lane_id & 4) * 2 + lane_id % 4;
        // return {pack_idx * 16 + m, (lane_id & 8) >> 3};

        const int m = lane_id / 16 * 4 + (lane_id & 12) * 2 + lane_id % 4;
        return {pack_idx * M + m, 0};
    }

    __device__ static int2 get_offset(int thread_idx)
    {
        return int2{unique(thread_idx, 0).x, 0};
    }

    template<class S, class D>
    __device__ static void copy(S&& src_ptr, D&& dst_ptr, bool)
    {
        Lds(*(Frag*)dst_ptr, src_ptr);
    }
};

}  // namespace turbomind::gemm
