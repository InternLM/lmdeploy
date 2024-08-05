// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/gemm/simt.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

template<class T, int K_>
struct SmemCopy_MMA_SIMT_A {
    static constexpr int M = simt::OP_M;
    static constexpr int K = simt::OP_K;

    static constexpr int OP_N = simt::OP_N;

    static constexpr int kFragNum = 1;

    using Frag = Array<T, K>;

    __device__ static int2 get_offset(int thread_idx)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        return {lane_id / OP_N, 0};
    }

    template<class S, class D>
    __device__ static void copy(S&& src_ptr, D&& dst_ptr, bool)  // -> (m, k)
    {
        Lds(*(Frag*)dst_ptr, (S &&) src_ptr);
    }

    __device__ static int2 unique(int thread_idx, int pack_idx)  // -> (unique id, repeat id)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        return {pack_idx * M + lane_id / OP_N, lane_id % OP_N};
    }
};

template<class T, int K_>
struct SmemCopy_MMA_SIMT_B {
    static constexpr int M = simt::OP_N;
    static constexpr int K = simt::OP_K;

    static constexpr int OP_N = simt::OP_N;

    static constexpr int kFragNum = 1;

    using Frag = Array<T, K>;

    __device__ static int2 get_offset(int thread_idx)  // -> (m, k)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        return {lane_id % OP_N, 0};
    }

    template<class S, class D>
    __device__ static void copy(S&& src_ptr, D&& dst_ptr, bool)
    {
        Lds(*(Frag*)dst_ptr, (S &&) src_ptr);
    }

    __device__ static int2 unique(int thread_idx, int pack_idx)  // -> (unique id, repeat id)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        return {pack_idx * OP_N + lane_id % OP_N, lane_id / OP_N};
    }
};

template<class T, int K_>
struct SmemCopy_MMA_SIMT_V {
    static constexpr int M = simt::OP_N;
    static constexpr int K = K_;

    static constexpr int OP_N = simt::OP_N;

    static constexpr int kFragNum = 1;

    using Frag = Array<T, 1>;

    __device__ static int2 unique(int thread_idx, int pack_idx)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        return {pack_idx * OP_N + lane_id % OP_N, lane_id / OP_N};
    }

    __device__ static int2 get_offset(int thread_idx)  // -> (m, k)
    {
        return {unique(thread_idx, 0).x, 0};
    }

    template<class S, class D>
    __device__ static void copy(S&& src_ptr, D&& dst_ptr, bool mask)
    {
        Lds(*(Frag*)dst_ptr, src_ptr);
    }
};

}  // namespace turbomind::gemm
