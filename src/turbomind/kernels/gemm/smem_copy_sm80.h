// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/gemm/smem_copy.h"

namespace turbomind::gemm {

template<bool Trans>
struct LDSM_x4 {
    template<class S, class D>
    __device__ static void apply(S src_ptr, D dst_ptr)
    {
        const uint32_t uint_ptr = cast_smem_ptr_to_uint(src_ptr);
        if constexpr (Trans) {
            ldsm_x4_trans(*(Array<uint32_t, 4>*)dst_ptr, uint_ptr);
        }
        else {
            ldsm_x4(*(Array<uint32_t, 4>*)dst_ptr, uint_ptr);
        }
    }
};

template<class T, bool trans>
struct SmemCopy_MMA_16816_A {
    static constexpr int M = 16;
    static constexpr int K = 16;

    static constexpr int kFragNum = 1;

    using Frag = Array<T, 8>;

    __device__ static int2 get_offset(int thread_idx)  // -> (m, k)
    {
        const int lane_id = thread_idx % WARP_SIZE;

        const int c = lane_id / 16 * 8;
        const int s = lane_id % 16;

        return trans ? int2{c, s} : int2{s, c};
    }

    template<class S, class D>
    __device__ static void copy(S&& src_ptr, D&& dst_ptr, bool)
    {
        LDSM_x4<trans>::apply((S&&)src_ptr, (D&&)dst_ptr);
    }

    __device__ static int2 unique(int thread_idx, int pack_idx)
    {
        return {pack_idx * WARP_SIZE + thread_idx % WARP_SIZE, 0};
    }
};

template<class T, bool trans>
struct SmemCopy_MMA_16816_B {
    static constexpr int M = 16;
    static constexpr int K = 16;

    static constexpr int kFragNum = 1;

    using Frag = Array<T, 8>;

    __device__ static int2 get_offset(int thread_idx)
    {
        const int lane_id = thread_idx % WARP_SIZE;

        const int c = lane_id / 8 * 8 % 16;
        const int s = lane_id % 8 + lane_id / 16 * 8;

        return trans ? int2{c, s} : int2{s, c};
    }

    template<class S, class D>
    __device__ static void copy(S&& src_ptr, D&& dst_ptr, bool)
    {
        LDSM_x4<trans>::apply((S&&)src_ptr, (D&&)dst_ptr);
    }

    __device__ static int2 unique(int thread_idx, int pack_idx)
    {
        return {pack_idx * WARP_SIZE + thread_idx % WARP_SIZE, 0};
    }
};

template<class T>
struct SmemCopy_MMA_16816_U {  // (M, K)
    static constexpr int M = 16;
    static constexpr int K = 1;

    static constexpr int kFragNum = 1;

    using Frag = Array<T, 2>;

    __device__ static int2 get_offset(int thread_idx)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        // Note: this forbids sub-tile group sizes
        return {lane_id / 4, 0};
    }

    template<class S, class D>
    __device__ static void copy(S&& src_ptr, D&& dst_ptr, bool mask)
    {
        PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
            Lds(*((Array<T, 1>*)dst_ptr + i), src_ptr + i * 8);
        }
    }

    __device__ static int2 unique(int thread_idx, int pack_idx)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        return {pack_idx * 8 + lane_id / 4, lane_id % 4};
    }
};

}  // namespace turbomind::gemm