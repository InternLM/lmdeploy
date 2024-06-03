// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

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

template<class T, bool Trans>
struct SmemCopy_MMA_16816_A {
    static constexpr int S = 16;
    static constexpr int C = 16;

    using Frag = Array<T, 8>;

    __device__ static int2 get_offset(int thread_idx)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        return {
            lane_id / 16 * 8,  // c
            lane_id % 16       // s
        };
    }

    template<class S, class D>
    __device__ static void copy(S&& src_ptr, D&& dst_ptr, bool)
    {
        LDSM_x4<Trans>::apply((S&&)src_ptr, (D&&)dst_ptr);
    }
};

template<class T, bool Trans>
struct SmemCopy_MMA_16816_B {
    static constexpr int S = 16;
    static constexpr int C = 16;

    using Frag = Array<T, 8>;

    __device__ static int2 get_offset(int thread_idx)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        return {
            lane_id / 8 * 8 % 16,           // c
            lane_id % 8 + lane_id / 16 * 8  // s
        };
    }

    template<class S, class D>
    __device__ static void copy(S&& src_ptr, D&& dst_ptr, bool)
    {
        LDSM_x4<Trans>::apply((S&&)src_ptr, (D&&)dst_ptr);
    }
};

template<class T>
struct LoadFragment_MMA_16816_Q {  // (M, K)
    static constexpr int C = 16;
    static constexpr int S = 16;

    using Frag = Array<T, 2>;

    __device__ static int2 get_offset(int lane_id)
    {
        return {lane_id / 4, 0};
    }

    // __device__ static void apply(const T* src, T* dst)
    // {
    //     PRAGMA_UNROLL
    //     for (int i = 0; i < 2; ++i) {
    //         Lds(*(Array<T, 1>*)dst + i, src + i * 8);
    //     }
    // }
};

}  // namespace turbomind::gemm