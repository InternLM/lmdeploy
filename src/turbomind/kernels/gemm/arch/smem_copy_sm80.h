// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/smem.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

template<bool trans>
struct LDSM_x4 {
    template<class S, class D>
    __device__ static void apply(S src_ptr, D dst_ptr)
    {
        const uint32_t uint_ptr = cast_smem_ptr_to_uint(src_ptr);
        if constexpr (trans) {
            ldsm_x4_trans(*(Array<uint32_t, 4>*)dst_ptr, uint_ptr);
        }
        else {
            ldsm_x4(*(Array<uint32_t, 4>*)dst_ptr, uint_ptr);
        }
    }
};

template<bool trans>
struct LDSM_x2 {
    template<class S, class D>
    __device__ static void apply(S src_ptr, D dst_ptr)
    {
        const uint32_t uint_ptr = cast_smem_ptr_to_uint(src_ptr);
        if constexpr (trans) {
            ldsm_x2_trans(*(Array<uint32_t, 2>*)dst_ptr, uint_ptr);
        }
        else {
            ldsm_x2(*(Array<uint32_t, 2>*)dst_ptr, uint_ptr);
        }
    }
};

template<bool trans>
struct LDSM_x1 {
    template<class S, class D>
    __device__ static void apply(S src_ptr, D dst_ptr)
    {
        const uint32_t uint_ptr = cast_smem_ptr_to_uint(src_ptr);
        if constexpr (trans) {
            ldsm_x1_trans(*(Array<uint32_t, 1>*)dst_ptr, uint_ptr);
        }
        else {
            ldsm_x1(*(Array<uint32_t, 1>*)dst_ptr, uint_ptr);
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
        LDSM_x4<trans>::apply((S &&) src_ptr, (D &&) dst_ptr);
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
        LDSM_x4<trans>::apply((S &&) src_ptr, (D &&) dst_ptr);
    }

    __device__ static int2 unique(int thread_idx, int pack_idx)
    {
        return {pack_idx * WARP_SIZE + thread_idx % WARP_SIZE, 0};
    }
};

template<class T, int M_, int K_, Order mat_order, Order thr_order>
struct LDSM_SM75_8x8 {
    static constexpr int M = M_;
    static constexpr int K = K_;

    static constexpr int iM = M / 8;
    static constexpr int iK = K / 8;

    static constexpr int kFragNum = 1;

    using Frag = Array<T, 2 * iM * iK>;

    __device__ static int2 get_offset(int thread_idx)
    {
        const int lane_id = thread_idx % WARP_SIZE;
        int       c, s;
        if constexpr (mat_order == kColMajor) {
            s = lane_id % 16;
            c = lane_id / 16 * 8;
        }
        else {
            s = lane_id / 16 * 8 + lane_id % 8;
            c = lane_id & 8;
        }
        int2 mk = cs2mk<thr_order>(c, s);
#if __CUDA_ARCH__ <= 750  // wrap ptrs around for sm_75
        mk.x %= M;
        mk.y %= K;
#endif
        return mk;
    }

    template<class S, class D>
    __device__ static void copy(S&& src_ptr, D&& dst_ptr, bool)
    {
        constexpr bool trans = thr_order != kRowMajor;
        if constexpr (sizeof(Frag) == 16) {
            LDSM_x4<trans>::apply((S &&) src_ptr, (D &&) dst_ptr);
        }
        else if constexpr (sizeof(Frag) == 8) {
            LDSM_x2<trans>::apply((S &&) src_ptr, (D &&) dst_ptr);
        }
        else if constexpr (sizeof(Frag) == 4) {
            LDSM_x1<trans>::apply((S &&) src_ptr, (D &&) dst_ptr);
        }
        else {
            static_assert(sizeof(S) != sizeof(S), "not implemented");
        }
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
