// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/smem.h"

namespace turbomind::gemm {

struct LDSM_x4_N {
    template<class T>
    __device__ static void copy(const T* src, T* dst)
    {
        ldsm_x4(*(Array<uint32_t, 4>*)dst, cast_smem_ptr_to_uint(src));
    }
};

struct LDSM_x4_T {
    template<class T>
    __device__ static void copy(const T* src, T* dst)
    {
        ldsm_x4_trans(*(Array<uint32_t, 4>*)dst, cast_smem_ptr_to_uint(src));
    }
};

struct LDS_128 {
    template<class T>
    __device__ static void copy(const T* src, T* dst)
    {
        *(uint4*)dst = *(const uint4*)src;
    }
};

struct LDS_64 {
    template<class T>
    __device__ static void copy(const T* src, T* dst)
    {
        *(uint2*)dst = *(const uint2*)src;
    }
};

struct LDS_32 {
    template<class T>
    __device__ static void copy(const T* src, T* dst)
    {
        *(uint*)dst = *(const uint*)src;
    }
};

template<class T, bool Trans>
struct SmemCopy_MMA_16816_A {
    static constexpr int   kWarpAccessS  = 16;
    static constexpr int   kWarpAccessC  = 16;
    static constexpr int   kFragmentSize = 8;
    __device__ static int2 get_offset(int lane_id)
    {
        return {
            lane_id / 16 * 8,  // c
            lane_id % 16       // s
        };
    }
    using Copy = std::conditional_t<Trans, LDSM_x4_T, LDSM_x4_N>;
    using Frag = Array<T, kFragmentSize>;
};

template<class T, bool Trans>
struct SmemCopy_MMA_16816_B {
    static constexpr int   kWarpAccessC  = 16;
    static constexpr int   kWarpAccessS  = 16;
    static constexpr int   kFragmentSize = 8;
    __device__ static int2 get_offset(int lane_id)
    {
        return {
            lane_id / 8 * 8 % 16,           // c
            lane_id % 8 + lane_id / 16 * 8  // s
        };
    }
    using Copy = std::conditional_t<Trans, LDSM_x4_T, LDSM_x4_N>;
    using Frag = Array<T, kFragmentSize>;
};

template<class Atom_, int S, int C>
struct SmemCopy_ {

    using Atom = Atom_;

    static constexpr int ITER_S = S / Atom::kWarpAccessS;
    static constexpr int ITER_C = C / Atom::kWarpAccessC;

    static constexpr int DELTA_S = Atom::kWarpAccessS;
    static constexpr int DELTA_C = Atom::kWarpAccessC;

    using Frag = typename Atom::Frag[ITER_S * ITER_C];

    template<class Accessor>
    __device__ static void copy(Accessor src, Frag& dst, int2 offset_cs)
    {
        const int2 thr_cs = Atom::get_offset(threadIdx.x % WARP_SIZE);
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int ss = offset_cs.y + thr_cs.y + s * DELTA_S;
                const int cc = offset_cs.x + thr_cs.x + c * DELTA_C;
                Atom::Copy::copy(&src(ss, cc), dst[s * ITER_C + c].data());
            }
        }
    }
};

}  // namespace turbomind::gemm