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

struct SmemCopy_MMA_16816_A {
    using CopyAtom = LDSM_x4_N;

    static constexpr int kFragmentSize = 8;

    static constexpr int kWarpAccessS = 16;
    static constexpr int kWarpAccessC = 16;

    __device__ static int2 get_offset(int lane_id)
    {
        return {
            lane_id / 16 * 8,  // c
            lane_id % 16       // s
        };
    }
};

struct SmemCopy_MMA_16816_B {
    using CopyAtom = LDSM_x4_N;

    static constexpr int kWarpAccessC = 16;
    static constexpr int kWarpAccessS = 16;

    static constexpr int kFragmentSize = 8;

    __device__ static int2 get_offset(int lane_id)
    {
        return {
            lane_id / 8 * 8 % 16,           // c
            lane_id % 8 + lane_id / 16 * 8  // s
        };
    }
};

template<class TiledCopy, class Accessor, class T>
__device__ void CopySmem(TiledCopy, Accessor smem, T* dst, int offset_s, int offset_c, int shape_s, int shape_c)
{
    const int2 thr_cs = TiledCopy::get_offset(threadIdx.x % WARP_SIZE);
    PRAGMA_UNROLL
    for (int s = 0; s < shape_s; s += TiledCopy::kWarpAccessS) {
        const int ss = offset_s + thr_cs.y + s;
        PRAGMA_UNROLL
        for (int c = 0; c < shape_c; c += TiledCopy::kWarpAccessC) {
            const int cc = offset_c + thr_cs.x + c;
            TiledCopy::CopyAtom::copy(&smem(ss, cc), dst);
            dst += TiledCopy::kFragmentSize;
        }
    }
}

}  // namespace turbomind::gemm