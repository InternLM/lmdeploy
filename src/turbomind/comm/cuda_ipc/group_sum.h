// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"

namespace turbomind::comm {

namespace detail {

template<class Syncgroup>
__device__ float GroupSum(const float val, int warps, Syncgroup syncgroup)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    float     sum     = val;
    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync((uint32_t)-1, sum, mask);
    }
    __shared__ float smem[32];
    // syncgroup();
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    syncgroup();
    for (int i = 1; i < warps; ++i) {
        sum += smem[warp_id / warps * warps + i];
    }
    // sum = {};
    // for (int i = 0; i < warps; ++i) {
    //     sum += smem[warp_id / warps * warps + i];
    // }
    return sum;
}

}  // namespace detail

}  // namespace turbomind::comm
