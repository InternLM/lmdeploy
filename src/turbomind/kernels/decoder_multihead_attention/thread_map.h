// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "../gemm_s_f16/common.h"
#include "src/turbomind/kernels/custom_ar_kernels.h"

namespace turbomind {

template<int C, int S, int AccessC, int WarpCount>
struct ThreadMapQ {
    static constexpr int kWarpCount = WarpCount;
    static constexpr int kAccessC   = AccessC;

    static constexpr int kWarpThreadC = C / kAccessC;
    static constexpr int kWarpThreadS = WARP_SIZE / kWarpThreadC;

    static_assert(kWarpThreadC <= WARP_SIZE);

    static constexpr int kWarpAccessC = kWarpThreadC * kAccessC;  // C
    static constexpr int kWarpAccessS = kWarpThreadS;

    static constexpr int kWarpIterC = C / kWarpAccessC;  // 1
    static constexpr int kWarpIterS = S / kWarpAccessS;

    static constexpr int kWarpC = 1;
    static constexpr int kWarpS = kWarpCount;

    static constexpr int kIterC = kWarpIterC / kWarpC;  // 1
    static constexpr int kIterS = std::max(kWarpIterS / kWarpS, 1);

    static constexpr int kFootprintC = kWarpAccessC * kIterC;  // C
    static constexpr int kFootprintS = kWarpAccessS * kIterS;

    static constexpr int kDeltaC = kWarpAccessC;
    static constexpr int kDeltaS = kWarpAccessS;

    __device__ static int2 get_offset(int warp_id, int lane_id)
    {
        int warp_offset_c = warp_id % kWarpC;
        int warp_offset_s = warp_id / kWarpC;

        int warp_thread_offset_c = lane_id % kWarpThreadC;
        int warp_thread_offset_s = lane_id / kWarpThreadC;

        int cta_thread_offset_c = kFootprintC * warp_offset_c + warp_thread_offset_c * kAccessC;
        int cta_thread_offset_s = kFootprintS * warp_offset_s + warp_thread_offset_s;

        return {cta_thread_offset_c, cta_thread_offset_s};
    }
};

template<int C, int S, int AccessC, int WarpThreadC, int WarpCount>
struct ThreadMapKv {
    static constexpr int kC = C;
    static constexpr int kS = S;

    static constexpr int kWarpCount = WarpCount;
    static constexpr int kAccessC   = AccessC;

    static constexpr int kWarpThreadC = WarpThreadC;
    static constexpr int kWarpThreadS = WARP_SIZE / kWarpThreadC;

    static_assert(kWarpThreadC <= WARP_SIZE);

    static constexpr int kWarpAccessC = kWarpThreadC * kAccessC;
    static constexpr int kWarpAccessS = kWarpThreadS;

    static constexpr int kWarpIterC = C / kWarpAccessC;
    static constexpr int kWarpIterS = S / kWarpAccessS;

    static constexpr int kWarpC = 1;
    static constexpr int kWarpS = kWarpCount;

    static constexpr int kIterC = kWarpIterC / kWarpC;
    static constexpr int kIterS = std::max(kWarpIterS / kWarpS, 1);

    static constexpr int kFootprintC = kWarpAccessC * kIterC;
    static constexpr int kFootprintS = kWarpAccessS * kIterS;

    static constexpr int kDeltaC = kWarpAccessC;
    static constexpr int kDeltaS = kWarpAccessS;

    __device__ static int2 get_offset(int warp_id, int lane_id)
    {
        int warp_offset_c = warp_id % kWarpC;
        int warp_offset_s = warp_id / kWarpC;

        int warp_thread_offset_c = lane_id % kWarpThreadC;
        int warp_thread_offset_s = lane_id / kWarpThreadC;

        int cta_thread_offset_c = kFootprintC * warp_offset_c + warp_thread_offset_c * kAccessC;
        int cta_thread_offset_s = kFootprintS * warp_offset_s + warp_thread_offset_s;

        return {cta_thread_offset_c, cta_thread_offset_s};
    }
};

}  // namespace turbomind
