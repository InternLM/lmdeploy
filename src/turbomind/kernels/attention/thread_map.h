// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "../gemm_s_f16/common.h"

#include <iostream>

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

template<int DimC, int DimS, int AccessC, int WarpCount, int WarpThreadC = DimC / AccessC>
struct RakedThreadMap {
    static constexpr int kDimC = DimC;
    static constexpr int kDimS = DimS;

    static constexpr int kWarpCount = WarpCount;
    static constexpr int kAccessC   = AccessC;

    static constexpr int kWarpThreadC = WarpThreadC;
    static constexpr int kWarpThreadS = WARP_SIZE / kWarpThreadC;

    static_assert(kWarpThreadC <= WARP_SIZE);

    static constexpr int kWarpAccessC = kWarpThreadC * kAccessC;
    static constexpr int kWarpAccessS = kWarpThreadS;

    static constexpr int kWarpIterC = (kDimC + kWarpAccessC - 1) / kWarpAccessC;
    static constexpr int kWarpIterS = kDimS / kWarpAccessS;

    static constexpr int kWarpC = 1;
    static constexpr int kWarpS = kWarpCount;

    static constexpr int kIterC = kWarpIterC / kWarpC;
    static constexpr int kIterS = std::max(kWarpIterS / kWarpS, 1);

    // Allow partial tile when there is ONLY 1 iteration
    static_assert(kDimC % kWarpAccessC == 0 || kIterC == 1);

    static_assert(kIterC > 0);
    static_assert(kIterS > 0);

    static constexpr bool kPartialC = kDimC % kWarpAccessC != 0;

    static constexpr int kFootprintC = kWarpAccessC * kIterC;
    static constexpr int kFootprintS = kWarpAccessS * kIterS;

    static constexpr int kDeltaC = kWarpAccessC;
    static constexpr int kDeltaS = kWarpAccessS;

    // static constexpr int kDeltaC = kWarpAccessC * kWarpC;
    // static constexpr int kDeltaS = kWarpAccessS * kWarpS;

    __device__ static int2 get_offset(int warp_id, int lane_id)
    {
        int warp_offset_c = warp_id % kWarpC;
        int warp_offset_s = warp_id / kWarpC;

        int warp_thread_offset_c = lane_id % kWarpThreadC;
        int warp_thread_offset_s = lane_id / kWarpThreadC;

        int cta_thread_offset_c = kFootprintC * warp_offset_c + warp_thread_offset_c * kAccessC;
        int cta_thread_offset_s = kFootprintS * warp_offset_s + warp_thread_offset_s;

        // int cta_thread_offset_c = kWarpAccessC * warp_offset_c + warp_thread_offset_c * kAccessC;
        // int cta_thread_offset_s = kWarpAccessS * warp_offset_s + warp_thread_offset_s;

        return {cta_thread_offset_c, cta_thread_offset_s};
    }
};

namespace {

template<class TMap>
void Print(TMap)
{
    std::cout << "     warps: " << TMap::kWarpCount << "\n";
    std::cout << "     shape: (" << TMap::kDimC << ", " << TMap::kDimS << ")\n";
    std::cout << "    access: (" << TMap::kAccessC << ", " << 1 << ")\n";
    std::cout << "warpThread: (" << TMap::kWarpThreadC << ", " << TMap::kWarpThreadS << ")\n";
    std::cout << "warpAccess: (" << TMap::kWarpAccessC << ", " << TMap::kWarpAccessS << ")\n";
    std::cout << "  warpIter: (" << TMap::kWarpIterC << ", " << TMap::kWarpIterS << ")\n";
    std::cout << "      warp: (" << TMap::kWarpC << ", " << TMap::kWarpS << ")\n";
    std::cout << "      iter: (" << TMap::kIterC << ", " << TMap::kIterS << ")\n";
    std::cout << " footprint: (" << TMap::kFootprintC << ", " << TMap::kFootprintS << ")\n";
    std::cout << "     delta: (" << TMap::kDeltaC << ", " << TMap::kDeltaS << ")\n";
    std::cout << "  partialC: " << TMap::kPartialC << "\n";
}

}  // namespace

}  // namespace turbomind
