// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/core/meta.h"

#include "src/turbomind/kernels/gemm/types.h"

#include <iostream>

namespace turbomind::gemm {

template<int DimC, int DimS, int AccessC, int WarpCount, int WarpThreadC = std::min(WARP_SIZE, DimC / AccessC)>
struct ThreadMap {
    static constexpr int kDimC = DimC;
    static constexpr int kDimS = DimS;

    static constexpr int kWarpCount = WarpCount;
    static constexpr int kAccessC   = AccessC;

    static constexpr int kWarpThreadC = WarpThreadC;
    static constexpr int kWarpThreadS = WARP_SIZE / kWarpThreadC;

    static_assert(kWarpThreadC <= WARP_SIZE);

    static constexpr int kWarpAccessC = kWarpThreadC * kAccessC;
    static constexpr int kWarpAccessS = kWarpThreadS;

    static constexpr int kWarpIterC = ceil_div(kDimC, kWarpAccessC);
    static constexpr int kWarpIterS = ceil_div(kDimS, kWarpAccessS);

    // Partition warps along the strided axis first to reduce strided iters
    static constexpr int kWarpS = kWarpIterS >= kWarpCount ? kWarpCount : kWarpIterS;
    static constexpr int kWarpC = kWarpCount > kWarpIterS ? kWarpCount / kWarpS : 1;

    static constexpr int kIterC = ceil_div(kWarpIterC, kWarpC);
    static constexpr int kIterS = ceil_div(kWarpIterS, kWarpS);

    // Allow partial tile when there is ONLY 1 iteration
    static_assert(kDimC % kWarpAccessC == 0 || kIterC == 1);

    // static_assert(kIterC > 0);
    // static_assert(kIterS > 0);

    static constexpr bool kAlignedC = (kDimC % kWarpAccessC == 0) && (kWarpIterC % kWarpC == 0);
    static constexpr bool kAlignedS = (kDimS % kWarpAccessS == 0) && (kWarpIterS % kWarpS == 0);

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

template<Order order, int M, int K>
__host__ __device__ static constexpr int2 idx2mk(int idx, pair<M, K>)
{
    if constexpr (order == kColMajor) {
        return {idx % M, idx / M};
    }
    else {
        return {idx / K, idx % K};
    }
}

enum class Partition
{
    kBlocked,
    kRaked,
};

template<int gM_, int gN_, Order order>
struct Blocked {
    static constexpr int gM = gM_;
    static constexpr int gN = gN_;

    // static_assert((gM - 1) * sM + (gN - 1) * sN == gM * gN - 1);

    static constexpr int dM = 1;
    static constexpr int dN = 1;

    static constexpr Partition pM = Partition::kBlocked;
    static constexpr Partition pN = Partition::kBlocked;

    template<int M, int N>
    __device__ static int2 get_offset(int idx, pair<M, N>)
    {
        constexpr int iM = ceil_div(M, gM);
        constexpr int iN = ceil_div(N, gN);

        // const int mi = idx / sM % gM;
        // const int ni = idx / sN % gN;

        const int2 mn = idx2mk<order>(idx, pair<gM, gN>{});
        return {mn.x * iM, mn.y * iN};
    }
};

template<int gM_, int gN_, Order order>
struct Raked {
    static constexpr int gM = gM_;
    static constexpr int gN = gN_;

    // static_assert((gM - 1) * sM + (gN - 1) * sN == gM * gN - 1);

    static constexpr int dM = gM;
    static constexpr int dN = gN;

    static constexpr Partition pM = Partition::kRaked;
    static constexpr Partition pN = Partition::kRaked;

    template<class Shape>
    __device__ static int2 get_offset(int idx, Shape)
    {
        return idx2mk<order>(idx, pair<gM, gN>{});
    }
};

template<int gM_, int gN_, Order order>
struct Blocked_C_Raked_S {
    static constexpr int gM = gM_;
    static constexpr int gN = gN_;

    static constexpr int dM = 1;
    static constexpr int dN = gN;

    static constexpr Partition pM = Partition::kBlocked;
    static constexpr Partition pN = Partition::kRaked;

    template<int M, int N>
    __device__ static int2 get_offset(int idx, pair<M, N>)
    {
        constexpr int iM = ceil_div(M, gM);

        const int2 mn = idx2mk<order>(idx, pair<gM, gN>{});
        return {mn.x * iM, mn.y};
    }
};

template<int C,
         int S,
         int AccessC,
         template<int, int, Order>
         typename Arrangement_,
         int WarpCount,
         int WarpThrC = std::min(WARP_SIZE, C / AccessC)>
struct ThreadMap_V2 {
    static constexpr int kDimC = C;
    static constexpr int kDimS = S;

    static constexpr int kWarpCount = WarpCount;
    static constexpr int kAccessC   = AccessC;

    static_assert(WarpThrC <= WARP_SIZE);

    static constexpr int kWarpThreadC = WarpThrC;
    static constexpr int kWarpThreadS = WARP_SIZE / kWarpThreadC;

    static constexpr int kWarpAccessC = kWarpThreadC * kAccessC;
    static constexpr int kWarpAccessS = kWarpThreadS;

    static constexpr int kWarpIterC = ceil_div(kDimC, kWarpAccessC);
    static constexpr int kWarpIterS = ceil_div(kDimS, kWarpAccessS);

    static constexpr int kWarpS = kWarpIterS >= kWarpCount ? kWarpCount : kWarpIterS;
    static constexpr int kWarpC = kWarpCount > kWarpIterS ? kWarpCount / kWarpS : 1;

    using Arrangement = Arrangement_<kWarpC, kWarpS, kColMajor>;

    static constexpr auto kPartitionM = Arrangement::pM;
    static constexpr auto kPartitionN = Arrangement::pN;

    static constexpr int kIterC = ceil_div(kWarpIterC, kWarpC);
    static constexpr int kIterS = ceil_div(kWarpIterS, kWarpS);

    static constexpr bool kAlignedC = (kDimC % kWarpAccessC == 0) && (kWarpIterC % kWarpC == 0);
    static constexpr bool kAlignedS = (kDimS % kWarpAccessS == 0) && (kWarpIterS % kWarpS == 0);

    static constexpr int kFootprintC = kWarpAccessC * kIterC;
    static constexpr int kFootprintS = kWarpAccessS * kIterS;

    static constexpr int kDeltaC = kWarpAccessC * Arrangement::dM;
    static constexpr int kDeltaS = kWarpAccessS * Arrangement::dN;

    __device__ static int2 get_offset(int warp_id, int lane_id)
    {
        const int2 warp_offset = Arrangement::get_offset(warp_id, pair<kWarpIterC, kWarpIterS>{});

        int warp_thr_offset_c = lane_id % kWarpThreadC;
        int warp_thr_offset_s = lane_id / kWarpThreadC;

        if constexpr (kWarpThreadC == WARP_SIZE) {
            warp_thr_offset_c = lane_id;
            warp_thr_offset_s = 0;
        }

        const int offset_c = warp_offset.x * kWarpAccessC + warp_thr_offset_c * kAccessC;
        const int offset_s = warp_offset.y * kWarpAccessS + warp_thr_offset_s;

        return {offset_c, offset_s};
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
    std::cout << "   aligned: (" << TMap::kAlignedC << "," << TMap::kAlignedS << ")\n";
}

}  // namespace

}  // namespace turbomind::gemm
