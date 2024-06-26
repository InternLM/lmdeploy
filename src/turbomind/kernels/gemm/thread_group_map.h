// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/math.h"

#include <iostream>

namespace turbomind::gemm {

template<int M_, int N_, int K_, int TM, int TN, int TK, int GM, int GN, int GK>
struct RakedThreadGroupMap {
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int K = K_;

    static constexpr int TileM = TM;
    static constexpr int TileN = TN;
    static constexpr int TileK = TK;

    static constexpr int kGroupM = GM;
    static constexpr int kGroupN = GN;
    static constexpr int kGroupK = GK;

    static constexpr int kGroupCount = GM * GN * GK;

    static constexpr int M1 = GM * TM;
    static constexpr int N1 = GN * TN;
    static constexpr int K1 = GK * TK;

    static constexpr int kIterM = M / M1;
    static constexpr int kIterN = N / N1;
    static constexpr int kIterK = K / K1;

    static constexpr int kFootprintM = kIterM * TM;
    static constexpr int kFootprintN = kIterN * TN;
    static constexpr int kFootprintK = kIterK * TK;

    static constexpr int kDeltaM = TM;
    static constexpr int kDeltaN = TN;
    static constexpr int kDeltaK = TK;

    __device__ static int3 get_offset(int group_id)
    {
        const int m = group_id % GM;
        const int n = group_id / GM % GN;
        const int k = group_id / GM / GN;
        return {m * kFootprintM, n * kFootprintN, k * kFootprintK};
    }
};

namespace {

template<class TMap>
void Print_(TMap)
{
    std::cout << "M, N, K = " << TMap::M << " " << TMap::N << " " << TMap::K << "\n";
    std::cout << "TM, TN, TK = " << TMap::TileM << " " << TMap::TileN << " " << TMap::TileK << "\n";
    std::cout << "group count = " << TMap::kGroupCount << "\n";
    std::cout << "M1, N1, K1 = " << TMap::M1 << " " << TMap::N1 << " " << TMap::K1 << "\n";
    std::cout << "itM, itN, itK = " << TMap::kIterM << " " << TMap::kIterN << " " << TMap::kIterK << "\n";
    std::cout << "fpM, fpN, fpK = " << TMap::kFootprintM << " " << TMap::kFootprintN << " " << TMap::kFootprintK
              << "\n";
    std::cout << "dM, dN, dK = " << TMap::kDeltaM << " " << TMap::kDeltaN << " " << TMap::kDeltaK << "\n";
}

}  // namespace

/// TODO: Striped partition?

}  // namespace turbomind::gemm