// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "cute/algorithm/gemm.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"

namespace turbomind::gemm::detail {

// Commit grouping and retained WGMMA-group depth are independent policies.
// WaitGroups == -1 preserves the established default: split issue paths retain
// one group, while an unsplit tile retains two.
template<int NSlices, bool SeparateAtoms, int WaitGroups = -1>
struct GmmaIssue {
    static_assert(NSlices >= 1);
    static constexpr int kWaitGroups = WaitGroups >= 0 ? WaitGroups : (NSlices > 1 || SeparateAtoms ? 1 : 2);
    static_assert(kWaitGroups >= 0 && kWaitGroups <= 7);

    template<class Mma, class TensorA, class TensorB, class TensorC>
    __device__ __forceinline__ static void run(Mma& mma, const TensorA& a, const TensorB& b, TensorC& c)
    {
        static_assert(cute::size<2>(typename TensorC::layout_type{}) == NSlices);
        if constexpr (NSlices > 1 || SeparateAtoms) {
            CUTE_UNROLL
            for (int mma_n = 0; mma_n < cute::size<2>(c); ++mma_n) {
                CUTE_UNROLL
                for (int mma_m = 0; mma_m < cute::size<1>(c); ++mma_m) {
                    cute::warpgroup_arrive();
                    cute::gemm(mma, a(cute::_, mma_m), b(cute::_, mma_n), c(cute::_, mma_m, mma_n));
                    mma.accumulate_ = cute::GMMA::ScaleOut::One;
                    cute::warpgroup_commit_batch();
                    cute::warpgroup_wait<kWaitGroups>();
                }
            }
        }
        else {
            cute::warpgroup_arrive();
            cute::gemm(mma, a, b, c);
            mma.accumulate_ = cute::GMMA::ScaleOut::One;
            cute::warpgroup_commit_batch();
            cute::warpgroup_wait<kWaitGroups>();
        }
    }
};

}  // namespace turbomind::gemm::detail
