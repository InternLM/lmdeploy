#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include <cuda_pipeline_primitives.h>

namespace turbomind::gemm {

template<class Impl_>
struct Mainloop_sm80 {

    using Impl = Impl_;

    using T = typename Impl::T;

    using FragC = typename Impl::FragC;

    using SharedStorage = typename Impl::SharedStorage;

    static constexpr int Stages = Impl::Stages;

    using ThreadMapA = typename Impl::ThreadMapA;
    using ThreadMapB = typename Impl::ThreadMapB;

    using SmemLayoutA = typename Impl::SmemLayoutA;
    using SmemLayoutB = typename Impl::SmemLayoutB;

    using GmemIterA = GmemIteratorSm80<T, ThreadMapA, SmemLayoutA, 0>;
    using GmemIterB = GmemIteratorSm80<T, ThreadMapB, SmemLayoutB, 1>;

    static constexpr int kBatchA = (ThreadMapA::kIterS + Impl::ITER_K - 1) / Impl::ITER_K;
    static constexpr int kBatchB = (ThreadMapB::kIterS + Impl::ITER_K - 1) / Impl::ITER_K;

    __device__ void Wait()
    {
        __pipeline_wait_prior(Stages - 2);
        __syncthreads();
    }

    template<class GmemIter, class SmemIter>
    __device__ void AdvanceSmemStage(GmemIter& gmem_iter, SmemIter& smem_iter)
    {
        gmem_iter.smem_data_ = smem_iter.data;
        smem_iter.Advance();
    }

    __device__ void
    operator()(GmemIterA& gmem_A, GmemIterB& gmem_B, FragC& frag_C, int tile_iter, SharedStorage& storage)
    {
        typename Impl::StateA state_A{storage};
        typename Impl::StateB state_B{storage};

        // a separate counter tends to generate better code
        int gmem_iter = tile_iter;
        int gmem_mask = true;

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            AdvanceSmemStage(gmem_A, state_A);
            AdvanceSmemStage(gmem_B, state_B);
            gmem_A.ClearSmem();
            gmem_B.ClearSmem();
        }
        // r: 0, w:s-1

        __syncthreads();

        PRAGMA_UNROLL
        for (int stage = 0; stage < Stages - 1; ++stage) {
            AdvanceSmemStage(gmem_A, state_A);
            AdvanceSmemStage(gmem_B, state_B);
            gmem_A.Prefetch(gmem_mask);
            gmem_B.Prefetch(gmem_mask);
            __pipeline_commit();
            gmem_A.Advance();
            gmem_B.Advance();
            if (--gmem_iter == 0) {
                gmem_mask = false;
            }
        }
        // r:-1, w:-2

        constexpr bool kFusePrefetch = false;

        auto prefetch = [&](int k) {
            if constexpr (kFusePrefetch) {
                int batch_A = min((k + 1) * kBatchA, ThreadMapA::kIterS) - k * kBatchA;
                int batch_B = min((k + 1) * kBatchB, ThreadMapB::kIterS) - k * kBatchB;
                gmem_A.Prefetch(k * kBatchA, batch_A, gmem_mask);
                gmem_B.Prefetch(k * kBatchB, batch_B, gmem_mask);
                if (k == Impl::ITER_K - 1) {
                    __pipeline_commit();
                    gmem_A.Advance();
                    gmem_B.Advance();
                    if (--gmem_iter == 0) {
                        gmem_mask = false;
                    }
                }
            }
        };

        auto advance_and_wait_smem_stage = [&] {
            Wait();
            AdvanceSmemStage(gmem_A, state_A);
            AdvanceSmemStage(gmem_B, state_B);
        };

        advance_and_wait_smem_stage();
        // r: 0, w:-1

        state_A.Load(0, 0);
        state_B.Load(0, 0);

        if constexpr (kFusePrefetch) {
            prefetch(0);
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter > 0; --tile_iter) {
            if constexpr (!kFusePrefetch) {
                gmem_A.Prefetch(gmem_mask);
                gmem_B.Prefetch(gmem_mask);
                __pipeline_commit();
                gmem_A.Advance();
                gmem_B.Advance();
                if (--gmem_iter == 0) {
                    gmem_mask = false;
                }
            }
            Impl::Compute(state_A, state_B, frag_C, 0, prefetch, advance_and_wait_smem_stage);
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
    }
};

}  // namespace turbomind::gemm