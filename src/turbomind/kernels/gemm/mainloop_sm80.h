#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include <cuda_pipeline_primitives.h>

namespace turbomind::gemm {

template<class Impl_>
struct Mainloop_sm80 {

    using Impl = Impl_;

    using T  = typename Impl::T;
    using Tb = typename Impl::Tb;
    using Tq = typename Impl::Tq;

    using FragC = typename Impl::FragC;

    using SharedStorage = typename Impl::SharedStorage;

    static constexpr int Stages = Impl::Stages;

    using ThreadMapA = typename Impl::ThreadMapA;
    using ThreadMapB = typename Impl::ThreadMapB;
    using ThreadMapQ = typename Impl::ThreadMapQ;

    using SmemLayoutA = typename Impl::SmemLayoutA;
    using SmemLayoutB = typename Impl::SmemLayoutB;
    using SmemLayoutQ = typename Impl::SmemLayoutQ;

    using GmemIterA = GmemIteratorSm80<T, ThreadMapA, SmemLayoutA, 0>;
    using GmemIterB = GmemIteratorSm80<Tb, ThreadMapB, SmemLayoutB, 1>;
    using GmemIterQ = GmemIteratorSm80<Tq, ThreadMapQ, SmemLayoutQ, 2, Impl::G_CTA, std::is_same_v<T, Tb>>;

    static constexpr bool kUseGmemQ = !std::is_same_v<T, Tb>;

    static constexpr int kBatchA = (ThreadMapA::kIterS + Impl::ITER_K - 1) / Impl::ITER_K;
    static constexpr int kBatchB = (ThreadMapB::kIterS + Impl::ITER_K - 1) / Impl::ITER_K;
    static constexpr int kBatchQ = (ThreadMapB::kIterS + Impl::ITER_K - 1) / Impl::ITER_K;

    static constexpr int G_CTA = Impl::G_CTA;

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

    __device__ void operator()(
        GmemIterA& gmem_A, GmemIterB& gmem_B, GmemIterQ& gmem_Q, FragC& frag_C, int tile_iter, SharedStorage& storage)
    {
        typename Impl::StateA state_A{storage};
        typename Impl::StateQ state_Q{storage};
        typename Impl::StateB state_B{storage};

        state_B.state_Q = &state_Q;

        // a separate counter tends to generate better code
        int gmem_iter = tile_iter;
        int gmem_mask = true;

        // w: 012345678
        // r:    012345

        // 4-stage   *
        //    012345678
        // w: 0___0___0___0
        // r:    0___0___0

        // w: 0___1___2___3
        // r:    0___1___2
        // w: 0_2_0_2_0_2
        // r:    0_2_0_2_0_2

        // 3-stage
        // k: 012345678
        // w: 0___1___2
        // r:   0___1___2

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            AdvanceSmemStage(gmem_A, state_A);
            AdvanceSmemStage(gmem_B, state_B);
            AdvanceSmemStage(gmem_Q, state_Q);
            gmem_A.ClearSmem();
            gmem_B.ClearSmem();
            gmem_Q.ClearSmem();
        }
        // r: 0, w:s-1

        __syncthreads();

        PRAGMA_UNROLL
        for (int stage = 0; stage < Stages - 1; ++stage) {
            AdvanceSmemStage(gmem_A, state_A);
            AdvanceSmemStage(gmem_B, state_B);
            AdvanceSmemStage(gmem_Q, state_Q);
            gmem_A.Prefetch(gmem_mask);
            gmem_B.Prefetch(gmem_mask);
            gmem_Q.Prefetch(gmem_mask);
            __pipeline_commit();
            gmem_A.Advance();
            gmem_B.Advance();
            gmem_Q.Advance();
            if (--gmem_iter == 0) {
                gmem_mask = false;
            }
        }
        // r:-1, w:-2

        constexpr bool kFusePrefetch = true;

        auto prefetch = [&](int k) {
            if constexpr (kFusePrefetch) {
                int batch_A = min((k + 1) * kBatchA, ThreadMapA::kIterS) - k * kBatchA;
                int batch_B = min((k + 1) * kBatchB, ThreadMapB::kIterS) - k * kBatchB;
                int batch_Q = min((k + 1) * kBatchQ, ThreadMapB::kIterS) - k * kBatchQ;
                gmem_A.Prefetch(k * kBatchA, batch_A, gmem_mask);
                gmem_B.Prefetch(k * kBatchB, batch_B, gmem_mask);
                gmem_Q.Prefetch(k * kBatchB, batch_Q, gmem_mask);
                if (k == Impl::ITER_K - 1) {
                    __pipeline_commit();
                    gmem_A.Advance();
                    gmem_B.Advance();
                    gmem_Q.Advance();
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
            AdvanceSmemStage(gmem_Q, state_Q);
        };

        advance_and_wait_smem_stage();
        // r: 0, w:-1

        // start counting Q iters
        state_Q.counting = true;

        state_A.Load(0, 0);
        state_B.Load(0, 0);
        state_B.Transform(0);

        if constexpr (kFusePrefetch) {
            prefetch(0);
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter > 0; --tile_iter) {
            if constexpr (!kFusePrefetch) {
                gmem_A.Prefetch(gmem_mask);
                gmem_B.Prefetch(gmem_mask);
                gmem_Q.Prefetch(gmem_mask);
                __pipeline_commit();
                gmem_A.Advance();
                gmem_B.Advance();
                gmem_Q.Advance();
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