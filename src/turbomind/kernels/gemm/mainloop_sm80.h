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

    static constexpr int kMaxIterS  = std::max(ThreadMapA::kIterS, ThreadMapB::kIterS);
    static constexpr int kGmemBatch = (kMaxIterS + Impl::ITER_K - 1) / Impl::ITER_K;

    __device__ void Wait()
    {
        __pipeline_wait_prior(Stages - 2);
        __syncthreads();
    }

    template<class DataIter>
    __device__ void operator()(
        GmemIterA& gmem_A, GmemIterB& gmem_B, FragC& frag_C, DataIter& data_iter, int tile_iter, SharedStorage& storage)
    {
        Impl::SetSmem(gmem_A, gmem_B, storage);

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            gmem_A.ClearSmem();
            gmem_B.ClearSmem();
            gmem_A.Advance(Stages);
            gmem_B.Advance(Stages);
        }

        __syncthreads();

        PRAGMA_UNROLL
        for (int stage = 0; stage < Stages - 1; ++stage) {
            gmem_A.SetTile(data_iter);
            gmem_B.SetTile(data_iter);
            gmem_A.Prefetch(data_iter, 0);
            gmem_B.Prefetch(data_iter, 0);
            __pipeline_commit();
            gmem_A.Advance(Stages);
            gmem_B.Advance(Stages);
            ++data_iter;
        }

        typename Impl::StateA state_A{storage};
        typename Impl::StateB state_B{storage};

        Wait();

        state_A.Load(0, 0);
        state_B.Load(0, 0);

        PRAGMA_NO_UNROLL
        for (; tile_iter > 0; --tile_iter) {
            auto prefetch = [&](int k) {
                const int begin = k * kGmemBatch;
                gmem_A.Prefetch(data_iter, begin, kGmemBatch, 0);
                gmem_B.Prefetch(data_iter, begin, kGmemBatch, 0);
                const int end = begin + kGmemBatch;
                if (kMaxIterS <= end && end < kMaxIterS + kGmemBatch) {
                    __pipeline_commit();
                    ++data_iter;
                }
            };
            gmem_A.SetTile(data_iter);
            gmem_B.SetTile(data_iter);
            Impl::Compute(state_A, state_B, frag_C, 0, prefetch, [&] {
                Wait();
                gmem_A.Advance(Stages);
                gmem_B.Advance(Stages);
                state_A.Advance();
                state_B.Advance();
                state_A.Load(0, 0);
                state_B.Load(0, 0);
            });
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
    }
};

}  // namespace turbomind::gemm