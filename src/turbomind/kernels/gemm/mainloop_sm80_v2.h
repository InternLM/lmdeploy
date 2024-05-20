#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include <cuda_pipeline_primitives.h>

namespace turbomind::gemm {

template<class Pointer, int Step, int Stages>
struct SmemIter {
    Pointer base_;
    Pointer pointer;
    int     pipe_iter_;

    __device__ SmemIter(Pointer base): base_{base}, pointer{base}, pipe_iter_{} {}

    __device__ void Advance()
    {
        pipe_iter_ += 1;
        pointer += Step;
        if (pipe_iter_ == Stages) {
            pipe_iter_ = 0;
            pointer    = base_;
        }
    }
};

template<int M_, int N_, int K_, class TiledMma, class OperandA, class OperandB, class OperandQ, int Stages_>
struct MainloopSm80_v2 {

    using MMA_Atom = typename TiledMma::MMA_Atom;

    using FragC = typename MMA_Atom::FragC[TiledMma::ITER_M][TiledMma::ITER_N];

    static constexpr int Stages = Stages_;

    static constexpr int CTA_M = M_;
    static constexpr int CTA_N = N_;
    static constexpr int CTA_K = K_;

    static constexpr int WARP_M = TiledMma::M;
    static constexpr int WARP_N = TiledMma::N;
    static constexpr int WARP_K = TiledMma::K;

    static constexpr int G = 128;

    using Ta = typename OperandA::Dtype;
    using Tb = typename OperandB::Dtype;
    using Tq = typename OperandQ::Dtype;

    using SmemLayoutA = typename OperandA::SmemLayout;
    using SmemLayoutB = typename OperandB::SmemLayout;
    // using SmemLayoutQ = typename OperandQ::SmemLayout;

    using SmemCopyA = typename OperandA::SmemCopy;
    using SmemCopyB = typename OperandB::SmemCopy;

    using SmemAccessorA = SmemAccessor<Ta, SmemLayoutA>;
    using SmemAccessorB = SmemAccessor<Tb, SmemLayoutB>;

    using GmemIterA = typename OperandA::GmemIter;
    using GmemIterB = typename OperandB::GmemIter;
    using GmemIterQ = typename OperandQ::GmemIter;

    static constexpr auto LayoutA = OperandA::Layout;
    static constexpr auto LayoutB = OperandB::Layout;
    static constexpr auto LayoutQ = OperandQ::Layout;

    static constexpr int WARP_CNT_M = M_ / TiledMma::M;
    static constexpr int WARP_CNT_N = N_ / TiledMma::N;
    static constexpr int WARP_CNT_K = K_ / TiledMma::K;
    static constexpr int WARP_CNT   = WARP_CNT_M * WARP_CNT_N * WARP_CNT_K;

    static constexpr int kMaxPrefetchIter =
        std::min(ceil_div(std::max(GmemIterA::ITER_S, GmemIterB::ITER_S), 4), TiledMma::ITER_K);

    static constexpr int kBatchA = ceil_div(GmemIterA::ITER_S, kMaxPrefetchIter);
    static constexpr int kBatchB = ceil_div(GmemIterB::ITER_S, kMaxPrefetchIter);
    // static constexpr int kBatchQ = ceil_div(GmemIterQ::ITER_S, kMaxPrefetchIter);

    struct SharedStorage {
        __align__(16) Array<Ta, Stages * SmemLayoutA::kSize> A;
        __align__(16) Array<Tb, Stages * SmemLayoutB::kSize> B;
    };

    __device__ void Wait()
    {
        __pipeline_wait_prior(Stages - 2);
        __syncthreads();
    }

    template<class GmemIter, class SmemIter>
    __device__ void AdvanceSmemStage(GmemIter& gmem_iter, SmemIter& smem_iter)
    {
        gmem_iter.smem_data_ = smem_iter.pointer;
        smem_iter.Advance();
    }

    __device__ void operator()(
        GmemIterA& gmem_A, GmemIterB& gmem_B, GmemIterQ& gmem_Q, FragC& frag_C, int tile_iter, SharedStorage& storage)
    {
        typename MMA_Atom::FragA frag_A[TiledMma::ITER_K][TiledMma::ITER_M];
        typename MMA_Atom::FragB frag_B[TiledMma::ITER_K][TiledMma::ITER_N];

        SmemIter<get_pointer_type<Ta>, SmemLayoutA::kSize, Stages> smem_A{storage.A.data()};
        SmemIter<get_pointer_type<Tb>, SmemLayoutB::kSize, Stages> smem_B{storage.B.data()};

        // a separate counter tends to generate better code
        int gmem_iter = tile_iter;
        int gmem_mask = true;

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            AdvanceSmemStage(gmem_A, smem_A);
            AdvanceSmemStage(gmem_B, smem_B);
            gmem_A.ClearSmem();
            gmem_B.ClearSmem();
        }
        // r: 0, w:s-1

        __syncthreads();

        PRAGMA_UNROLL
        for (int stage = 0; stage < Stages - 1; ++stage) {
            AdvanceSmemStage(gmem_A, smem_A);
            AdvanceSmemStage(gmem_B, smem_B);
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

        constexpr bool kFusePrefetch = true;

        auto prefetch = [&](int k) {
            if constexpr (kFusePrefetch) {
                int batch_A = min((k + 1) * kBatchA, GmemIterA::ITER_S) - k * kBatchA;
                int batch_B = min((k + 1) * kBatchB, GmemIterB::ITER_S) - k * kBatchB;
                gmem_A.Prefetch(k * kBatchA, batch_A, gmem_mask);
                gmem_B.Prefetch(k * kBatchB, batch_B, gmem_mask);
                if (k == TiledMma::ITER_K - 1) {
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
            AdvanceSmemStage(gmem_A, smem_A);
            AdvanceSmemStage(gmem_B, smem_B);
        };

        const int warp_id  = threadIdx.x / WARP_SIZE;
        const int offset_m = warp_id_m(warp_id) * TiledMma::M;
        const int offset_n = warp_id_n(warp_id) * TiledMma::N;
        const int offset_k = warp_id_k(warp_id) * TiledMma::K;

        auto Load = [&](int k) {
            SmemCopyA::copy(SmemAccessorA{smem_A.pointer},
                            frag_A[k][0].data(),
                            offset_m,
                            offset_k + k * SmemCopyA::Atom::kWarpAccessC);
            SmemCopyB::copy(SmemAccessorB{smem_B.pointer},
                            frag_B[k][0].data(),
                            offset_n,
                            offset_k + k * SmemCopyB::Atom::kWarpAccessC);
        };

        advance_and_wait_smem_stage();
        // r: 0, w:-1

        Load(0);

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
            constexpr int ITER_K = TiledMma::ITER_K;
            static_assert(ITER_K > 1);

            PRAGMA_UNROLL
            for (int k = 0; k < ITER_K; ++k) {
                // preload for next iter
                Load((k + 1) % ITER_K);
                PRAGMA_UNROLL
                for (int n = 0; n < TiledMma::ITER_N; ++n) {
                    PRAGMA_UNROLL
                    for (int m = 0; m < TiledMma::ITER_M; ++m) {
                        MMA_Atom::fma(frag_C[m][n], frag_A[k][m], frag_B[k][n], frag_C[m][n]);
                    }
                }
                prefetch((k + 1) % ITER_K);
                if (k + 1 == ITER_K - 1) {
                    advance_and_wait_smem_stage();
                }
            }
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
    }

    __device__ static int warp_id_m(int warp_id)
    {
        if constexpr (WARP_CNT_M == 1) {
            return 0;
        }
        else {
            return warp_id % WARP_CNT_M;
        }
    }

    __device__ static int warp_id_n(int warp_id)
    {
        if constexpr (WARP_CNT_N == 1) {
            return 0;
        }
        else {
            return warp_id / WARP_CNT_M % WARP_CNT_N;
        }
    }

    __device__ static int warp_id_k(int warp_id)
    {
        if constexpr (WARP_CNT_K == 1) {
            return 0;
        }
        else {
            return warp_id / WARP_CNT_M / WARP_CNT_N;
        }
    }

    template<class Tc, class Func>
    __device__ static void StoreC(FragC& frag_C, SharedStorage& storage, Func&& func)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;

        const int warp_idx_m = warp_id_m(warp_id);
        const int warp_idx_n = warp_id_n(warp_id);
        // const int warp_idx_k = warp_id_k(warp_id);

        const int warp_offset_m = warp_idx_m * WARP_M;
        const int warp_offset_n = warp_idx_n * WARP_N;

        static_assert(WARP_CNT_K == 1);

        PRAGMA_UNROLL
        for (int m = 0; m < TiledMma::ITER_M; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < TiledMma::ITER_N; ++n) {
                MMA_Atom::foreach_C(frag_C[m][n], [&](auto vec, int mi, int ni) {
                    ((Func&&)func)(warp_offset_m + m * MMA_Atom::M + mi,  //
                                   warp_offset_n + n * MMA_Atom::N + ni,
                                   cast<Tc>(vec));
                });
            }
        }
    }
};

}  // namespace turbomind::gemm