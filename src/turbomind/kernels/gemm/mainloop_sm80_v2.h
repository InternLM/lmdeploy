// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include "src/turbomind/kernels/gemm/operand.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
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

template<int M_,
         int N_,
         int K_,
         class MMA,
         class OperandA_,
         class OperandB_,
         class OperandU_,
         class OperandV_,
         class Transform_,
         int Stages_>
struct MainloopSm80_v2 {

    using MMA_Atom = typename MMA::Atom;
    using MMA_Map  = typename MMA::Map;

    using FragC = typename MMA_Atom::FragC[MMA::kMmaIterM][MMA::kMmaIterN];

    static constexpr int Stages = Stages_;

    static constexpr int CTA_M = M_;
    static constexpr int CTA_N = N_;
    static constexpr int CTA_K = K_;

    static constexpr int WARP_M = MMA_Map::kFootprintM;
    static constexpr int WARP_N = MMA_Map::kFootprintN;
    static constexpr int WARP_K = MMA_Map::kFootprintK;

    static constexpr int WARPS = MMA::kThreadCount / WARP_SIZE;

    using OperandA = MakeOperand<OperandA_, IteratorSm80, CTA_M, CTA_K, WARPS>;
    using OperandU = MakeOperand<OperandU_, IteratorSm80, CTA_M, CTA_K, WARPS>;

    using OperandB = MakeOperand<OperandB_, IteratorSm80, CTA_N, CTA_K, WARPS>;
    using OperandV = MakeOperand<OperandV_, IteratorSm80, CTA_N, CTA_K, WARPS>;

    using Transform = Transform_;

    using Ta = typename OperandA::Dtype;
    using Tb = typename OperandB::Dtype;
    using Tu = typename OperandU::Dtype;
    using Tv = typename OperandV::Dtype;

    // primary  : AB
    // qparam   : UV
    // secondary: XY

    using SmemLayoutA = typename OperandA::SmemLayout;
    using SmemLayoutB = typename OperandB::SmemLayout;
    using SmemLayoutU = typename OperandU::SmemLayout;
    using SmemLayoutV = typename OperandV::SmemLayout;

    using SmemCopyA = SmemCopy<OperandA, MMA_Map::kFootprintM, MMA_Map::TileK, MMA_Map::kDeltaM, MMA_Map::kDeltaK>;
    using SmemCopyU = SmemCopy<OperandU, MMA_Map::kFootprintM, MMA_Map::TileK, MMA_Map::kDeltaM, MMA_Map::kDeltaK>;
    using SmemCopyB = SmemCopy<OperandB, MMA_Map::kFootprintN, MMA_Map::TileK, MMA_Map::kDeltaN, MMA_Map::kDeltaK>;
    using SmemCopyV = SmemCopy<OperandV, MMA_Map::kFootprintN, MMA_Map::TileK, MMA_Map::kDeltaN, MMA_Map::kDeltaK>;

    using SmemAccessorA = SmemAccessor<Ta, SmemLayoutA>;
    using SmemAccessorB = SmemAccessor<Tb, SmemLayoutB>;
    using SmemAccessorU = SmemAccessor<Tu, SmemLayoutU>;
    using SmemAccessorV = SmemAccessor<Tv, SmemLayoutV>;

    using GmemIterA = typename OperandA::GmemIter;
    using GmemIterB = typename OperandB::GmemIter;
    using GmemIterU = typename OperandU::GmemIter;
    using GmemIterV = typename OperandV::GmemIter;

    static constexpr int kMaxPrefetchIter =
        std::min(ceil_div(std::max(GmemIterA::ITER_S, GmemIterB::ITER_S), 4), MMA::kTileIterK);

    static constexpr int kBatchA = ceil_div(GmemIterA::ITER_S, kMaxPrefetchIter);
    static constexpr int kBatchB = ceil_div(GmemIterB::ITER_S, kMaxPrefetchIter);
    static constexpr int kBatchU = ceil_div(GmemIterU::ITER_S, kMaxPrefetchIter);
    static constexpr int kBatchV = ceil_div(GmemIterV::ITER_S, kMaxPrefetchIter);

    struct SharedStorage {
        __align__(16) Array<Ta, Stages * SmemLayoutA::kSize> A;
        __align__(16) Array<Tb, Stages * SmemLayoutB::kSize> B;
        __align__(16) Array<Tu, Stages * SmemLayoutU::kSize> U;
        __align__(16) Array<Tv, Stages * SmemLayoutV::kSize> V;
    };

    __device__ void Wait()
    {
        __pipeline_wait_prior(Stages - 2);
        __syncthreads();
    }

    template<class GmemIter, class SmemIter>
    __device__ void _advance_smem(GmemIter& gmem_iter, SmemIter& smem_iter)
    {
        gmem_iter.smem_data_ = smem_iter.pointer;
        smem_iter.Advance();
    }

    template<class A, class B, class U, class V>
    struct Binding {
        A&         a;
        B&         b;
        U&         u;
        V&         v;
        __device__ Binding(A& a, B& b, U& u, V& v): a{a}, b{b}, u{u}, v{v} {}  // CTAD
    };

    // zip with
    template<class BindingG, class BindingS>
    __device__ void AdvanceSmemStage(BindingG& g, BindingS& s)
    {
        _advance_smem(g.a, s.a);
        _advance_smem(g.b, s.b);
        _advance_smem(g.u, s.u);
        _advance_smem(g.v, s.v);
    }

    template<class Binding>
    __device__ void ClearSmem(Binding& g)
    {
        g.a.ClearSmem();
        g.b.ClearSmem();
        g.u.ClearSmem();
        g.v.ClearSmem();
    }

    template<class Binding>
    __device__ void Prefetch(Binding& g, bool mask)
    {
        g.a.Prefetch(mask);
        g.b.Prefetch(mask);
        g.u.Prefetch(mask);
        g.v.Prefetch(mask);
    }

    template<class Binding>
    __device__ void Prefetch(Binding& g, int k, bool mask)
    {
        int batch_A = min((k + 1) * kBatchA, GmemIterA::ITER_S) - k * kBatchA;
        int batch_B = min((k + 1) * kBatchB, GmemIterB::ITER_S) - k * kBatchB;
        int batch_U = min((k + 1) * kBatchU, GmemIterU::ITER_S) - k * kBatchU;
        int batch_V = min((k + 1) * kBatchV, GmemIterV::ITER_S) - k * kBatchV;
        g.a.Prefetch(k * kBatchA, batch_A, mask);
        g.b.Prefetch(k * kBatchB, batch_B, mask);
        g.u.Prefetch(k * kBatchU, batch_U, mask);
        g.v.Prefetch(k * kBatchV, batch_V, mask);
    }

    template<class Binding>
    __device__ void AdvanceGmemStage(Binding& g)
    {
        g.a.Advance();
        g.b.Advance();
        g.u.Advance();
        g.v.Advance();
    }

    __device__ void operator()(GmemIterA&     gmem_A,
                               GmemIterB&     gmem_B,
                               GmemIterU&     gmem_U,
                               GmemIterU&     gmem_V,
                               FragC&         frag_C,
                               int            tile_iter,
                               SharedStorage& storage)
    {
        static_assert(MMA::kAtomK == 1);

        // mma_iter_x = tile_iter_x * atom_x
        typename MMA_Atom::FragA frag_A[MMA::kTileIterK][MMA::kMmaIterM];
        typename MMA_Atom::FragB frag_B[MMA::kTileIterK][MMA::kMmaIterN];

        typename SmemCopyA::Frag data_A[MMA::kTileIterK];
        typename SmemCopyB::Frag data_B[MMA::kTileIterK];
        typename SmemCopyU::Frag data_U[MMA::kTileIterK];
        typename SmemCopyV::Frag data_V[MMA::kTileIterK];

        SmemIter<get_pointer_type<Ta>, SmemLayoutA::kSize, Stages> smem_A{storage.A.data()};
        SmemIter<get_pointer_type<Tb>, SmemLayoutB::kSize, Stages> smem_B{storage.B.data()};
        SmemIter<get_pointer_type<Tu>, SmemLayoutU::kSize, Stages> smem_U{storage.U.data()};
        SmemIter<get_pointer_type<Tv>, SmemLayoutV::kSize, Stages> smem_V{storage.V.data()};

        // a separate counter tends to generate better code
        int gmem_iter = tile_iter;
        int gmem_mask = true;

        Binding gmem_iters{gmem_A, gmem_B, gmem_U, gmem_V};
        Binding smem_iters{smem_A, smem_B, smem_U, smem_V};

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            AdvanceSmemStage(gmem_iters, smem_iters);
            ClearSmem(gmem_iters);
        }

        // r: 0, w:s-1

        __syncthreads();

        constexpr bool kFusePrefetch = true;

        auto prefetch_stage = [&] {
            Prefetch(gmem_iters, gmem_mask);
            __pipeline_commit();
            AdvanceGmemStage(gmem_iters);
            if (--gmem_iter == 0) {
                gmem_mask = false;
            }
        };

        auto prefetch_batch = [&](int k) {
            Prefetch(gmem_iters, k, gmem_mask);
            if (k == MMA::kTileIterK - 1) {
                __pipeline_commit();
                AdvanceGmemStage(gmem_iters);
                if (--gmem_iter == 0) {
                    gmem_mask = false;
                }
            }
        };

        auto advance_and_wait_smem_stage = [&] {
            Wait();
            AdvanceSmemStage(gmem_iters, smem_iters);
        };

        const int3 offset_mnk = MMA::get_offset(threadIdx.x);
        const int  offset_m   = offset_mnk.x;
        const int  offset_n   = offset_mnk.y;
        const int  offset_k   = offset_mnk.z;

        auto preload = [&](int k) {
            const int current_k = offset_k + k * MMA_Atom::K;
            SmemCopyA::copy(SmemAccessorA{smem_A.pointer}, data_A[k], {offset_m, current_k});
            SmemCopyU::copy(SmemAccessorU{smem_U.pointer}, data_U[k], {offset_m, current_k});
            SmemCopyB::copy(SmemAccessorB{smem_B.pointer}, data_B[k], {offset_n, current_k});
            SmemCopyV::copy(SmemAccessorV{smem_V.pointer}, data_V[k], {offset_n, current_k});
        };

        PRAGMA_UNROLL
        for (int stage = 0; stage < Stages - 1; ++stage) {
            AdvanceSmemStage(gmem_iters, smem_iters);
            prefetch_stage();
        }
        // r:-1, w:-2

        advance_and_wait_smem_stage();
        // r: 0, w:-1

        preload(0);
        Transform::transform(frag_A, frag_B, 0, data_A, data_B, data_U, data_V);

        if constexpr (kFusePrefetch) {
            prefetch_batch(0);
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter > 0; --tile_iter) {
            if constexpr (!kFusePrefetch) {
                prefetch_stage();
            }
            constexpr int ITER_K = MMA::kTileIterK;
            static_assert(ITER_K > 1);

            PRAGMA_UNROLL
            for (int k = 0; k < ITER_K; ++k) {
                // preload for next iter
                preload((k + 1) % ITER_K);
                PRAGMA_UNROLL
                for (int n = 0; n < MMA::kMmaIterN; ++n) {
                    PRAGMA_UNROLL
                    for (int m = 0; m < MMA::kMmaIterM; ++m) {
                        MMA_Atom::fma(frag_C[m][n], frag_A[k][m], frag_B[k][n], frag_C[m][n]);
                    }
                }
                if constexpr (kFusePrefetch) {
                    prefetch_batch((k + 1) % ITER_K);
                }
                if (k + 1 == ITER_K - 1) {
                    advance_and_wait_smem_stage();
                }
                Transform::transform(frag_A, frag_B, (k + 1) % ITER_K, data_A, data_B, data_U, data_V);
            }
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
    }

    template<class Tc, class Func>
    __device__ static void StoreC(FragC& frag_C, SharedStorage& storage, Func&& func)
    {
        const int3 offset_mnk = MMA::get_offset(threadIdx.x);
        const int  offset_m   = offset_mnk.x;
        const int  offset_n   = offset_mnk.y;
        // const int  offset_k   = offset_mnk.z;

        // static_assert(WARP_CNT_K == 1);

        PRAGMA_UNROLL
        for (int m = 0; m < MMA::kMmaIterM; ++m) {
            PRAGMA_UNROLL
            for (int n = 0; n < MMA::kMmaIterN; ++n) {
                MMA_Atom::foreach_C(frag_C[m][n], [&](auto vec, int mi, int ni) {
                    ((Func&&)func)(offset_m + m * MMA_Atom::M + mi,  //
                                   offset_n + n * MMA_Atom::N + ni,
                                   cast<Tc>(vec));
                });
            }
        }
    }
};

}  // namespace turbomind::gemm