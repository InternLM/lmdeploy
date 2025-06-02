#pragma once

#include <numeric>
#include <utility>

#include <cuda_fp8.h>
#include <cuda_pipeline_primitives.h>

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/arch/mma_sm90_desc.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/atom/mma_traits.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"

#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/meta.h"
#include "src/turbomind/kernels/core/smem.h"

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/cp_async.h"
#include "src/turbomind/kernels/gemm/iterator_sm90.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/scheduler.cuh"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

#include "src/turbomind/kernels/gemm/sm90_utils.h"

namespace turbomind::gemm {

struct GemmUniversalSm90_v5 {

    static constexpr bool kDebug = false;

    using Arch = Sm90;

    using MMA_Atom = GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<>;

    static constexpr typename cute::MMA_Traits<MMA_Atom>::Shape_MNK MMA_Shape{};

    static constexpr int MMA_ATOM_M = cute::get<0>(MMA_Shape);
    static constexpr int MMA_ATOM_N = cute::get<1>(MMA_Shape);
    static constexpr int MMA_ATOM_K = cute::get<2>(MMA_Shape);

    static constexpr int WARPGORUPS = 4;

    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int TILE_K = 128;

    static constexpr int WG_M = 1;
    static constexpr int WG_N = 2;

    static constexpr int WG_TILE_M = TILE_M / WG_M;
    static constexpr int WG_TILE_N = TILE_N / WG_N;

    static constexpr int kSchedStepSize = 2;

    static constexpr int MMA_ITER_M = WG_TILE_M / MMA_ATOM_M;
    static constexpr int MMA_ITER_N = WG_TILE_N / MMA_ATOM_N;
    static constexpr int MMA_ITER_K = TILE_K / MMA_ATOM_K;

    static constexpr int kMulticastA = 1;
    static constexpr int kMulticastB = 2;

    static constexpr int kClusterSize = kMulticastA * kMulticastB;

    static constexpr int Stages = 5;

    static constexpr bool kSplitK     = false;
    static constexpr int  kChunkSizeK = TILE_K;

    static constexpr int WARPGROUP_SIZE = 128;
    static constexpr int kMathGroupSize = 256;

    static constexpr int CTA_SIZE = WARPGROUP_SIZE * (WARPGORUPS + 1);

    using Ta = __nv_fp8_e4m3;
    using Tb = __nv_fp8_e4m3;
    using Tc = nv_bfloat16;

    using Tu = float;
    using Tv = float;

    using Cluster = arch::Cluster<kMulticastB, kMulticastA, kRowMajor>;

    static constexpr auto is_grouped_gemm = false;

    using Scheduler = TileScheduler<kRowMajor, Cluster, true, true, TILE_M, TILE_N, false>;

    static constexpr int kMulticastU = is_grouped_gemm ? 1 : kMulticastA;

    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBar = cutlass::arch::ClusterBarrier;

    static constexpr int MAX_K        = 32768;
    static constexpr int MAX_K_BLOCKS = cdiv(MAX_K, 128);

    static constexpr int kAlignmentU = 16 / sizeof(Tu);
    static constexpr int kBoxU       = TILE_M + (is_grouped_gemm ? kAlignmentU : 0);

    static constexpr int kTmaTxBytes =
        sizeof(Ta) * (TILE_M * TILE_K) + sizeof(Tb) * (TILE_N * TILE_K) + sizeof(Tu) * kBoxU;

    // ! Smem addr must be SBO aligned for TMA load/store
    struct SharedStorage {
        struct Source {
            __align__(1024) Array<Ta, Stages * TILE_M * TILE_K> A;
            __align__(1024) Array<Tb, Stages * TILE_N * TILE_K> B;
            __align__(1024) Tu U[Stages][round_up<int>(kBoxU, 128 / sizeof(Tu))];
            __align__(1024) Tv V[2][2][MAX_K_BLOCKS];
        };
        Source source;
        __align__(1024) Array<Tc, TILE_M * TILE_N> C;
        __align__(128) uint64_t producer_bar[Stages];
        __align__(128) uint64_t consumer_bar[Stages];
        int pipe_count[2];
    };

    static constexpr int kSmemSize = sizeof(SharedStorage);

    static constexpr int kSwizzleC = 2 * std::gcd(TILE_N, 128 / sizeof(Tc));

    using LayoutC = std::conditional_t<kSwizzleC >= 32,
                                       SmemLayoutV2<TILE_M, TILE_N, -1, kSwizzleC / sizeof(Tc)>,
                                       SmemLayoutV2<TILE_M, TILE_N>>;

    static constexpr int OUTER_N = std::gcd(MMA_ATOM_N, 128);

    __device__ void operator()(const CUtensorMap& tm_a,
                               const CUtensorMap& tm_b,
                               const CUtensorMap& tm_c,
                               const CUtensorMap& tm_u,
                               const CUtensorMap& tm_v,
                               const MatrixParam& param_A,
                               const MatrixParam& param_B,
                               const MatrixParam& param_U,
                               const MatrixParam& param_V,
                               const MatrixParam& param_C,
                               Scheduler          sched,
                               CUtensorMap*       tensormap_buf,
                               char*              smem_buf)
    {
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        uint64_t* producer_bar = storage.producer_bar;
        uint64_t* consumer_bar = storage.consumer_bar;

        if (threadIdx.x == 0) {
            PRAGMA_UNROLL
            for (int s = 0; s < Stages; ++s) {
                ProducerBar::init(&producer_bar[s], 1);
                ConsumerBar::init(&consumer_bar[s], kClusterSize * (kMathGroupSize / WARP_SIZE));
            }
            cutlass::arch::fence_view_async_shared();
            if constexpr (kClusterSize > 1) {
                cutlass::arch::fence_barrier_init();
            }
            PRAGMA_UNROLL
            for (int i = 0; i < 2; ++i) {
                storage.pipe_count[i] = 0;
            }
        }

        (kClusterSize > 1) ? cute::cluster_sync() : __syncthreads();

        const int wg_idx = cutlass::canonical_warp_group_idx();

        if (wg_idx == WARPGORUPS) {
            cutlass::arch::warpgroup_reg_dealloc<32>();

            static_assert(TILE_M % kMulticastA == 0);
            static_assert(TILE_N % kMulticastB == 0);

            const int warp_id = cutlass::canonical_warp_idx_sync();

            if (warp_id % 4 == 0) {

                Cluster cluster(cute::block_id_in_cluster().x);

                const int mc_offset_m = cluster.cta_n() * (TILE_M / kMulticastA);
                const int mc_offset_n = cluster.cta_m() * (TILE_N / kMulticastB);

                auto  smem_A = storage.source.A.data() + mc_offset_m * TILE_K;
                auto  smem_B = storage.source.B.data() + mc_offset_n * TILE_K;
                auto& smem_U = storage.source.U;

                sched.grid_init();

                cutlass::PipelineState<Stages> write_state{0, 1, 0};

                while (sched.next()) {
                    auto [valid_cta_tile_p, cluster_tile_p] = sched.is_valid_tile();

                    if (!cluster_tile_p) {
                        // OOB tile caused by swizzle pattern
                        continue;
                    }

                    const auto tile_offset              = sched.tile_offset();
                    const auto [iter_k_beg, iter_k_end] = sched.iter_k_range();

                    if (cute::elect_one_sync()) {

                        const int offset_k = iter_k_beg * TILE_K;

                        const uint16_t mask_A = cluster.mask_m();
                        const uint16_t mask_B = cluster.mask_n();

                        const int offset_m = tile_offset.x * TILE_M;
                        const int offset_n = tile_offset.y * TILE_N;

                        int k_iter = iter_k_end - iter_k_beg;

                        GmemIteratorSm90<kMulticastA> gmem_A{&tm_a, {offset_k, offset_m + mc_offset_m}, {TILE_K, 0}};
                        GmemIteratorSm90<kMulticastB> gmem_B{&tm_b, {offset_k, offset_n + mc_offset_n}, {TILE_K, 0}};

                        // column-major
                        GmemIteratorSm90<kMulticastA> gmem_U{&tm_u, {offset_m + mc_offset_m, offset_k / 128}, {0, 1}};

                        for (; k_iter > 0; --k_iter) {
                            int pipe = write_state.index();
                            ConsumerBar::wait(&consumer_bar[pipe], write_state.phase());
                            ProducerBar::arrive_and_expect_tx(&producer_bar[pipe], kTmaTxBytes);
                            gmem_A.Step(&producer_bar[pipe], &smem_A[pipe * TILE_M * TILE_K], mask_A);
                            gmem_B.Step(&producer_bar[pipe], &smem_B[pipe * TILE_N * TILE_K], mask_B);
                            gmem_U.Step(&producer_bar[pipe], &smem_U[pipe][0] + mc_offset_m, mask_A);
                            ++write_state;
                        }
                    }
                }
            }
        }
        else {
            cutlass::arch::warpgroup_reg_alloc<112>();

            sched.grid_init(kSchedStepSize);

            auto& smem_A = storage.source.A;
            auto& smem_B = storage.source.B;
            auto& smem_U = storage.source.U;

            const int math_group_idx = wg_idx / 2;

            const int wg_idx_m = WG_M > 1 ? wg_idx % 2 % WG_M : 0;
            const int wg_idx_n = WG_N > 1 ? wg_idx % 2 / WG_M : 0;

            auto smem_desc_A = make_smem_desc(&smem_A[wg_idx_m * WG_TILE_M * TILE_K], 1);
            auto smem_desc_B = make_smem_desc(&smem_B[wg_idx_n * WG_TILE_N * TILE_K], 1);

            SmemDescIterV2<Stages, ((sizeof(Ta) * TILE_M * TILE_K) >> 4)> smem_iter_A{smem_desc_A};
            SmemDescIterV2<Stages, ((sizeof(Tb) * TILE_N * TILE_K) >> 4)> smem_iter_B{smem_desc_B};

            constexpr int kStepMA = (sizeof(Ta) * MMA_ATOM_M * TILE_K) >> 4;
            constexpr int kStepNB = (sizeof(Tb) * MMA_ATOM_N * TILE_K) >> 4;
            constexpr int kStepKA = (sizeof(Ta) * MMA_ATOM_K) >> 4;
            constexpr int kStepKB = (sizeof(Tb) * MMA_ATOM_K) >> 4;

            const int thread_idx = threadIdx.x % kMathGroupSize;

            auto math_barrier_sync = [&](int phase, int alive = 1) {
                constexpr int base       = (int)cutlass::arch::ReservedNamedBarriers::FirstUserBarrier;
                const int     barrier_id = base + math_group_idx ^ phase;

                // if (thread_idx == 0) {
                //     printf("math_barrier_sync %2d%2d%4d\n", math_group_idx, math_group_idx ^ phase,
                //     (int)threadIdx.x);
                // }

                constexpr int threads = WARPGORUPS * WARPGROUP_SIZE;
                int           res     = 0;

                asm volatile("{\n"
                             "  .reg.pred p;\n"
                             "  setp.ne.b32 p, %3, 0;\n"
                             "  barrier.cta.red.or.pred p, %1, %2, p;\n"
                             "  selp.s32 %0, 1, 0, p;\n"
                             "}\n"
                             : "=r"(res)
                             : "r"(barrier_id), "r"(threads), "r"(alive));

                // if (thread_idx == 0) {
                //     printf(
                //         "math_barrier_sync %2d%2d%4d DONE\n", math_group_idx, math_group_idx ^ phase,
                //         (int)threadIdx.x);
                // }

                return res;
            };

            cutlass::arch::NamedBarrier math_group_barrier(kMathGroupSize, 2 + math_group_idx);  // 2,3

            sched.next(math_group_idx);

            if (math_group_idx == 1) {
                math_barrier_sync(1);
            }

            while (sched.next(kSchedStepSize)) {
                auto [cta_tile_p, cluster_tile_p] = sched.is_valid_tile();

                if (!cluster_tile_p) {
                    // OOB tile caused by swizzle pattern
                    continue;
                }

                MMA_Atom::CRegisters frag_C[MMA_ITER_N];
                MMA_Atom::CRegisters accum_C[MMA_ITER_M][MMA_ITER_N]{};

                const auto tile_offset              = sched.tile_offset();
                const auto [iter_k_beg, iter_k_end] = sched.iter_k_range();

                const auto [M, N, K, L] = sched.gemm_shape();

                const int offset_m = tile_offset.x * TILE_M;
                const int offset_n = tile_offset.y * TILE_N;

                // if (thread_idx == 0) {
                //     printf("TILE %d %d\n", offset_m, offset_n);
                // }

                int k_iter = iter_k_end - iter_k_beg;

                const int warp_id = threadIdx.x / WARP_SIZE;
                const int lane_id = threadIdx.x % WARP_SIZE;

                cutlass::PipelineState<Stages> pipe_state{};

                auto consumer_arrive = [&] {
                    __syncwarp();
                    if constexpr (kClusterSize > 1) {
                        ConsumerBar::arrive(&consumer_bar[pipe_state.index()], lane_id, lane_id < kClusterSize);
                    }
                    else {
                        if (lane_id == 0) {
                            ConsumerBar::arrive(&consumer_bar[pipe_state.index()]);
                        }
                    }
                };

                if constexpr (kClusterSize > 1) {
                    if (!cta_tile_p) {  // other CTAs in the cluster are still alive
                        math_barrier_sync(0);
                        pipe_state.advance(storage.pipe_count[math_group_idx ^ 1]);
                        for (; k_iter > 0; --k_iter) {
                            ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                            consumer_arrive();
                            ++pipe_state;
                        }
                        // const int thread_idx = threadIdx.x % kMathGroupSize;
                        if (thread_idx == 0) {
                            storage.pipe_count[math_group_idx] = pipe_state.count();
                        }
                        math_barrier_sync(1);
                        continue;
                    }
                }

                uint32_t pred_V{};
                Fetch_V(pred_V,
                        param_V,
                        K,
                        N,
                        sched.tile_offset().y * TILE_N,
                        math_group_idx,
                        wg_idx_n,
                        sched.group_idx_,
                        storage,
                        cta_tile_p);

                float scale_V[2];
                int   iter_V{};
                auto  Load_V = [&] {
                    scale_V[0] = storage.source.V[0][math_group_idx][iter_V];
                    if (pred_V) {
                        scale_V[1] = storage.source.V[1][math_group_idx][iter_V];
                    }
                    ++iter_V;
                };

                float     scale_U[MMA_ITER_M][2];
                const int offset_U = warp_id % 4 * 16 + lane_id / 4;
                auto      Load_U   = [&] {
                    for (int m = 0; m < MMA_ITER_M; ++m) {
                        scale_U[m][0] = smem_U[pipe_state.index()][offset_U + m * MMA_ATOM_M];
                        scale_U[m][1] = smem_U[pipe_state.index()][offset_U + m * MMA_ATOM_M + 8];
                    }
                };

                auto scale_accum = [&](int m) {  // cta_n = mma_iter_n * wg_n * mma_atom_n
                    float scales[2][2];
                    scales[0][0] = scale_U[m][0] * scale_V[0];
                    scales[1][0] = scale_U[m][1] * scale_V[0];
                    scales[0][1] = scale_U[m][0] * scale_V[1];
                    scales[1][1] = scale_U[m][1] * scale_V[1];
                    PRAGMA_UNROLL
                    for (int n = 0; n < MMA_ITER_N; ++n) {
                        PRAGMA_UNROLL
                        for (int c0 = 0; c0 < MMA_ATOM_N; c0 += OUTER_N) {
                            bool pred = (pred_V & (1U << (c0 / OUTER_N)));
                            PRAGMA_UNROLL
                            for (int cc = 0; cc < OUTER_N; cc += 8) {
                                int c = c0 + cc;
                                // clang-format off
                                accum_C[m][n][c / 2 + 0] += (pred ? scales[0][1] : scales[0][0]) * frag_C[n][c / 2 + 0]; 
                                accum_C[m][n][c / 2 + 1] += (pred ? scales[0][1] : scales[0][0]) * frag_C[n][c / 2 + 1]; 
                                accum_C[m][n][c / 2 + 2] += (pred ? scales[1][1] : scales[1][0]) * frag_C[n][c / 2 + 2]; 
                                accum_C[m][n][c / 2 + 3] += (pred ? scales[1][1] : scales[1][0]) * frag_C[n][c / 2 + 3];
                                // clang-format on
                            }
                        }
                    }

                };

                auto gmma = [&](int m) {
                    PRAGMA_UNROLL
                    for (int k = 0; k < MMA_ITER_K; ++k) {
                        PRAGMA_UNROLL
                        for (int n = 0; n < MMA_ITER_N; ++n) {
                            wgmma<MMA_Atom>(smem_iter_A, smem_iter_B, frag_C[n], k == 0);
                            smem_iter_B += kStepNB;
                        }
                        smem_iter_B -= MMA_ITER_N * kStepNB;
                        smem_iter_A += kStepKA;
                        smem_iter_B += kStepKB;
                    }
                    smem_iter_A -= MMA_ITER_K * kStepKA;
                    smem_iter_B -= MMA_ITER_K * kStepKB;
                    smem_iter_A += kStepMA;
                    cute::warpgroup_commit_batch();
                };

                // static_assert(MMA_ITER_N == 1);

                __pipeline_wait_prior(0);

                math_barrier_sync(0);

                pipe_state.advance(storage.pipe_count[math_group_idx ^ 1]);

                smem_iter_A.Reset(pipe_state.index());
                smem_iter_B.Reset(pipe_state.index());
                Load_V();
                ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                Load_U();

                cute::warpgroup_arrive();
                gmma(0);
                cute::warpgroup_wait<0>();
                scale_accum(0);

                cute::warpgroup_arrive();
                gmma(1);
                cute::warpgroup_wait<0>();
                scale_accum(1);

                consumer_arrive();
                ++pipe_state;
                --k_iter;

                Load_V();
                ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                Load_U();
                smem_iter_A.Reset(pipe_state.index());
                smem_iter_B.Reset(pipe_state.index());

                // if (thread_idx == 0) {
                //     printf("k_iter %d\n", k_iter);
                // }
                for (; k_iter > 1; --k_iter) {
                    cute::warpgroup_arrive();
                    gmma(0);
                    cute::warpgroup_wait<0>();
                    scale_accum(0);

                    cute::warpgroup_arrive();
                    gmma(1);
                    cute::warpgroup_wait<0>();
                    scale_accum(1);

                    consumer_arrive();
                    ++pipe_state;
                    Load_V();
                    ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                    Load_U();
                    smem_iter_A.Reset(pipe_state.index());
                    smem_iter_B.Reset(pipe_state.index());
                }

                cute::warpgroup_arrive();
                gmma(0);
                cute::warpgroup_wait<0>();
                scale_accum(0);

                cute::warpgroup_arrive();
                gmma(1);
                cute::warpgroup_wait<0>();

                consumer_arrive();
                ++pipe_state;

                // const int thread_idx = threadIdx.x % kMathGroupSize;
                if (thread_idx == 0) {
                    storage.pipe_count[math_group_idx] = pipe_state.count();
                }

                math_barrier_sync(1);

                scale_accum(1);

                // epilogue
                PRAGMA_UNROLL
                for (int m = 0; m < MMA_ITER_M; ++m) {
                    PRAGMA_UNROLL
                    for (int n = 0; n < MMA_ITER_N; ++n) {

                        constexpr int N       = LayoutC::C0;
                        constexpr int SW_bits = log2(kSwizzleC / 16);

                        static_assert(!SW_bits || MMA_ATOM_N % LayoutC::C0 == 0);

                        const int m0 = m * MMA_ATOM_M + wg_idx_m * WG_TILE_M;
                        const int n0 = n * MMA_ATOM_N + wg_idx_n * WG_TILE_N;

                        PRAGMA_UNROLL
                        for (int i = 0; i < MMA_ATOM_N; i += 16) {
                            __align__(16) Array<Tc, 8> tvec = cast<Tc>(*(Array<float, 8>*)&accum_C[m][n][i / 2]);

                            int mm = m0 + warp_id % 4 * 16 + (lane_id & 8);
                            int nn = n0 + i / N * N;

                            int addr = ((nn / N) * TILE_M * N) + (mm * N) + (nn % N);

                            int s = lane_id % 8;
                            int c = (lane_id & 16) / 2 + i % N;

                            addr += Swizzle<SW_bits, 3, 3>::apply(s * N + c);

                            auto& uvec = (Array<uint32_t, 4>&)tvec;
                            cute::SM90_U32x4_STSM_N::copy(
                                uvec[0], uvec[1], uvec[2], uvec[3], (cutlass::uint128_t&)storage.C[addr]);
                        }
                    }
                }

                cute::tma_store_fence();  // visibility: smem -> async proxy

                math_group_barrier.sync();

                if (thread_idx < LayoutC::C1) {
                    const int tma_n = thread_idx * LayoutC::C0;
                    cute::SM90_TMA_STORE::copy(
                        &tm_c, &storage.C[thread_idx * TILE_M * LayoutC::C0], offset_n + tma_n, offset_m);
                    cute::tma_store_arrive();
                    cute::tma_store_wait<0>();
                }

                math_group_barrier.sync();

            }  // scheduler loop

            // if (thread_idx == 0) {
            //     printf("thread %d, wg %d, TAIL\n", (int)threadIdx.x, wg_idx);
            // }

            if (math_group_idx == 0) {
                math_barrier_sync(0, 0);
                while (math_barrier_sync(1, 0)) {
                    math_barrier_sync(0, 0);
                }
            }
            else {
                while (math_barrier_sync(0, 0)) {
                    math_barrier_sync(1, 0);
                }
            }

            // if (thread_idx == 0) {
            //     printf("thread %d, wg %d, EXIT\n", (int)threadIdx.x, wg_idx);
            // }

            if (threadIdx.x % kMathGroupSize < LayoutC::C1) {
                cute::tma_store_wait<0>();
            }
        }

        if constexpr (kClusterSize > 1) {
            cute::cluster_arrive();
            cute::cluster_wait();
        }

    }  // operator()

    __device__ void Fetch_V(uint32_t&          pred_V,
                            const MatrixParam& param_V,
                            int                K,
                            int                N,
                            int                offset_n,
                            int                math_group_idx,
                            int                wg_idx_n,
                            int                group_idx,
                            SharedStorage&     storage,
                            bool               active)
    {
        const int wg_offset_k = 0;
        const int wg_offset_n = offset_n + wg_idx_n * WG_TILE_N;

        auto Copy = [k = cdiv(K, 128)](Tv* dst, const Tv* src, bool pred) {
            const int tid = threadIdx.x % kMathGroupSize;
            // PRAGMA_UNROLL
            // for (int i = 0; i < MAX_K_BLOCKS; i += kMathGroupSize) {
            //     if (int p = tid + i; p < k && pred) {
            //         dst[p] = __ldg(&src[p]);
            //     }
            // }
            PRAGMA_UNROLL
            for (int i = 0; i < MAX_K_BLOCKS; i += WARPGROUP_SIZE) {
                int p = tid + i;
                CP_ASYNC<CacheOp::kAlways, 4, 0>::apply(cast_smem_ptr_to_uint(&dst[p]), &src[p], p < k && pred);
            }
        };

        const Tv* gmem_V{};
        if (active) {
            gmem_V = is_grouped_gemm ? ((Tv**)param_V.ptr)[group_idx] : (const Tv*)param_V.ptr;
            gmem_V += (wg_offset_n / 128) * param_V.stride + (wg_offset_k / 128);
        }

        Copy(storage.source.V[0][math_group_idx], gmem_V, active);

        pred_V = 0;

        // if constexpr (OUTER_N != 128) {
        if constexpr (128 % OUTER_N != 0) {

            static_assert(MMA_ATOM_N <= 128 + OUTER_N, "MMA inst is crossing more than 2 scale blocks");

            constexpr uint32_t mask = (1UL << (WG_TILE_N / OUTER_N)) - 1;

            int phase = 128 - wg_offset_n % 128;
            pred_V    = (mask << (phase / OUTER_N)) & mask;

            bool pred = active && pred_V && wg_offset_n / 128 + 1 < cdiv(N, 128);
            Copy(storage.source.V[1][math_group_idx], gmem_V + param_V.stride, pred);

            // if constexpr (WG_N > 1) {
            //     constexpr int tiles = MMA_ATOM_N / OUTER_N;
            //     pred_V              = (pred_V >> (wg_idx_n * tiles)) & ((1 << tiles) - 1);
            // }
        }

        __pipeline_commit();
    }
};

}  // namespace turbomind::gemm