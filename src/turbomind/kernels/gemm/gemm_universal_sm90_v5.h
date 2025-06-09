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

template<Order raster_order, bool is_grouped_gemm_>
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

    static constexpr int WG_M = 2;
    static constexpr int WG_N = 1;

    static constexpr int WG_TILE_M = TILE_M / WG_M;
    static constexpr int WG_TILE_N = TILE_N / WG_N;

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

    static constexpr auto is_grouped_gemm = is_grouped_gemm_;

    using Scheduler = TileScheduler<raster_order, Cluster, true, true, TILE_M, TILE_N, is_grouped_gemm>;

    static constexpr int kMulticastU = is_grouped_gemm ? 1 : kMulticastA;

    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBar = cutlass::arch::ClusterBarrier;

    static constexpr int MAX_K        = 32768;
    static constexpr int MAX_K_BLOCKS = cdiv(MAX_K, 128);

    static constexpr int kAlignmentU = 16 / sizeof(Tu);
    static constexpr int kBoxU       = TILE_M + (is_grouped_gemm ? kAlignmentU : 0);

    // Alignment requirement for SMEM addr. This forbids multicast factor 8.
    static_assert(kMulticastU == 1 || sizeof(Tu) * kBoxU / kMulticastU % 128 == 0);

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
        __align__(128) CUtensorMap tma_desc_buf[5];  //
        int                         pipe_count[2];
        typename Scheduler::Storage sched;
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
            sched.init_dyanmic(storage.sched, kClusterSize * (kMathGroupSize / WARP_SIZE + 1));
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

            cutlass::arch::NamedBarrier producers_bar(WARP_SIZE * 2, 5);

            const int  warp_id = cutlass::canonical_warp_idx_sync();
            const bool cta_0   = cute::block_id_in_cluster().x == 0;

            if (warp_id % 4 == 0) {

                Cluster cluster(cute::block_id_in_cluster().x);

                const int mc_offset_m = cluster.cta_n() * (TILE_M / kMulticastA);
                const int mc_offset_n = cluster.cta_m() * (TILE_N / kMulticastB);

                auto  smem_A = storage.source.A.data() + mc_offset_m * TILE_K;
                auto  smem_B = storage.source.B.data() + mc_offset_n * TILE_K;
                auto& smem_U = storage.source.U;

                if constexpr (is_grouped_gemm) {
                    init_tma_descs<3>({&tm_a, &tm_b, &tm_u}, storage.tma_desc_buf);
                }

                cutlass::PipelineState<Stages> write_state{0, 1, 0};

                auto sched_state = sched.init_consumer(storage.sched);

                typename Scheduler::Tile* tile;

                while (sched_state.acquire(tile)) {

                    // if (cute::elect_one_sync()) {
                    //     printf("READ m %d n %d g %d v %s\n",
                    //            tile->offset_m,
                    //            tile->offset_n,
                    //            tile->group_idx,
                    //            tile->is_valid_cluster ? "true" : "false");
                    // }

                    if (tile->is_valid_cluster) {

                        const CUtensorMap* Adesc = &tm_a;
                        const CUtensorMap* Bdesc = &tm_b;
                        const CUtensorMap* Udesc = &tm_u;

                        if constexpr (is_grouped_gemm) {
                            const int g = tile->group_idx;

                            const int m  = tile->m;
                            const int m0 = tile->m0;
                            const int m1 = m0 + m;

                            Array<void*, 3> global_addrs;
                            global_addrs[0] = (Ta*)param_A.ptr + m0 * (int64_t)param_A.stride;
                            global_addrs[1] = ((void**)param_B.ptr)[g];

                            const int beg_u = m0 / kAlignmentU * kAlignmentU;
                            const int end_u = round_up(m1, kAlignmentU);
                            global_addrs[2] = (Tu*)param_U.ptr + beg_u;

                            Array<int, 3> dims;
                            dims[0] = m;
                            dims[1] = sched.gemm_shape().y;
                            dims[2] = end_u - beg_u;

                            auto descs = update_tma_descs(tensormap_buf, storage.tma_desc_buf, global_addrs, dims);
                            Adesc      = &descs[0];
                            Bdesc      = &descs[1];
                            Udesc      = &descs[2];

                            PRAGMA_UNROLL
                            for (int i = 0; i < 3; ++i) {
                                cute::tma_descriptor_fence_acquire((cute::TmaDescriptor*)&descs[i]);
                            }
                        }

                        if (cute::elect_one_sync()) {

                            const int offset_k = 0;

                            const uint16_t mask_A = cluster.mask_m();
                            const uint16_t mask_B = cluster.mask_n();

                            const int offset_m = tile->offset_m;
                            const int offset_n = tile->offset_n;

                            int k_iter = sched.k_iters_;

                            GmemIteratorSm90<kMulticastA> gmem_A{
                                Adesc, {offset_k, offset_m + mc_offset_m}, {TILE_K, 0}};
                            GmemIteratorSm90<kMulticastB> gmem_B{
                                Bdesc, {offset_k, offset_n + mc_offset_n}, {TILE_K, 0}};

                            const int mc_offset_u = kMulticastU > 1 ? mc_offset_m : 0;
                            // column-major
                            GmemIteratorSm90<kMulticastU> gmem_U{
                                Udesc, {offset_m + mc_offset_u, offset_k / 128}, {0, 1}};

                            for (; k_iter > 0; --k_iter) {
                                int pipe = write_state.index();
                                ConsumerBar::wait(&consumer_bar[pipe], write_state.phase());
                                ProducerBar::arrive_and_expect_tx(&producer_bar[pipe], kTmaTxBytes);
                                gmem_A.Step(&producer_bar[pipe], &smem_A[pipe * TILE_M * TILE_K], mask_A);
                                gmem_B.Step(&producer_bar[pipe], &smem_B[pipe * TILE_N * TILE_K], mask_B);
                                gmem_U.Step(&producer_bar[pipe], &smem_U[pipe][0] + mc_offset_u, mask_A);
                                ++write_state;
                            }
                        }
                    }

                    if constexpr (Scheduler::is_dynamic) {
                        if (cta_0) {
                            producers_bar.arrive_unaligned();
                        }
                    }

                    sched_state.release();
                }
                sched_state.acquire(tile);
                sched_state.release();
            }
            else if (warp_id % 4 == 1 && cta_0) {
                auto sched_state = sched.init_producer(storage.sched);
                while (sched_state.next()) {
                    if constexpr (Scheduler::is_dynamic) {
                        producers_bar.arrive_and_wait_unaligned();
                    }
                }
                // send extra null tile
                sched_state.next();
            }
        }
        else {
            cutlass::arch::warpgroup_reg_alloc<112>();

            const int math_group_idx = wg_idx / 2;

            if constexpr (is_grouped_gemm) {
                if (threadIdx.x % kMathGroupSize / WARP_SIZE == 0) {
                    init_tma_descs<1>({&tm_c}, storage.tma_desc_buf + 3 + math_group_idx);
                }
            }

            auto& smem_A = storage.source.A;
            auto& smem_B = storage.source.B;
            auto& smem_U = storage.source.U;

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

                constexpr int threads = WARPGORUPS * WARPGROUP_SIZE;
                int           res     = 0;

                // if (thread_idx == 0) {
                //     printf("math_barrier_sync %4d, %4d, %4d\n", math_group_idx, math_group_idx ^ phase, alive);
                // }

                asm volatile("{\n"
                             "  .reg.pred p;\n"
                             "  setp.ne.b32 p, %3, 0;\n"
                             "  barrier.cta.red.or.pred p, %1, %2, p;\n"
                             "  selp.s32 %0, 1, 0, p;\n"
                             "}\n"
                             : "=r"(res)
                             : "r"(barrier_id), "r"(threads), "r"(alive));

                // if (thread_idx == 0) {
                //     printf("math_barrier_sync %4d, %4d, %4d DONE\n", math_group_idx, math_group_idx ^ phase, alive);
                // }

                return res;
            };

            cutlass::arch::NamedBarrier math_group_barrier(kMathGroupSize, 2 + math_group_idx);  // 2,3

            cutlass::PipelineState<Stages> pipe_state{};

            const int warp_id = cutlass::canonical_warp_idx_sync();
            const int lane_id = cutlass::canonical_lane_idx();

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

            auto sched_state = sched.init_consumer(storage.sched);

            if (math_group_idx == 1) {
                ++sched_state.pipe;
                math_barrier_sync(1);
            }

            typename Scheduler::Tile* tile;

            sched_state.acquire(tile);

            while (tile->alive) {

                // if (thread_idx == 0) {
                //     printf("TILE %d %d %d %d %s\n",
                //            //    sched.cluster_idx_,
                //            math_group_idx,
                //            tile->offset_m,
                //            tile->offset_n,
                //            tile->group_idx,
                //            tile->is_valid_cluster ? "true" : "false");
                // }

                if (tile->is_valid_cta) {

                    // const auto tile_offset = sched.tile_offset();

                    MMA_Atom::CRegisters frag_C;
                    MMA_Atom::CRegisters accum_C[MMA_ITER_M][MMA_ITER_N]{};

                    // const auto [iter_k_beg, iter_k_end] = sched.iter_k_range();

                    const auto [_, N, K, L] = sched.gemm_shape();

                    const int offset_m = tile->offset_m;
                    const int offset_n = tile->offset_n;

                    int k_iter = sched.k_iters_;

                    if constexpr (is_grouped_gemm) {
                        if (thread_idx / WARP_SIZE == 0) {
                            auto addr = (Tc*)param_C.ptr + tile->m0 * (int64_t)param_C.stride;
                            int  idx  = 3 + math_group_idx;
                            update_tma_descs<1>(tensormap_buf + idx, storage.tma_desc_buf + idx, {addr}, {tile->m});
                        }
                    }

                    uint32_t pred_V{};
                    Fetch_V(pred_V,
                            param_V,
                            K,
                            N,
                            tile->offset_n,
                            math_group_idx,
                            wg_idx_n,
                            tile->group_idx,
                            storage,
                            tile->is_valid_cta);

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
                    const int offset_U = wg_idx_m * WG_TILE_M + warp_id % 4 * 16 + lane_id / 4;
                    int       align_U  = 0;
                    if constexpr (is_grouped_gemm) {
                        align_U = tile->m0 % kAlignmentU;
                    }
                    auto Load_U = [&] {
                        for (int m = 0; m < MMA_ITER_M; ++m) {
                            scale_U[m][0] = smem_U[pipe_state.index()][align_U + offset_U + m * MMA_ATOM_M];
                            scale_U[m][1] = smem_U[pipe_state.index()][align_U + offset_U + m * MMA_ATOM_M + 8];
                        }
                    };

                    float scales[2][2];
                    auto  scale_accum = [&](int n) {  // cta_n = mma_iter_n * wg_n * mma_atom_n
                        int m        = 0;
                        scales[0][0] = scale_U[m][0] * scale_V[0];
                        scales[1][0] = scale_U[m][1] * scale_V[0];
                        scales[0][1] = scale_U[m][0] * scale_V[1];
                        scales[1][1] = scale_U[m][1] * scale_V[1];
                        PRAGMA_UNROLL
                        for (int c0 = 0; c0 < MMA_ATOM_N; c0 += OUTER_N) {
                            bool pred = (pred_V & (1U << (c0 / OUTER_N)));
                            PRAGMA_UNROLL
                            for (int cc = 0; cc < OUTER_N; cc += 8) {
                                int c = c0 + cc;
                                // clang-format off
                                accum_C[m][n][c / 2 + 0] += (pred ? scales[0][1] : scales[0][0]) * frag_C[c / 2 + 0]; 
                                accum_C[m][n][c / 2 + 1] += (pred ? scales[0][1] : scales[0][0]) * frag_C[c / 2 + 1]; 
                                accum_C[m][n][c / 2 + 2] += (pred ? scales[1][1] : scales[1][0]) * frag_C[c / 2 + 2]; 
                                accum_C[m][n][c / 2 + 3] += (pred ? scales[1][1] : scales[1][0]) * frag_C[c / 2 + 3];
                                // clang-format on
                            }
                        }
                    };

                    auto gmma = [&](int n) {
                        PRAGMA_UNROLL
                        for (int k = 0; k < MMA_ITER_K; ++k) {
                            wgmma<MMA_Atom>(smem_iter_A, smem_iter_B, frag_C, k == 0);
                            smem_iter_A += kStepKA;
                            smem_iter_B += kStepKB;
                        }
                        smem_iter_A -= MMA_ITER_K * kStepKA;
                        smem_iter_B -= MMA_ITER_K * kStepKB;
                        smem_iter_B += kStepNB;
                        if (n == MMA_ITER_N - 1) {
                            smem_iter_B -= MMA_ITER_N * kStepNB;
                        }
                        cute::warpgroup_commit_batch();
                    };

                    static_assert(MMA_ITER_M == 1);

                    // __pipeline_wait_prior(0);

                    math_barrier_sync(0);

                    pipe_state = {};
                    pipe_state.advance(storage.pipe_count[math_group_idx ^ 1]);

                    // if (thread_idx == 0) {
                    //     printf("ACQUIRE %4d, %4d\n", math_group_idx, (int)pipe_state.count());
                    // }

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

                    if (thread_idx == 0) {
                        storage.pipe_count[math_group_idx] = pipe_state.count();
                        // printf("RELEASE %4d, %4d\n", math_group_idx, (int)pipe_state.count());
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
                                // fill(tvec, Tc(255));
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
                        const void* Cdesc = &tm_c;
                        const int   tma_n = thread_idx * LayoutC::C0;
                        if constexpr (is_grouped_gemm) {
                            Cdesc = &tensormap_buf[3 + math_group_idx + blockIdx.x * 5];
                            cute::tma_descriptor_fence_acquire((cute::TmaDescriptor*)Cdesc);
                        }
                        cute::SM90_TMA_STORE::copy(
                            Cdesc, &storage.C[thread_idx * TILE_M * LayoutC::C0], offset_n + tma_n, offset_m);
                        cute::tma_store_arrive();
                        cute::tma_store_wait<0>();
                    }

                    // math_group_barrier.sync();

                }  // valid cta tile
                else {
                    math_barrier_sync(0);

                    pipe_state = {};
                    pipe_state.advance(storage.pipe_count[math_group_idx ^ 1]);

                    if (tile->is_valid_cluster) {
                        // other CTAs in the cluster are still alive
                        for (int k_iter = sched.k_iters_; k_iter > 0; --k_iter) {
                            ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                            consumer_arrive();
                            ++pipe_state;
                        }
                    }

                    if (thread_idx == 0) {
                        storage.pipe_count[math_group_idx] = pipe_state.count();
                    }

                    math_barrier_sync(1);
                }

                sched_state.release(2);
                sched_state.acquire(tile);

                // if (thread_idx == 0) {
                //     printf("pipe = %d\n", sched_state.pipe.index());
                // }
            }  // scheduler loop

            // if (thread_idx == 0) {
            //     printf("TAIL %d\n", math_group_idx);
            // }

            if (math_group_idx == 0) {
                math_barrier_sync(0, 0);
                if (thread_idx == 0) {
                    storage.pipe_count[0] = storage.pipe_count[1];
                }
                while (math_barrier_sync(1, 0)) {
                    math_barrier_sync(0, 0);
                    if (thread_idx == 0) {
                        storage.pipe_count[0] = storage.pipe_count[1];
                    }
                }
            }
            else {
                while (math_barrier_sync(0, 0)) {
                    if (thread_idx == 0) {
                        storage.pipe_count[1] = storage.pipe_count[0];
                    }
                    math_barrier_sync(1, 0);
                }
            }

            // if (thread_idx == 0) {
            //     printf("EXIT %d\n", math_group_idx);
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

    template<int N>
    __device__ void init_tma_descs(Array<const CUtensorMap*, N> param_desc, CUtensorMap* smem_desc)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;

        if (lane_id < sizeof(CUtensorMap) / sizeof(uint2)) {
            PRAGMA_UNROLL
            for (int i = 0; i < N; ++i) {
                ((uint2*)&smem_desc[i])[lane_id] = ((uint2*)param_desc[i])[lane_id];
            }
        }

        __syncwarp();
    }

    template<int N>
    __device__ CUtensorMap*
    update_tma_descs(CUtensorMap* gmem_desc, CUtensorMap* smem_desc, Array<void*, N> global_addrs, Array<int, N> dims)
    {
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            PRAGMA_UNROLL
            for (int i = 0; i < N; ++i) {
                uint32_t uint_ptr = cast_smem_ptr_to_uint(&smem_desc[i]);
                // clang-format off
                asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" ::"r"(uint_ptr), "l"(global_addrs[i]));
                if (i != 2) {
                    asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;" ::"r"(uint_ptr), "r"(dims[i]));
                } else { // special case for U
                    asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" ::"r"(uint_ptr), "r"(dims[i]));
                }
                // clang-format on
            }
        }

        __syncwarp();

        constexpr int kNumPerCta = 5;  // a,b,u,c0,c1
        auto          gmem_ptr   = &gmem_desc[blockIdx.x * kNumPerCta];
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            uint32_t uint_ptr = cast_smem_ptr_to_uint(&smem_desc[i]);
            // clang-format off
            asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;" :: "l"(gmem_ptr + i), "r"(uint_ptr));
            // clang-format on
        }

        return gmem_ptr;
    }

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
            PRAGMA_UNROLL
            for (int i = 0; i < MAX_K_BLOCKS; i += kMathGroupSize) {
                if (int p = tid + i; p < k && pred) {
                    dst[p] = __ldg(&src[p]);
                }
            }
            // PRAGMA_UNROLL
            // for (int i = 0; i < MAX_K_BLOCKS; i += WARPGROUP_SIZE) {
            //     int p = tid + i;
            //     CP_ASYNC<CacheOp::kAlways, 4, 0>::apply(cast_smem_ptr_to_uint(&dst[p]), &src[p], p < k && pred);
            // }
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

        // __pipeline_commit();
    }
};

}  // namespace turbomind::gemm