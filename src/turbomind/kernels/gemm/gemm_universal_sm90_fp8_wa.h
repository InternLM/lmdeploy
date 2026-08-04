#pragma once

#include <numeric>
#include <type_traits>
#include <utility>

#include <cuda_fp8.h>
#include <cuda_pipeline_primitives.h>

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/arch/mma_sm90_desc.hpp"
#include "cute/tensor.hpp"

#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/smem.h"

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/cp_async.h"
#include "src/turbomind/kernels/gemm/iterator_sm90.h"
#include "src/turbomind/kernels/gemm/matrix_ptr.h"
#include "src/turbomind/kernels/gemm/scheduler.cuh"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

/*
 * SM90 blockscaled FP8 GEMM — weight-as-A (WA) GMMA binding.
 *
 *   LlamaLinear API: A=act (M,K), B=weight (K,N), U/V scales
 *   GMMA: A_smem=weight (OUT×K), B_smem=act (BATCH×K)
 *   TileShape (OUT,BATCH,K)=(TILE_N,TILE_M,TILE_K)
 *   Scales: V sparse on GMMA-M (OUT); U dense on GMMA-N (BATCH)
 *
 * Host launch reuses KernelImplSm90 (API TMA boxes unchanged).
 */
#include "src/turbomind/kernels/gemm/gmma_bf16_sm90.h"
#include "src/turbomind/kernels/gemm/gmma_fp8_sm90.h"
#include "src/turbomind/kernels/gemm/prepare_moe_tma_descs_sm90_fp8.h"
#include "src/turbomind/kernels/gemm/sm90_fp8_wa_traits.h"
#include "src/turbomind/kernels/gemm/sm90_utils.h"

namespace turbomind::gemm {

template<Order    raster_order,
         int      multicast_a,
         int      multicast_b,
         bool     is_grouped_gemm_,
         Striding kStridingA_     = (is_grouped_gemm_ ? Striding::kIndexed : Striding::kFlat),
         class Tile_              = Sm90Fp8WaTile_64x128,
         bool kSupportsFusedSilu_ = false>
struct GemmUniversalSm90_Fp8Wa {

    static constexpr bool kDebug = false;

    using Arch = Sm90;
    using Tile = Tile_;

    static constexpr bool kSupportsFusedSilu = kSupportsFusedSilu_;

    static constexpr int TILE_M = Tile::TILE_M;
    static constexpr int TILE_N = Tile::TILE_N;
    static constexpr int TILE_K = Tile::TILE_K;

    static constexpr int WG_M = Tile::WG_M;
    static constexpr int WG_N = Tile::WG_N;

    static constexpr int WG_TILE_M = TILE_M / WG_M;  // BATCH per WG
    static constexpr int WG_TILE_N = TILE_N / WG_N;  // OUT per WG
    static_assert(TILE_M % WG_M == 0);
    static_assert(TILE_N % WG_N == 0);
    static_assert(WG_TILE_N % 64 == 0);  // WGMMA atom M along OUT

    static constexpr int kSchedWarpGroups = 1;

    static constexpr int WARPGROUPS = WG_M * WG_N;

    static constexpr Order kRasterOrder = raster_order;
    static constexpr int   kAlgoFamily  = 2;

    // GMMA-M=OUT=WG_TILE_N, GMMA-N=BATCH=WG_TILE_M
    using GMMA   = ScaledGmmaFP8_WA<WG_TILE_N, WG_TILE_M, TILE_K, 1, 1, 1, 1, Tile::kMaxOpN>;
    using AccumC = typename GMMA::AccumC;
    using FragC  = typename GMMA::FragC;

    // Fused SiLU: [g128|u128] along OUT. WG_N == 1 pairs GMMA-M atoms i with i+2 in
    // register (ITER_M=4); WG_N == 2 stages gate/up through smem (ITER_M=2 per WG).
    static_assert(!kSupportsFusedSilu || (TILE_N == 256 && GMMA::OP_M == 64 && GMMA::ITER_M == 4 / WG_N));

    static constexpr int kMulticastA = multicast_a;
    static constexpr int kMulticastB = multicast_b;

    static constexpr int kClusterSize = kMulticastA * kMulticastB;

    static constexpr int Stages = Tile::Stages;

    static constexpr bool kSplitK     = false;
    static constexpr int  kChunkSizeK = TILE_K;

    static constexpr int WARPGROUP_SIZE = 128;

    static constexpr int kMathGroupSize = WARPGROUP_SIZE * WARPGROUPS;

    static constexpr int CTA_SIZE = WARPGROUP_SIZE * (WARPGROUPS + 1);

    using Ta = __nv_fp8_e4m3;
    using Tb = __nv_fp8_e4m3;
    using Tc = nv_bfloat16;

    using Tu = float;
    using Tv = float;
    using Tw = float;  // dynamic output group scales (fused path)

    using Cluster = arch::Cluster<kMulticastB, kMulticastA, kRowMajor>;

    static constexpr auto is_grouped_gemm = is_grouped_gemm_;

    static constexpr Striding kStridingA = kStridingA_;
    static constexpr Striding kStridingB = is_grouped_gemm_ ? Striding::kBlocked : Striding::kFlat;
    static constexpr Striding kStridingC = is_grouped_gemm_ ? Striding::kBlocked : Striding::kFlat;

    // Indexed gather: A/U via cp.async; TMA only B (+ C store descs).
    static constexpr bool kIndexedGather = (kStridingA_ == Striding::kIndexed);

    using Scheduler = TileScheduler<raster_order, Cluster, true, true, TILE_M, TILE_N, Stages, is_grouped_gemm>;

    static constexpr int kMulticastU = is_grouped_gemm ? 1 : kMulticastA;

    using ProducerBar = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBar = cutlass::arch::ClusterBarrier;

    static constexpr int kAlignmentU = 16 / sizeof(Tu);
    static constexpr int kBoxU       = TILE_M + (is_grouped_gemm ? kAlignmentU : 0);

    // Alignment requirement for SMEM addr. This forbids multicast factor 8.
    static_assert(kMulticastU == 1 || sizeof(Tu) * kBoxU / kMulticastU % 128 == 0);

    static constexpr int kTmaTxBytesWeight = (int)sizeof(Tb) * (TILE_N * TILE_K);
    static constexpr int kTmaTxBytesAct    = (int)sizeof(Ta) * (TILE_M * TILE_K);
    static constexpr int kTmaTxBytesU      = (int)sizeof(Tu) * kBoxU;
    // Dense / blocked-A: expect_tx(A+B+U). Indexed-A: expect_tx(B only); A/U via cp.async noinc.
    static constexpr int kTmaTxBytes = kTmaTxBytesWeight + (kIndexedGather ? 0 : (kTmaTxBytesAct + kTmaTxBytesU));

    // Dense: unused. Grouped indexed: [B, C]. Grouped blocked: [A, B, U, C].
    static constexpr int kTmaDescNum = !is_grouped_gemm_ ? 1 : (kIndexedGather ? 2 : 4);
    static constexpr int kCdescIdx   = kIndexedGather ? 1 : 3;

    // Indexed gather stores act into GMMA-B (BATCH×K).
    using SmemLayoutB_2D = decltype(cute::tile_to_shape(cute::SM90::GMMA::Layout_K_SW128_Atom<cutlass::float_e4m3_t>{},
                                                        cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_K>{}),
                                                        cute::Step<cute::_1, cute::_2>{}));

    // setmaxnreg: each WG ≤ 256, multiples of 8. Budgets come from Tile
    // (TMA vs indexed). 2 math WGs pack to 504; 1 math WG packs to ≤512.
    static constexpr int kProducerRegs = kIndexedGather ? Tile::kProducerRegsIndexed : Tile::kProducerRegsTma;
    static constexpr int kMathRegs     = kIndexedGather ? Tile::kMathRegsIndexed : Tile::kMathRegsTma;
    static_assert(kProducerRegs >= 24 && kProducerRegs % 8 == 0);
    static_assert(kMathRegs >= 24 && kMathRegs % 8 == 0 && kMathRegs <= 256);
    static_assert(WARPGROUPS == 1 || WARPGROUPS == 2);
    static_assert(WARPGROUPS != 2 || kProducerRegs + 2 * kMathRegs == 504);
    static_assert(WARPGROUPS != 1 || kProducerRegs + kMathRegs <= 512);

    // ! SMEM addr must be SBO aligned for TMA load/store
    struct SharedStorage {
        // GMMA-A = weight (OUT×K); GMMA-B = act (BATCH×K)
        __align__(1024) Array<Tb, Stages * TILE_N * TILE_K> A;
        __align__(1024) Array<Ta, Stages * TILE_M * TILE_K> B;
        // Fused amax reduce: per math WG, [warp][t0][scale_i], scale_i < OP_N/4 ≤ 64
        __align__(128) float fused_amax_scratch[WARPGROUPS][4 * 4 * 64];
        // f32 gate/up staging for WG_N == 2 fused SiLU (gate/up split across math WGs).
        static constexpr int kSiluStageElems = (kSupportsFusedSilu && WG_N == 2) ? TILE_N * TILE_M : 1;
        __align__(128) float silu_stage[kSiluStageElems];
        __align__(128) Tu U[Stages][round_up<int>(kBoxU, 128)];  // at least 128 byte alignment
        __align__(128) Tv V[Stages][2];
        __align__(8) uint64_t producer_bar[Stages];
        __align__(8) uint64_t consumer_bar[Stages];
        typename Scheduler::Storage sched;
        int                         gather_alive;
        int                         gather_k_iters;
        int                         gather_m0;
        int                         gather_M_group;
        int                         gather_offset_m;
    };

    template<bool kFuseSilu>
    struct Output {
        static_assert(!kFuseSilu || kSupportsFusedSilu);

        using Tc = std::conditional_t<kFuseSilu, __nv_fp8_e4m3, nv_bfloat16>;

        static constexpr int kStoreOut = kFuseSilu ? TILE_N / 2 : TILE_N;

        struct LayoutC {
            // One box covering the whole CTA tile (all math WGs share smem C).
            static constexpr int S0       = TILE_M;
            static constexpr int C0       = kStoreOut;
            static constexpr int C1       = 1;
            static constexpr int C1_store = 1;
        };

        static constexpr int kSwizzleC = 0;
    };

    static constexpr int kOutputOffset = round_up<int>(sizeof(SharedStorage), 1024);

    static constexpr int GetSmemSize(bool fuse_silu)
    {
        return kOutputOffset
               + (fuse_silu ? TILE_M * (TILE_N / 2) * (int)sizeof(__nv_fp8_e4m3) :
                              TILE_M * TILE_N * (int)sizeof(nv_bfloat16));
    }

    static constexpr int kSmemSize = GetSmemSize(false);

    static constexpr int OUTER_M = GMMA::OUTER_M;
    // Weight-scale predicates along OUT (problem N / GMMA-M).
    static constexpr int MMA_SUBTILE_M = WG_TILE_N / OUTER_M;

    // Host: rebase per-expert TMA maps into workspace before GEMM launch.
    static int* PrepareTmaDescs(const CUtensorMap& tm_a,
                                const CUtensorMap& tm_b,
                                const CUtensorMap& tm_u,
                                const CUtensorMap& tm_c,
                                const MatrixParam& param_A,
                                const MatrixParam& param_B,
                                const MatrixParam& param_U,
                                const MatrixParam& param_C,
                                bool               fuse_silu,
                                CUtensorMap*       out,
                                int                num_groups,
                                int                M,
                                int                N,
                                cudaStream_t       stream)
    {
        if constexpr (!is_grouped_gemm_) {
            return nullptr;
        }
        int* offsets = reinterpret_cast<int*>(out + num_groups * kTmaDescNum);
        prepare_moe_tma_descs_sm90_fp8<kAlignmentU, kStridingA_><<<num_groups, 32, 0, stream>>>(
            tm_a, tm_b, tm_u, tm_c, param_A, param_B, param_U, param_C, fuse_silu, out, offsets, M, N);
        return offsets;
    }

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
                               const MatrixParam& param_W,
                               bool               fuse_silu,
                               Scheduler          sched,
                               CUtensorMap*       tensormap_buf,
                               char*              smem_buf)
    {
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        uint64_t* producer_bar = storage.producer_bar;
        uint64_t* consumer_bar = storage.consumer_bar;

        constexpr int kProducerBarInit = kIndexedGather ? (1 + WARPGROUP_SIZE) : (1 + 1);

        if (threadIdx.x == 0) {
            PRAGMA_UNROLL
            for (int s = 0; s < Stages; ++s) {
                ProducerBar::init(&producer_bar[s], kProducerBarInit);
                ConsumerBar::init(&consumer_bar[s], WARPGROUPS * kClusterSize * 4);
            }
            sched.init_dyanmic(storage.sched, kClusterSize * (WARPGROUPS * 4 + 1));
            cutlass::arch::fence_view_async_shared();
            if constexpr (kClusterSize > 1) {
                cutlass::arch::fence_barrier_init();
            }
        }

        (kClusterSize > 1) ? cute::cluster_sync() : __syncthreads();

        const int wg_idx = cutlass::canonical_warp_group_idx();

        if (wg_idx == WARPGROUPS) {
            cutlass::arch::warpgroup_reg_dealloc<kProducerRegs>();

            static_assert(TILE_M % kMulticastA == 0);
            static_assert(TILE_N % kMulticastB == 0);

            cutlass::arch::NamedBarrier producers_bar(WARP_SIZE * 2, 7);

            const int  warp_id    = cutlass::canonical_warp_idx_sync();
            const int  warp_in_wg = warp_id % 4;
            const bool cta_0      = cute::block_id_in_cluster().x == 0;

            if constexpr (kIndexedGather) {
                // Full producer WG gather. Scheduler folded onto warp0.
                cutlass::arch::NamedBarrier gather_bar(
                    /*num_threads=*/WARPGROUP_SIZE, cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);

                Cluster cluster(cute::block_id_in_cluster().x);

                const int mc_offset_n = cluster.cta_m() * (TILE_N / kMulticastB);

                // WA: weight→GMMA-A (storage.A), act→GMMA-B (storage.B)
                auto* smem_weight = storage.A.data() + mc_offset_n * TILE_K;
                auto* smem_act    = storage.B.data();
                auto& smem_U      = storage.U;
                auto& smem_V      = storage.V;

                cutlass::PipelineState<Stages> write_state{0, 1, 0};

                typename Scheduler::ConsumerState sched_state    = sched.init_consumer(storage.sched);
                typename Scheduler::ProducerState prod_state     = sched.init_producer(storage.sched);
                int                               lane_predicate = 0;
                const int                         lane_id        = threadIdx.x % WARP_SIZE;
                const int                         prod_tid       = threadIdx.x - WARPGROUPS * WARPGROUP_SIZE;

                if (warp_in_wg == 0) {
                    lane_predicate = cute::elect_one_sync();
                }

                const Ta*  act_gmem = (const Ta*)param_A.ptr;
                const int  ldA      = param_A.stride;
                const int* idxs     = param_A.idxs;
                const Tu*  u_gmem   = (const Tu*)param_U.ptr;
                const int  ldU      = param_U.stride;
                const int  K        = sched.gemm_shape().z;

                constexpr int kVec = 16;  // uint4 / sizeof(fp8)
                constexpr int nvec = TILE_M * (TILE_K / kVec);
                // TILE_M=8 → nvec=64 < WARPGROUP_SIZE: one slot, idle threads predicated off.
                constexpr int kSlots = (nvec + WARPGROUP_SIZE - 1) / WARPGROUP_SIZE;
                static_assert(kSlots >= 1);
                // U: one float per producer thread along TILE_M (thread m loads row m).
                static_assert(TILE_M <= WARPGROUP_SIZE);

                typename Scheduler::Tile* tile;

                while (true) {
                    const CUtensorMap* Bdesc   = &tm_b;
                    uint16_t           mask_B  = 0;
                    int                coord_n = 0;
                    int                k_iters = 0;
                    const Tv*          gmem_V0 = (const Tv*)param_V.ptr;
                    const Tv*          gmem_V1 = nullptr;
                    int                ldV     = param_V.stride;

                    if (warp_in_wg == 0 && cta_0) {
                        (void)prod_state.next();
                    }

                    if (warp_in_wg == 0) {
                        const bool alive = sched_state.acquire(tile);
                        int        m0 = 0, M_group = 0, offset_m = 0;

                        if (alive && tile->is_valid_cluster) {
                            if constexpr (is_grouped_gemm) {
                                const int g  = tile->group_idx;
                                Bdesc        = &tensormap_buf[g * kTmaDescNum];
                                const auto v = resolve<Tv, Striding::kBlocked>(param_V, g);
                                gmem_V0      = (const Tv*)v.ptr.ptr;
                                ldV          = v.ptr.stride;
                            }

                            mask_B   = cluster.mask_n();
                            coord_n  = tile->offset_n + mc_offset_n;
                            offset_m = tile->offset_m;
                            m0       = [&] {
                                if constexpr (is_grouped_gemm) {
                                    return tile->m0;
                                }
                                return 0;
                            }();
                            M_group = [&] {
                                if constexpr (is_grouped_gemm) {
                                    return tile->m1 - tile->m0;
                                }
                                return sched.gemm_shape().x;
                            }();
                            k_iters = sched.k_iters_;

                            gmem_V0 += (tile->offset_n / 128) * ldV;
                            gmem_V1 = gmem_V0;
                            if (tile->offset_n / 128 + 1 < cdiv(sched.gemm_shape().y, 128)) {
                                gmem_V1 += ldV;
                            }
                        }

                        if (lane_id == 0) {
                            storage.gather_alive    = alive ? 1 : 0;
                            storage.gather_k_iters  = k_iters;
                            storage.gather_m0       = m0;
                            storage.gather_M_group  = M_group;
                            storage.gather_offset_m = offset_m;
                        }
                        __syncwarp();
                    }

                    // Tile header only.
                    gather_bar.arrive_and_wait();

                    if (storage.gather_alive == 0) {
                        break;
                    }

                    k_iters            = storage.gather_k_iters;
                    const int m0       = storage.gather_m0;
                    const int M_group  = storage.gather_M_group;
                    const int offset_m = storage.gather_offset_m;
                    int       coord_k  = 0;

                    // iterator_sm80 style: idxs → src bases, then += TILE_K each K tile.
                    const Ta* src_data_vec_[kSlots];
                    int       m_own[kSlots];
                    int       kk_own[kSlots];
                    bool      pred_row[kSlots];
                    const Tu* src_u_base;
                    int       m_u;
                    bool      pred_u;
                    const int u_pad = m0 % kAlignmentU;

                    PRAGMA_UNROLL
                    for (int t = 0; t < kSlots; ++t) {
                        const int  i      = prod_tid + t * WARPGROUP_SIZE;
                        const bool in_vec = i < nvec;
                        const int  m      = in_vec ? i / (TILE_K / kVec) : 0;
                        const int  kk     = in_vec ? (i % (TILE_K / kVec)) * kVec : 0;
                        const int  packed = m0 + offset_m + m;
                        const bool row_ok = in_vec && (offset_m + m) < M_group;
                        const int  token  = (idxs && row_ok) ? __ldg(idxs + packed) : packed;
                        m_own[t]          = m;
                        kk_own[t]         = kk;
                        pred_row[t]       = row_ok;
                        src_data_vec_[t]  = act_gmem + (int64_t)token * ldA + kk;
                    }

                    {
                        // TILE_M < WARPGROUP_SIZE: idle threads must skip U ZFILL entirely
                        // (pred=false still zeros dst — see sm90_bf16).
                        const int  m      = prod_tid;
                        const bool in_m   = m < TILE_M;
                        const int  packed = m0 + offset_m + m;
                        const bool row_ok = in_m && (offset_m + m) < M_group;
                        const int  token  = (idxs && row_ok) ? __ldg(idxs + packed) : packed;
                        m_u               = m;
                        pred_u            = row_ok;
                        src_u_base        = u_gmem + token;
                    }

                    // Warp0-only weight TMA + V; full WG gathers A/U.
                    GmemIteratorSm90<kMulticastB> gmem_B{
                        (warp_in_wg == 0) ? Bdesc : &tm_b, {0, (warp_in_wg == 0) ? coord_n : 0}, {TILE_K, 0}};

                    for (; k_iters > 0; --k_iters) {
                        const int pipe = write_state.index();
                        ConsumerBar::wait(&consumer_bar[pipe], write_state.phase());

                        if (warp_in_wg == 0 && lane_predicate) {
                            ProducerBar::arrive_and_expect_tx(&producer_bar[pipe], kTmaTxBytes);
                            gmem_B.Step(&producer_bar[pipe], &smem_weight[pipe * TILE_N * TILE_K], mask_B);
                            uint32_t uint_ptr_V = cast_smem_ptr_to_uint(smem_V[pipe]);
                            CP_ASYNC<CacheOp::kAlways, 4, 0>::apply(uint_ptr_V, gmem_V0, true);
                            CP_ASYNC<CacheOp::kAlways, 4, 0>::apply(uint_ptr_V + sizeof(Tv), gmem_V1, true);
                            ++gmem_V0;
                            ++gmem_V1;
                        }

                        {
                            cute::Tensor sB = cute::make_tensor(cute::make_smem_ptr(smem_act + pipe * TILE_M * TILE_K),
                                                                SmemLayoutB_2D{});

                            PRAGMA_UNROLL
                            for (int t = 0; t < kSlots; ++t) {
                                const bool pred = pred_row[t] && (coord_k + kk_own[t]) < K;
                                auto*      dst  = &sB(m_own[t], kk_own[t]);
                                cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>::copy(
                                    *reinterpret_cast<const uint4*>(src_data_vec_[t]),
                                    *reinterpret_cast<uint4*>(dst),
                                    pred);
                                src_data_vec_[t] += TILE_K;
                            }

                            // U: one float per thread along TILE_M; ColMajor (token + k_col*ldU).
                            // Idle (m >= TILE_M): do not issue ZFILL — pred=false still zeros dst.
                            if (m_u < TILE_M) {
                                const bool pred = pred_u && (coord_k < K);
                                auto*      dst  = &smem_U[pipe][u_pad + m_u];
                                const Tu*  src  = src_u_base + (coord_k / 128) * ldU;
                                cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint32_t>::copy(
                                    *reinterpret_cast<const uint32_t*>(src), *reinterpret_cast<uint32_t*>(dst), pred);
                            }

                            cutlass::arch::cpasync_barrier_arrive_noinc(&producer_bar[pipe]);
                        }

                        ++write_state;
                        coord_k += TILE_K;
                    }

                    if (warp_in_wg == 0) {
                        sched_state.release();
                    }
                }

                if (warp_in_wg == 0) {
                    sched_state.release();
                    if (cta_0) {
                        sched.tail(prod_state);
                    }
                    if constexpr (kClusterSize > 1) {
                        if (lane_predicate) {
                            for (int i = 0; i < Stages; ++i) {
                                ConsumerBar::wait(&consumer_bar[write_state.index()], write_state.phase());
                                ++write_state;
                            }
                        }
                        __syncwarp();
                    }
                }
            }
            else if (warp_in_wg == 0) {
                Cluster cluster(cute::block_id_in_cluster().x);

                const int mc_offset_m = cluster.cta_n() * (TILE_M / kMulticastA);
                const int mc_offset_n = cluster.cta_m() * (TILE_N / kMulticastB);

                auto  smem_weight = storage.A.data() + mc_offset_n * TILE_K;
                auto  smem_act    = storage.B.data() + mc_offset_m * TILE_K;
                auto& smem_U      = storage.U;
                auto& smem_V      = storage.V;

                cutlass::PipelineState<Stages> write_state{0, 1, 0};

                auto sched_state = sched.init_consumer(storage.sched);

                int lane_predicate = cute::elect_one_sync();

                typename Scheduler::Tile* tile;

                while (sched_state.acquire(tile)) {

                    if (tile->is_valid_cluster) {

                        const CUtensorMap* Adesc = &tm_a;
                        const CUtensorMap* Bdesc = &tm_b;
                        const CUtensorMap* Udesc = &tm_u;

                        const Tv* gmem_V0 = (const Tv*)param_V.ptr;
                        const Tv* gmem_V1;
                        int       ldV = param_V.stride;

                        if constexpr (is_grouped_gemm) {
                            // Descs published by prepare_moe_tma_descs on this stream;
                            // fence_acquire only needed after in-kernel tensormap replace.
                            const int          g     = tile->group_idx;
                            CUtensorMap* const descs = tensormap_buf + g * kTmaDescNum;
                            if constexpr (kStridingA == Striding::kBlocked) {
                                Adesc = &descs[0];
                                Bdesc = &descs[1];
                                Udesc = &descs[2];
                            }
                            else {
                                Bdesc = &descs[0];
                            }
                            const auto v = resolve<Tv, Striding::kBlocked>(param_V, g);
                            gmem_V0      = (const Tv*)v.ptr.ptr;
                            ldV          = v.ptr.stride;
                        }

                        if (lane_predicate) {
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

                            gmem_V0 += (offset_n / 128) * ldV + (offset_k / 128);
                            gmem_V1 = gmem_V0;
                            if (offset_n / 128 + 1 < cdiv(sched.gemm_shape().y, 128)) {
                                gmem_V1 += ldV;
                            }

                            for (; k_iter > 0; --k_iter) {
                                int pipe = write_state.index();
                                ConsumerBar::wait(&consumer_bar[pipe], write_state.phase());
                                ProducerBar::arrive_and_expect_tx(&producer_bar[pipe], kTmaTxBytes);
                                // API A=act → GMMA-B; API B=weight → GMMA-A
                                gmem_A.Step(&producer_bar[pipe], &smem_act[pipe * TILE_M * TILE_K], mask_A);
                                gmem_B.Step(&producer_bar[pipe], &smem_weight[pipe * TILE_N * TILE_K], mask_B);
                                gmem_U.Step(&producer_bar[pipe], smem_U[pipe] + mc_offset_u, mask_A);
                                uint32_t uint_ptr_V = cast_smem_ptr_to_uint(smem_V[pipe]);
                                CP_ASYNC<CacheOp::kAlways, 4, 0>::apply(uint_ptr_V, gmem_V0, true);
                                CP_ASYNC<CacheOp::kAlways, 4, 0>::apply(uint_ptr_V + sizeof(Tv), gmem_V1, true);
                                ++gmem_V0;
                                ++gmem_V1;
                                cutlass::arch::cpasync_barrier_arrive_noinc(&producer_bar[pipe]);
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

                }  // scheduler loop

                // release last tile
                sched_state.release();

                if constexpr (kClusterSize > 1) {
                    if (lane_predicate) {
                        for (int i = 0; i < Stages; ++i) {
                            ConsumerBar::wait(&consumer_bar[write_state.index()], write_state.phase());
                            ++write_state;
                        }
                    }
                    __syncwarp();
                }
            }
            else if (warp_in_wg == 1 && cta_0) {
                if constexpr (!kIndexedGather) {
                    auto state = sched.init_producer(storage.sched);
                    while (state.next()) {
                        if constexpr (Scheduler::is_dynamic) {
                            producers_bar.arrive_and_wait_unaligned();
                        }
                    }
                    sched.tail(state);
                }
            }
        }
        else {
            cutlass::arch::warpgroup_reg_alloc<kMathRegs>();

            auto& smem_weight = storage.A;  // GMMA-A
            auto& smem_act    = storage.B;  // GMMA-B
            auto& smem_U      = storage.U;
            auto& smem_V      = storage.V;

            const int wg_idx_m = WG_M > 1 ? wg_idx % WG_M : 0;  // along BATCH
            const int wg_idx_n = WG_N > 1 ? wg_idx / WG_M : 0;  // along OUT

            // GMMA-A steps along OUT (TILE_N); GMMA-B along BATCH (TILE_M).
            auto smem_desc_A = make_smem_desc(&smem_weight[wg_idx_n * WG_TILE_N * TILE_K], 1);
            auto smem_desc_B = make_smem_desc(&smem_act[wg_idx_m * WG_TILE_M * TILE_K], 1);

            SmemDescIterV2<Stages, ((TILE_N * TILE_K) >> 4)> smem_iter_A{smem_desc_A};
            SmemDescIterV2<Stages, ((TILE_M * TILE_K) >> 4)> smem_iter_B{smem_desc_B};

            cutlass::arch::NamedBarrier barrier(WARPGROUP_SIZE, 2 + wg_idx);  // per math WG

            // All-math-WG sync (smem C is shared; one full-tile TMA store per tile).
            auto epi_synchronize = [&] {
                cutlass::arch::NamedBarrier::sync(kMathGroupSize,
                                                  cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            };

            cutlass::PipelineState<Stages> pipe_state{};

            const int warp_id = cutlass::canonical_warp_idx_sync();
            const int lane_id = cutlass::canonical_lane_idx();

            auto consumer_arrive = [&] {
                auto bar = &consumer_bar[pipe_state.index()];
                __syncwarp();
                if constexpr (kClusterSize > 1) {
                    ConsumerBar::arrive(bar, lane_id, lane_id < kClusterSize);
                }
                else {
                    if (lane_id == 0) {
                        ConsumerBar::arrive(bar);
                    }
                }
            };

            auto sched_state = sched.init_consumer(storage.sched);

            typename Scheduler::Tile* tile;

            sched_state.acquire(tile);

            while (tile->alive) {

                if (tile->is_valid_cta) {
                    AccumC accum_C{};
                    FragC  frag_C;

                    auto pred_W = Fetch_W(tile, wg_idx_n);

                    float scale_w[2];
                    auto  Load_W = [&] {
                        scale_w[0] = smem_V[pipe_state.index()][0];
                        scale_w[1] = smem_V[pipe_state.index()][1];
                    };

                    // Dense act scales along GMMA-N (BATCH): this thread's owned columns.
                    int u_pad = 0;
                    if constexpr (is_grouped_gemm) {
                        u_pad = tile->m0 % kAlignmentU;
                    }
                    const int            batch0 = wg_idx_m * WG_TILE_M;
                    typename GMMA::FragA act_scales{};
                    auto                 Load_A = [&] {
                        const int pipe = pipe_state.index();
                        // n = 2*t0 + 8*v2; t0 = lane%4
                        const int lane_n0 = 2 * (lane_id % 4);
                        PRAGMA_UNROLL
                        for (int s = 0; s < GMMA::OP_N / 8; ++s) {
                            const int n0          = batch0 + lane_n0 + 8 * s;
                            act_scales[2 * s + 0] = smem_U[pipe][u_pad + n0];
                            act_scales[2 * s + 1] = smem_U[pipe][u_pad + n0 + 1];
                        }
                    };

                    auto gmma = [&] {  //
                        GMMA::apply(smem_iter_A, smem_iter_B, frag_C, accum_C, scale_w, act_scales, pred_W);
                    };

                    if constexpr (is_grouped_gemm) {
                        if (threadIdx.x == 0) {  // single-store issuer
                            cute::tma_store_wait<0>();
                        }
                        epi_synchronize();
                    }

                    int k_iter = sched.k_iters_;

                    ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                    Load_W();
                    Load_A();
                    smem_iter_A.Reset(pipe_state.index());
                    smem_iter_B.Reset(pipe_state.index());
                    gmma();
                    consumer_arrive();
                    ++pipe_state;
                    --k_iter;

                    ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                    Load_W();
                    Load_A();
                    smem_iter_A.Reset(pipe_state.index());
                    smem_iter_B.Reset(pipe_state.index());

                    PRAGMA_NO_UNROLL
                    for (; k_iter > 1; --k_iter) {
                        gmma();
                        consumer_arrive();
                        ++pipe_state;
                        ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                        Load_W();
                        Load_A();
                        smem_iter_A.Reset(pipe_state.index());
                        smem_iter_B.Reset(pipe_state.index());
                    }

                    gmma();

                    const int thread_idx = threadIdx.x % WARPGROUP_SIZE;
                    if constexpr (!is_grouped_gemm) {
                        if (threadIdx.x == 0) {  // single-store issuer
                            cute::tma_store_wait<0>();
                        }
                        epi_synchronize();
                    }

                    consumer_arrive();
                    ++pipe_state;

                    auto run_epilogue = [&](auto fused_silu) {
                        constexpr bool kFuseSilu = decltype(fused_silu)::value;
                        using OutputTraits       = Output<kFuseSilu>;
                        using OutputT            = typename OutputTraits::Tc;
                        using LayoutC            = typename OutputTraits::LayoutC;

                        constexpr int kStoreOut = OutputTraits::kStoreOut;
                        OutputT*      smem_C    = reinterpret_cast<OutputT*>(smem_buf + kOutputOffset);

                        // Fused SiLU: silu(gate)*up on GMMA-M atoms {0,1}×{2,3}, then
                        // per-BATCH-row amax over fused OUT=128 → FP8 + W scales.
                        // (WG_N == 1 only; WG_N == 2 stages gate/up through smem below.)
                        if constexpr (kFuseSilu && WG_N == 1) {
                            static_assert(GMMA::ITER_N == 1);
                            constexpr float kQmax   = 448.f;
                            constexpr int   kScales = GMMA::kActScalesPerThread;

                            PRAGMA_UNROLL
                            for (int i_m = 0; i_m < 2; ++i_m) {
                                auto&         gate     = accum_C[i_m][0][0][0][0][0];
                                auto&         up       = accum_C[i_m + 2][0][0][0][0][0];
                                constexpr int kNumRegs = (int)(sizeof(gate) / sizeof(float));
                                PRAGMA_UNROLL
                                for (int i = 0; i < kNumRegs; ++i) {
                                    const float g = gate[i];
                                    const float u = up[i];
                                    gate[i]       = fdividef(g, 1.f + expf(-g)) * u;
                                }
                            }

                            // Partial amax over this thread's OUT rows in gate atoms {0,1}.
                            // CRegisters: [0]=(m,n0), [1]=(m,n1), [2]=(m+8,n0), [3]=(m+8,n1).
                            float amax[kScales];
                            PRAGMA_UNROLL
                            for (int i = 0; i < kScales; ++i) {
                                amax[i] = 0.f;
                            }
                            PRAGMA_UNROLL
                            for (int i_m = 0; i_m < 2; ++i_m) {
                                auto& gate = accum_C[i_m][0][0][0][0][0];
                                PRAGMA_UNROLL
                                for (int c = 0, s = 0; c < GMMA::OP_N; c += 8, ++s) {
                                    amax[2 * s + 0] = fmaxf(amax[2 * s + 0], fabsf(gate[c / 2 + 0]));
                                    amax[2 * s + 0] = fmaxf(amax[2 * s + 0], fabsf(gate[c / 2 + 2]));
                                    amax[2 * s + 1] = fmaxf(amax[2 * s + 1], fabsf(gate[c / 2 + 1]));
                                    amax[2 * s + 1] = fmaxf(amax[2 * s + 1], fabsf(gate[c / 2 + 3]));
                                }
                            }
                            // Reduce across OUT-owning lanes (t1) within the warp.
                            PRAGMA_UNROLL
                            for (int i = 0; i < kScales; ++i) {
                                amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffffu, amax[i], 4));
                                amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffffu, amax[i], 8));
                                amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffffu, amax[i], 16));
                            }
                            // Reduce across warps (t2) for full fused OUT=128.
                            {
                                const int t0      = lane_id % 4;
                                const int t1      = (lane_id / 4) % 8;
                                const int warp    = warp_id % 4;
                                float*    scratch = storage.fused_amax_scratch[wg_idx];
                                if (t1 == 0) {
                                    PRAGMA_UNROLL
                                    for (int i = 0; i < kScales; ++i) {
                                        scratch[(warp * 4 + t0) * 64 + i] = amax[i];
                                    }
                                }
                                barrier.sync();
                                PRAGMA_UNROLL
                                for (int i = 0; i < kScales; ++i) {
                                    float m = scratch[(0 * 4 + t0) * 64 + i];
                                    m       = fmaxf(m, scratch[(1 * 4 + t0) * 64 + i]);
                                    m       = fmaxf(m, scratch[(2 * 4 + t0) * 64 + i]);
                                    m       = fmaxf(m, scratch[(3 * 4 + t0) * 64 + i]);
                                    amax[i] = fmaxf(m, 1e-8f);
                                }
                                barrier.sync();
                            }

                            float inv[kScales];
                            PRAGMA_UNROLL
                            for (int i = 0; i < kScales; ++i) {
                                inv[i] = kQmax / amax[i];
                            }
                            PRAGMA_UNROLL
                            for (int i_m = 0; i_m < 2; ++i_m) {
                                auto& gate = accum_C[i_m][0][0][0][0][0];
                                PRAGMA_UNROLL
                                for (int c = 0, s = 0; c < GMMA::OP_N; c += 8, ++s) {
                                    gate[c / 2 + 0] *= inv[2 * s + 0];
                                    gate[c / 2 + 1] *= inv[2 * s + 1];
                                    gate[c / 2 + 2] *= inv[2 * s + 0];
                                    gate[c / 2 + 3] *= inv[2 * s + 1];
                                }
                            }

                            // W: one scale per BATCH row (act); n_group along OUT tiles.
                            if (param_W.ptr && (lane_id / 4) % 8 == 0 && (warp_id % 4) == 0) {
                                const int n_group  = tile->offset_n / TILE_N;
                                const int t0       = lane_id % 4;
                                const int batch0   = wg_idx_m * WG_TILE_M;
                                Tw*       W        = reinterpret_cast<Tw*>(param_W.ptr);
                                const int ldW      = param_W.stride;
                                int       row_end  = 0x7fffffff;
                                int       row_base = tile->offset_m + batch0;
                                if constexpr (is_grouped_gemm) {
                                    row_base += tile->m0;
                                    row_end = tile->m1;
                                }
                                PRAGMA_UNROLL
                                for (int c = 0, s = 0; c < GMMA::OP_N; c += 8, ++s) {
                                    const int row0 = row_base + 2 * t0 + c;
                                    const int row1 = row0 + 1;
                                    if (row0 < row_end && (batch0 + 2 * t0 + c) < TILE_M) {
                                        W[(int64_t)n_group * ldW + row0] = amax[2 * s + 0] / kQmax;
                                    }
                                    if (row1 < row_end && (batch0 + 2 * t0 + c + 1) < TILE_M) {
                                        W[(int64_t)n_group * ldW + row1] = amax[2 * s + 1] / kQmax;
                                    }
                                }
                            }
                        }

                        if constexpr (kFuseSilu && WG_N == 2) {
                            // WG_N == 2 staged fused SiLU: WG0's accum holds the gate
                            // slab, WG1's the up slab. Exchange raw f32 through smem, then
                            // each WG computes silu(g)*u for its fused half, cross-WG amax
                            // per BATCH row, quantize, write its half of smem C.
                            constexpr float kQmax    = 448.f;
                            constexpr int   kWgFused = (TILE_N / 2) / WG_N;
                            float*          stage    = storage.silu_stage;  // [TILE_N][TILE_M]

                            GMMA::foreach_C(accum_C, [&](auto& C, int m_atom, int n_atom) {
                                const int out0    = wg_idx_n * WG_TILE_N + m_atom * GMMA::OP_M;
                                const int batch0  = wg_idx_m * WG_TILE_M + n_atom * GMMA::OP_N;
                                const int m0      = out0 + (warp_id % 4) * 16 + lane_id / 4;
                                const int lane_n0 = 2 * (lane_id % 4);
                                PRAGMA_UNROLL
                                for (int c = 0; c < GMMA::OP_N; c += 8) {
                                    const int n0                                 = batch0 + lane_n0 + c;
                                    stage[(int64_t)m0 * TILE_M + n0]             = C[c / 2 + 0];
                                    stage[(int64_t)m0 * TILE_M + (n0 + 1)]       = C[c / 2 + 1];
                                    stage[(int64_t)(m0 + 8) * TILE_M + n0]       = C[c / 2 + 2];
                                    stage[(int64_t)(m0 + 8) * TILE_M + (n0 + 1)] = C[c / 2 + 3];
                                }
                            });
                            epi_synchronize();

                            // Linear repartition: kWgFused×TILE_M items, one row per thread pair.
                            constexpr int kItems = kWgFused * TILE_M / WARPGROUP_SIZE;
                            float         vals[kItems];
                            float         amax = 1e-8f;
                            const int     base = thread_idx * kItems;
                            PRAGMA_UNROLL
                            for (int i = 0; i < kItems; ++i) {
                                const int   idx = base + i;
                                const int   row = idx / kWgFused;
                                const int   of  = wg_idx_n * kWgFused + idx % kWgFused;
                                const float g   = stage[(int64_t)of * TILE_M + row];
                                const float u   = stage[(int64_t)(of + TILE_N / 2) * TILE_M + row];
                                vals[i]         = fdividef(g, 1.f + expf(-g)) * u;
                                amax            = fmaxf(amax, fabsf(vals[i]));
                            }
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));

                            float*    scratch = storage.fused_amax_scratch[wg_idx];
                            const int row     = base / kWgFused;
                            if ((thread_idx & 1) == 0) {
                                scratch[row] = amax;
                            }
                            epi_synchronize();
                            const float row_amax = fmaxf(
                                fmaxf(storage.fused_amax_scratch[0][row], storage.fused_amax_scratch[1][row]), 1e-8f);
                            const float inv = kQmax / row_amax;

                            PRAGMA_UNROLL
                            for (int i = 0; i < kItems; ++i) {
                                const int idx                      = base + i;
                                const int r                        = idx / kWgFused;
                                const int c                        = wg_idx_n * kWgFused + idx % kWgFused;
                                smem_C[(int64_t)r * kStoreOut + c] = OutputT(vals[i] * inv);
                            }

                            // W: one scale per BATCH row (spans both WGs); WG0 pair leaders store.
                            if (param_W.ptr && wg_idx == 0 && (thread_idx & 1) == 0) {
                                const int n_group = tile->offset_n / TILE_N;
                                Tw*       W       = reinterpret_cast<Tw*>(param_W.ptr);
                                const int ldW     = param_W.stride;
                                int       row_end = 0x7fffffff;
                                int       row_g   = tile->offset_m + row;
                                if constexpr (is_grouped_gemm) {
                                    row_g += tile->m0;
                                    row_end = tile->m1;
                                }
                                if (row_g < row_end && row < TILE_M) {
                                    W[(int64_t)n_group * ldW + row_g] = row_amax / kQmax;
                                }
                            }
                        }
                        else {
                            // WA CRegisters (OUT, BATCH) → problem smem C (BATCH, kStoreOut) row-major.
                            static_assert(GMMA::OP_N % 8 == 0);
                            GMMA::foreach_C(accum_C, [&](auto& C, int m_atom, int n_atom) {
                                if constexpr (kFuseSilu) {
                                    if (m_atom >= 2) {
                                        return;  // up atoms consumed
                                    }
                                }
                                // m_atom along OUT, n_atom along BATCH
                                const int out0    = wg_idx_n * (kStoreOut / WG_N) + m_atom * GMMA::OP_M;
                                const int batch0  = wg_idx_m * WG_TILE_M + n_atom * GMMA::OP_N;
                                const int m0      = out0 + (warp_id % 4) * 16 + lane_id / 4;
                                const int lane_n0 = 2 * (lane_id % 4);
                                PRAGMA_UNROLL
                                for (int c = 0, s = 0; c < GMMA::OP_N; c += 8, ++s) {
                                    const int n0                                     = batch0 + lane_n0 + c;
                                    smem_C[(int64_t)n0 * kStoreOut + m0]             = OutputT(C[c / 2 + 0]);
                                    smem_C[(int64_t)(n0 + 1) * kStoreOut + m0]       = OutputT(C[c / 2 + 1]);
                                    smem_C[(int64_t)n0 * kStoreOut + (m0 + 8)]       = OutputT(C[c / 2 + 2]);
                                    smem_C[(int64_t)(n0 + 1) * kStoreOut + (m0 + 8)] = OutputT(C[c / 2 + 3]);
                                }
                            });
                        }

                        cutlass::arch::fence_view_async_shared();
                        cute::tma_store_fence();
                        epi_synchronize();

                        const int offset_m = tile->offset_m;
                        const int offset_n = tile->offset_n;

                        const void* Cdesc = &tm_c;

                        // One full-tile store for all math WGs (smem C is shared).
                        if (threadIdx.x == 0) {
                            if constexpr (is_grouped_gemm) {
                                Cdesc = tensormap_buf + tile->group_idx * kTmaDescNum + kCdescIdx;
                            }
                            const int store_n = kFuseSilu ? (offset_n / 2) : offset_n;
                            cute::SM90_TMA_STORE::copy(Cdesc, smem_C, store_n, offset_m);
                            cute::tma_store_arrive();
                        }
                        if constexpr (is_grouped_gemm) {
                            if (threadIdx.x == 0) {
                                cute::tma_store_wait<0>();
                            }
                            epi_synchronize();
                        }
                    };

                    if constexpr (kSupportsFusedSilu) {
                        if (fuse_silu) {
                            run_epilogue(std::true_type{});
                        }
                        else {
                            run_epilogue(std::false_type{});
                        }
                    }
                    else {
                        run_epilogue(std::false_type{});
                    }
                }
                else if (tile->is_valid_cluster) {
                    int k_iter = sched.k_iters_;
                    for (; k_iter > 0; --k_iter) {
                        ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                        consumer_arrive();
                        ++pipe_state;
                    }
                }

                sched_state.release();
                sched_state.acquire(tile);

            }  // scheduler loop

            // release last tile
            sched_state.release();

            if (threadIdx.x % WARPGROUP_SIZE == 0) {
                cute::tma_store_wait<0>();
            }
        }

    }  // operator()

    // Weight-scale 128-block pred along OUT (problem N / GMMA-M).
    __device__ auto Fetch_W(typename Scheduler::Tile* tile, int wg_idx_n)
    {
        constexpr int BLK_SUBTILE_M = 128 / OUTER_M;
        static_assert(MMA_SUBTILE_M - 1 < BLK_SUBTILE_M + 1);

        Array<bool, MMA_SUBTILE_M> pred_W{};
        // i=0 is covered too: for 1 WG offset < 128 keeps pred_W[0] false (as before);
        // for WG_N == 2 the wg_idx_n term selects the second V scale block.
        const int offset = tile->offset_n % 128 + wg_idx_n * WG_TILE_N;
        PRAGMA_UNROLL
        for (int i = 0; i < MMA_SUBTILE_M; ++i) {
            pred_W[i] = (i * OUTER_M + offset) >= 128;
        }

        return pred_W;
    }
};

}  // namespace turbomind::gemm
