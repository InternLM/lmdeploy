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

    // Each math WG owns one contiguous (OUT,BATCH) tile.  Keeping the
    // TiledMma WG-local preserves the gate/up split for WG_N=2 and avoids a
    // full cooperative temporary fragment before applying block scales.
    using AtomLayoutMNK = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
    using Traits   = GmmaFP8WaTraits<WG_TILE_N, WG_TILE_M, TILE_K, AtomLayoutMNK>;
    using TiledMma = typename Traits::TiledMma;

    static constexpr int OP_M   = Traits::kOpM;
    static constexpr int OP_N   = Traits::kOpN;
    static constexpr int OP_K   = Traits::kOpK;
    static_assert(OP_N <= Tile::kMaxOpN);
    static_assert(WG_TILE_M == OP_N);

    // Fused SiLU: [g128|u128] along OUT. WG_N == 1 pairs GMMA-M atoms i with i+2 in
    // register; WG_N == 2 stages gate/up through smem (two M fragments per WG).
    static_assert(!kSupportsFusedSilu || (TILE_N == 256 && OP_M == 64 && WG_TILE_N / OP_M == 4 / WG_N));

    static constexpr int kMulticastA = multicast_a;
    static constexpr int kMulticastB = multicast_b;

    static constexpr int kClusterSize = kMulticastA * kMulticastB;

    static constexpr int Stages = Tile::Stages;

    static constexpr bool kSplitK     = false;
    static constexpr int  kChunkSizeK = TILE_K;

    static constexpr int WARPGROUP_SIZE = 128;

    static constexpr int kMathGroupSize = WARPGROUP_SIZE * WARPGROUPS;

    static constexpr int CTA_SIZE = WARPGROUP_SIZE * (WARPGROUPS + 1);

    static_assert(!kSupportsFusedSilu || WG_N != 2
                  || (WG_M == 1 && WG_TILE_N / OP_M == 2 && OP_N == WG_TILE_M));
    static_assert(!kSupportsFusedSilu || WG_N != 2
                  || (OP_M / WG_N == 32 && (OP_M / WG_N) * TILE_M % WARPGROUP_SIZE == 0
                      && ((OP_M / WG_N) * TILE_M / WARPGROUP_SIZE) % 4 == 0));

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

    using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
    using PipelineState    = typename MainloopPipeline::PipelineState;
    using PipelineStorage  = typename MainloopPipeline::SharedStorage;
    using ClusterShape     = cute::Shape<cute::Int<kClusterSize>, cute::_1, cute::_1>;
    using ProducerBar      = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBar      = cutlass::arch::ClusterBarrier;

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

    // One canonical SMEM layout is shared by the TMA/cp.async producers and
    // the CuTe GMMA descriptor fragments.  Indexed gather uses its 2D B view.
    using SmemLayoutA = decltype(cute::tile_to_shape(typename Traits::SmemLayoutAtomA{},
                                                     cute::make_shape(cute::Int<TILE_N>{},
                                                                      cute::Int<TILE_K>{},
                                                                      cute::Int<Stages>{}),
                                                     cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutB = decltype(cute::tile_to_shape(typename Traits::SmemLayoutAtomB{},
                                                     cute::make_shape(cute::Int<TILE_M>{},
                                                                      cute::Int<TILE_K>{},
                                                                      cute::Int<Stages>{}),
                                                     cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutB_2D = decltype(cute::tile_to_shape(typename Traits::SmemLayoutAtomB{},
                                                        cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_K>{}),
                                                        cute::Step<cute::_1, cute::_2>{}));

    static constexpr int  kGatherVec       = 16 / (int)sizeof(Ta);
    static constexpr int  kGatherThreadsK  = TILE_K / kGatherVec;
    static constexpr int  kGatherThreadsM  = WARPGROUP_SIZE / kGatherThreadsK;
    static constexpr bool kUseTiledGather  = TILE_M >= kGatherThreadsM && TILE_M % kGatherThreadsM == 0;
    static constexpr int  kGatherSlots     = cute::ceil_div(TILE_M * kGatherThreadsK, WARPGROUP_SIZE);
    using GatherCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>, Ta>;
    using GatherTiledCopy = decltype(cute::make_tiled_copy(
        GatherCopyAtom{},
        cute::Layout<cute::Shape<cute::Int<kGatherThreadsM>, cute::Int<kGatherThreadsK>>,
                     cute::Stride<cute::Int<kGatherThreadsK>, cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1, cute::Int<kGatherVec>>>{}));
    static_assert(kGatherVec * (int)sizeof(Ta) == 16);
    static_assert(kGatherThreadsM * kGatherThreadsK == WARPGROUP_SIZE);
    static_assert(cute::size(GatherTiledCopy{}) == WARPGROUP_SIZE);
    static_assert(!kUseTiledGather || TILE_M * kGatherThreadsK % WARPGROUP_SIZE == 0);

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
        // Bounded gate/up exchange for WG_N=2 fused SiLU.
        static constexpr int kSiluStageElems =
            (kSupportsFusedSilu && WG_N == 2) ? 2 * OP_M * WG_TILE_M : 1;
        __align__(1024) float silu_stage[kSiluStageElems];
        __align__(128) Tu U[Stages][round_up<int>(kBoxU, 128)];  // at least 128 byte alignment
        __align__(128) Tv V[Stages][2];
        __align__(8) uint64_t producer_bar[Stages];
        __align__(8) uint64_t consumer_bar[Stages];
        PipelineStorage             pipeline;
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

    // BF16 occupies four times the bytes of the fused FP8 half-output and is
    // therefore the maximum runtime-selectable shared-memory footprint.
    static constexpr int kSmemSize = GetSmemSize(false);
    static_assert(kSmemSize <= (228 << 10));

    static constexpr int OUTER_M = Traits::kOuterM;
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

        const int wg_idx     = cutlass::canonical_warp_group_idx();
        const int warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
        const int lane_id    = threadIdx.x % WARP_SIZE;

        if (threadIdx.x == 0) {
            if constexpr (!kIndexedGather) {
                PRAGMA_UNROLL
                for (int s = 0; s < Stages; ++s) {
                    ProducerBar::init(&producer_bar[s], 2);
                    ConsumerBar::init(&consumer_bar[s], WARPGROUPS * kClusterSize * 4);
                }
                if constexpr (kClusterSize > 1) {
                    cutlass::arch::fence_barrier_init();
                }
            }
            sched.init_dyanmic(storage.sched, kClusterSize * (WARPGROUPS * 4 + 1));
        }

        typename MainloopPipeline::Params pp;
        pp.transaction_bytes = (uint32_t)kTmaTxBytes;
        pp.num_consumers     = (uint32_t)kMathGroupSize;
        // Indexed: one TMA leader plus one cp.async .noinc arrival from every
        // gather thread. Dense: one TMA leader plus its V cp.async arrival.
        pp.num_producers     = kIndexedGather ? (1 + WARPGROUP_SIZE) : (1 + 1);
        pp.initializing_warp = 0;

        if (wg_idx == WARPGROUPS) {
            if constexpr (kIndexedGather) {
                pp.role      = MainloopPipeline::ThreadCategory::Producer;
                pp.is_leader = warp_in_wg == 0 && lane_id == 0;
            }
            else {
                pp.role      = warp_in_wg == 0 ? MainloopPipeline::ThreadCategory::Producer :
                                                MainloopPipeline::ThreadCategory::NonParticipant;
                pp.is_leader = warp_in_wg == 0 && lane_id == 0;
            }
        }
        else {
            pp.role      = MainloopPipeline::ThreadCategory::Consumer;
            pp.is_leader = 0;
        }

        MainloopPipeline pipeline(storage.pipeline, pp, ClusterShape{});

        if (threadIdx.x == 0) {
            cutlass::arch::fence_view_async_shared();
        }

        (kClusterSize > 1) ? cute::cluster_sync() : __syncthreads();

        if (wg_idx == WARPGROUPS) {
            cutlass::arch::warpgroup_reg_dealloc<kProducerRegs>();

            static_assert(TILE_M % kMulticastA == 0);
            static_assert(TILE_N % kMulticastB == 0);

            cutlass::arch::NamedBarrier producers_bar(WARP_SIZE * 2, 7);

            const int  warp_id    = cutlass::canonical_warp_idx_sync();
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

                PipelineState write_state = cutlass::make_producer_start_state<MainloopPipeline>();

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

                constexpr int nvec = TILE_M * kGatherThreadsK;
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
                    const Ta* gather_src[kGatherSlots];
                    Ta*       gather_dst[kGatherSlots];
                    bool      gather_pred[kGatherSlots];
                    int       gather_m[kGatherSlots];
                    int       gather_k[kGatherSlots];
                    bool      gather_slot_valid[kGatherSlots];
                    const Tu* src_u_base;
                    int       m_u;
                    bool      pred_u;
                    const int u_pad = m0 % kAlignmentU;

                    if constexpr (kUseTiledGather) {
                        auto gather_thr = GatherTiledCopy{}.get_slice(prod_tid);
                        auto gather_smem =
                            cute::make_tensor(cute::make_smem_ptr(storage.B.data()), SmemLayoutB_2D{});
                        auto gather_dst_part = gather_thr.partition_D(gather_smem);
                        auto gather_identity = cute::make_identity_tensor(
                            cute::Shape<cute::Int<TILE_M>, cute::Int<TILE_K>>{});
                        auto gather_coord = gather_thr.partition_D(gather_identity);
                        static_assert(cute::size<0>(gather_coord) == kGatherVec);
                        static_assert(cute::size<1>(gather_coord) == kGatherSlots);
                        static_assert(cute::size<2>(gather_coord) == 1);

                        PRAGMA_UNROLL
                        for (int slot = 0; slot < kGatherSlots; ++slot) {
                            const auto coord  = gather_coord(cute::make_coord(0, 0), slot, 0);
                            const int  m      = cute::get<0>(coord);
                            const int  kk     = cute::get<1>(coord);
                            const int  packed = m0 + offset_m + m;
                            const bool row_ok = (offset_m + m) < M_group;
                            const int  token  = (idxs && row_ok) ? __ldg(idxs + packed) : packed;
                            gather_src[slot]  = act_gmem + (int64_t)token * ldA + kk;
                            gather_dst[slot]  = &gather_dst_part(cute::make_coord(0, 0), slot, 0);
                            gather_pred[slot] = row_ok;
                            gather_k[slot]    = kk;
                        }
                    }
                    else {
                        // TILE_M=8 has only 64 vectors.  Keep the small predicated
                        // fallback instead of forcing an underfilled 128-thread TV map.
                        PRAGMA_UNROLL
                        for (int slot = 0; slot < kGatherSlots; ++slot) {
                            const int  i      = prod_tid + slot * WARPGROUP_SIZE;
                            const bool in_vec = i < nvec;
                            const int  m      = in_vec ? i / kGatherThreadsK : 0;
                            const int  kk     = in_vec ? (i % kGatherThreadsK) * kGatherVec : 0;
                            const int  packed = m0 + offset_m + m;
                            const bool row_ok = in_vec && (offset_m + m) < M_group;
                            const int  token  = (idxs && row_ok) ? __ldg(idxs + packed) : packed;
                            gather_m[slot]          = m;
                            gather_k[slot]          = kk;
                            gather_pred[slot]       = row_ok;
                            gather_slot_valid[slot] = in_vec;
                            gather_src[slot]        = act_gmem + (int64_t)token * ldA + kk;
                        }
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
                        pipeline.producer_acquire(write_state);
                        auto*     bar  = pipeline.producer_get_barrier(write_state);
                        const int pipe = write_state.index();

                        if (warp_in_wg == 0 && lane_predicate) {
                            gmem_B.Step(bar, &smem_weight[pipe * TILE_N * TILE_K], mask_B);
                            uint32_t uint_ptr_V = cast_smem_ptr_to_uint(smem_V[pipe]);
                            CP_ASYNC<CacheOp::kAlways, 4, 0>::apply(uint_ptr_V, gmem_V0, true);
                            CP_ASYNC<CacheOp::kAlways, 4, 0>::apply(uint_ptr_V + sizeof(Tv), gmem_V1, true);
                            ++gmem_V0;
                            ++gmem_V1;
                        }

                        {
                            PRAGMA_UNROLL
                            for (int slot = 0; slot < kGatherSlots; ++slot) {
                                if constexpr (!kUseTiledGather) {
                                    // pred=false still zeros dst; idle threads must not issue.
                                    if (!gather_slot_valid[slot]) {
                                        continue;
                                    }
                                }
                                const bool pred = gather_pred[slot] && (coord_k + gather_k[slot]) < K;
                                Ta* dst;
                                if constexpr (kUseTiledGather) {
                                    dst = gather_dst[slot] + pipe * TILE_M * TILE_K;
                                }
                                else {
                                    auto sB = cute::make_tensor(
                                        cute::make_smem_ptr(smem_act + pipe * TILE_M * TILE_K),
                                        SmemLayoutB_2D{});
                                    dst = &sB(gather_m[slot], gather_k[slot]);
                                }
                                cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>::copy(
                                    *reinterpret_cast<const uint4*>(gather_src[slot]),
                                    *reinterpret_cast<uint4*>(dst),
                                    pred);
                                gather_src[slot] += TILE_K;
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

                            cutlass::arch::cpasync_barrier_arrive_noinc(bar);
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
                    if (lane_predicate) {
                        pipeline.producer_tail(write_state);
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
                                const int pipe = write_state.index();
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

            // Canonical swizzled SMEM tensors are the single source of truth
            // for CuTe's GMMA descriptors.  Slice a contiguous tile per math
            // WG to preserve the established gate/up ownership.
            auto sA_full = cute::make_tensor(cute::make_smem_ptr(smem_weight.data()), SmemLayoutA{});
            auto sB_full = cute::make_tensor(cute::make_smem_ptr(smem_act.data()), SmemLayoutB{});
            auto sA = cute::local_tile(
                sA_full,
                cute::make_shape(cute::Int<WG_TILE_N>{}, cute::Int<TILE_K>{}, cute::Int<Stages>{}),
                cute::make_coord(wg_idx_n, 0, 0));
            auto sB = cute::local_tile(
                sB_full,
                cute::make_shape(cute::Int<WG_TILE_M>{}, cute::Int<TILE_K>{}, cute::Int<Stages>{}),
                cute::make_coord(wg_idx_m, 0, 0));

            TiledMma tiled_mma;
            auto     thr_mma = tiled_mma.get_thread_slice(threadIdx.x % WARPGROUP_SIZE);
            auto     tCrA    = thr_mma.make_fragment_A(thr_mma.partition_A(sA));
            auto     tCrB    = thr_mma.make_fragment_B(thr_mma.partition_B(sB));

            CUTE_STATIC_ASSERT_V(cute::size<1>(tCrA) == cute::Int<Traits::kRestM>{});
            CUTE_STATIC_ASSERT_V(cute::size<1>(tCrB) == cute::Int<Traits::kRestN>{});
            CUTE_STATIC_ASSERT_V(cute::size<2>(tCrA) == cute::Int<Traits::kKBlocks>{});
            CUTE_STATIC_ASSERT_V(cute::size<2>(tCrB) == cute::Int<Traits::kKBlocks>{});

            cutlass::arch::NamedBarrier barrier(WARPGROUP_SIZE, 2 + wg_idx);  // per math WG

            // All-math-WG sync (smem C is shared; one full-tile TMA store per tile).
            auto epi_synchronize = [&] {
                cutlass::arch::NamedBarrier::sync(kMathGroupSize,
                                                  cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            };

            PipelineState pipe_state{};

            const int warp_id = cutlass::canonical_warp_idx_sync();
            const int lane_id = cutlass::canonical_lane_idx();

            auto consumer_wait = [&] {
                if constexpr (kIndexedGather) {
                    auto token = pipeline.consumer_try_wait(pipe_state);
                    pipeline.consumer_wait(pipe_state, token);
                }
                else {
                    ProducerBar::wait(&producer_bar[pipe_state.index()], pipe_state.phase());
                }
            };
            auto consumer_release = [&] {
                if constexpr (kIndexedGather) {
                    pipeline.consumer_release(pipe_state);
                }
                else {
                    auto* bar = &consumer_bar[pipe_state.index()];
                    __syncwarp();
                    if constexpr (kClusterSize > 1) {
                        ConsumerBar::arrive(bar, lane_id, lane_id < kClusterSize);
                    }
                    else if (lane_id == 0) {
                        ConsumerBar::arrive(bar);
                    }
                }
            };

            auto sched_state = sched.init_consumer(storage.sched);

            typename Scheduler::Tile* tile;

            sched_state.acquire(tile);

            while (tile->alive) {

                if (tile->is_valid_cta) {
                    auto accum_C = cute::partition_fragment_C(
                        tiled_mma, cute::take<0, 2>(typename Traits::TileShape{}));
                    auto frag_C0 = cute::make_fragment_like(accum_C(cute::_, cute::Int<0>{}, cute::Int<0>{}));
                    auto frag_C1 = cute::make_fragment_like(frag_C0);
                    cute::clear(accum_C);

                    CUTE_STATIC_ASSERT_V(cute::size<2>(accum_C) == cute::_1{});
                    static_assert(cute::size<1>(typename decltype(accum_C)::layout_type{}) >= 2);
                    static_assert(cute::size<1>(typename decltype(accum_C)::layout_type{}) <= 4);

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
                    float act_scales[Traits::kActScalesPerThread];
                    auto                 Load_A = [&] {
                        const int pipe = pipe_state.index();
                        // n = 2*t0 + 8*v2; t0 = lane%4
                        const int lane_n0 = 2 * (lane_id % 4);
                        PRAGMA_UNROLL
                        for (int s = 0; s < OP_N / 8; ++s) {
                            const int n0          = batch0 + lane_n0 + 8 * s;
                            act_scales[2 * s + 0] = smem_U[pipe][u_pad + n0];
                            act_scales[2 * s + 1] = smem_U[pipe][u_pad + n0 + 1];
                        }
                    };

                    auto gmma = [&] {
                        const int read = pipe_state.index();
                        auto      tCrA_stage = tCrA(cute::_, cute::_, cute::_, read);
                        auto      tCrB_stage = tCrB(cute::_, cute::_, cute::_, read);
                        auto issue = [&](auto m, auto& frag_C) {
                            cute::warpgroup_fence_operand(frag_C);
                            tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
                            cute::warpgroup_arrive();
                            cute::for_each(
                                cute::make_seq<cute::size<2>(typename decltype(tCrA)::layout_type{})>{},
                                [&](auto k) {
                                    cute::gemm(tiled_mma,
                                               tCrA_stage(cute::_, m, k),
                                               tCrB_stage(cute::_, cute::Int<0>{}, k),
                                               frag_C);
                                    tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
                                });
                            cute::warpgroup_commit_batch();
                        };
                        auto scale = [&](auto m, auto& frag_C) {
                            cute::warpgroup_fence_operand(frag_C);
                            auto        accum = accum_C(cute::_, m, cute::Int<0>{});
                            const int   wi    = (m * OP_M) / OUTER_M;
                            const float sw    = pred_W[wi] ? scale_w[1] : scale_w[0];
                            PRAGMA_UNROLL
                            for (int c = 0, s = 0; c < OP_N; c += 8, ++s) {
                                const float sa0 = act_scales[2 * s + 0];
                                const float sa1 = act_scales[2 * s + 1];
                                accum(c / 2 + 0) += sw * sa0 * frag_C(c / 2 + 0);
                                accum(c / 2 + 1) += sw * sa1 * frag_C(c / 2 + 1);
                                accum(c / 2 + 2) += sw * sa0 * frag_C(c / 2 + 2);
                                accum(c / 2 + 3) += sw * sa1 * frag_C(c / 2 + 3);
                            }
                        };
                        if constexpr (kIndexedGather || WARPGROUPS == 1) {
                            // Preserve the established indexed/low-RF schedule
                            // literally: commit, wait<0>, then scale each atom.
                            cute::for_each(
                                cute::make_seq<cute::size<1>(typename decltype(accum_C)::layout_type{})>{},
                                [&](auto m) {
                                    cute::warpgroup_fence_operand(frag_C0);
                                    tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
                                    cute::warpgroup_arrive();
                                    cute::for_each(
                                        cute::make_seq<cute::size<2>(typename decltype(tCrA)::layout_type{})>{},
                                        [&](auto k) {
                                            cute::gemm(tiled_mma,
                                                       tCrA_stage(cute::_, m, k),
                                                       tCrB_stage(cute::_, cute::Int<0>{}, k),
                                                       frag_C0);
                                            tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
                                        });
                                    cute::warpgroup_commit_batch();
                                    cute::warpgroup_wait<0>();
                                    cute::warpgroup_fence_operand(frag_C0);

                                    auto        accum = accum_C(cute::_, m, cute::Int<0>{});
                                    const int   wi    = (m * OP_M) / OUTER_M;
                                    const float sw    = pred_W[wi] ? scale_w[1] : scale_w[0];
                                    PRAGMA_UNROLL
                                    for (int c = 0, s = 0; c < OP_N; c += 8, ++s) {
                                        const float sa0 = act_scales[2 * s + 0];
                                        const float sa1 = act_scales[2 * s + 1];
                                        accum(c / 2 + 0) += sw * sa0 * frag_C0(c / 2 + 0);
                                        accum(c / 2 + 1) += sw * sa1 * frag_C0(c / 2 + 1);
                                        accum(c / 2 + 2) += sw * sa0 * frag_C0(c / 2 + 2);
                                        accum(c / 2 + 3) += sw * sa1 * frag_C0(c / 2 + 3);
                                    }
                                });
                        }
                        else {
                            static_assert(Traits::kRestM == 2 || Traits::kRestM == 4);

                            // Keep one committed fragment in flight while
                            // scaling the preceding atom.
                            issue(cute::Int<0>{}, frag_C0);
                            issue(cute::Int<1>{}, frag_C1);
                            cute::warpgroup_wait<1>();
                            scale(cute::Int<0>{}, frag_C0);

                            if constexpr (Traits::kRestM > 2) {
                                issue(cute::Int<2>{}, frag_C0);
                                cute::warpgroup_wait<1>();
                                scale(cute::Int<1>{}, frag_C1);
                            }
                            if constexpr (Traits::kRestM > 3) {
                                issue(cute::Int<3>{}, frag_C1);
                                cute::warpgroup_wait<1>();
                                scale(cute::Int<2>{}, frag_C0);
                            }

                            cute::warpgroup_wait<0>();
                            if constexpr (Traits::kRestM == 2) {
                                scale(cute::Int<1>{}, frag_C1);
                            }
                            else {
                                scale(cute::Int<3>{}, frag_C1);
                            }
                        }
                    };

                    if constexpr (is_grouped_gemm) {
                        if (threadIdx.x == 0) {  // single-store issuer
                            cute::tma_store_wait<0>();
                        }
                        epi_synchronize();
                    }

                    int k_iter = sched.k_iters_;

                    consumer_wait();
                    Load_W();
                    Load_A();
                    gmma();
                    consumer_release();
                    ++pipe_state;
                    --k_iter;

                    consumer_wait();
                    Load_W();
                    Load_A();

                    PRAGMA_NO_UNROLL
                    for (; k_iter > 1; --k_iter) {
                        gmma();
                        consumer_release();
                        ++pipe_state;
                        consumer_wait();
                        Load_W();
                        Load_A();
                    }

                    gmma();

                    const int thread_idx = threadIdx.x % WARPGROUP_SIZE;
                    if constexpr (!is_grouped_gemm) {
                        if (threadIdx.x == 0) {  // single-store issuer
                            cute::tma_store_wait<0>();
                        }
                        epi_synchronize();
                    }

                    consumer_release();
                    ++pipe_state;

                    auto run_epilogue = [&](auto fused_silu) {
                        constexpr bool kFuseSilu = decltype(fused_silu)::value;
                        using OutputTraits       = Output<kFuseSilu>;
                        using OutputT            = typename OutputTraits::Tc;

                        constexpr int kStoreOut = OutputTraits::kStoreOut;
                        OutputT*      smem_C    = reinterpret_cast<OutputT*>(smem_buf + kOutputOffset);

                        if constexpr (!kFuseSilu) {
                            // Native WA ownership is:
                            //   out   = t1 + 16*t2 + 8*v1
                            //   batch = 2*t0 + v0 + 8*v2.
                            // Scalar b16 stores are cheaper than either STSM_T or exchanging
                            // adjacent OUT lanes with SHFL+PRMT on this path.  Keep all shared
                            // offsets 32-bit; only global matrix strides require 64-bit arithmetic.
                            PRAGMA_UNROLL
                            for (int m_atom = 0; m_atom < cute::size<1>(accum_C); ++m_atom) {
                                PRAGMA_UNROLL
                                for (int n_atom = 0; n_atom < cute::size<2>(accum_C); ++n_atom) {
                                    auto      C       = accum_C(cute::_, m_atom, n_atom);
                                    const int out0    = wg_idx_n * WG_TILE_N + m_atom * OP_M;
                                    const int batch0  = wg_idx_m * WG_TILE_M + n_atom * OP_N;
                                    const int m0      = out0 + (warp_id % 4) * 16 + lane_id / 4;
                                    const int lane_n0 = 2 * (lane_id % 4);
                                    PRAGMA_UNROLL
                                    for (int c = 0; c < OP_N; c += 8) {
                                        const int n0                         = batch0 + lane_n0 + c;
                                        smem_C[n0 * kStoreOut + m0]           = OutputT(C(c / 2 + 0));
                                        smem_C[(n0 + 1) * kStoreOut + m0]     = OutputT(C(c / 2 + 1));
                                        smem_C[n0 * kStoreOut + m0 + 8]       = OutputT(C(c / 2 + 2));
                                        smem_C[(n0 + 1) * kStoreOut + m0 + 8] = OutputT(C(c / 2 + 3));
                                    }
                                }
                            }
                        }

                        // Fused SiLU: silu(gate)*up on GMMA-M atoms {0,1}×{2,3}, then
                        // per-BATCH-row amax over fused OUT=128 → FP8 + W scales.
                        // (WG_N == 1 only; WG_N == 2 stages gate/up through smem below.)
                        if constexpr (kFuseSilu && WG_N == 1) {
                            CUTE_STATIC_ASSERT_V(cute::size<2>(accum_C) == cute::_1{});
                            constexpr float kQmax   = 448.f;
                            constexpr int   kScales = Traits::kActScalesPerThread;

                            PRAGMA_UNROLL
                            for (int i_m = 0; i_m < 2; ++i_m) {
                                auto          gate     = accum_C(cute::_, i_m, 0);
                                auto          up       = accum_C(cute::_, i_m + 2, 0);
                                constexpr int kNumRegs = OP_N / 2;
                                PRAGMA_UNROLL
                                for (int i = 0; i < kNumRegs; ++i) {
                                    const float g = gate(i);
                                    const float u = up(i);
                                    gate(i)       = fdividef(g, 1.f + expf(-g)) * u;
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
                                auto gate = accum_C(cute::_, i_m, 0);
                                PRAGMA_UNROLL
                                for (int c = 0, s = 0; c < OP_N; c += 8, ++s) {
                                    amax[2 * s + 0] = fmaxf(amax[2 * s + 0], fabsf(gate(c / 2 + 0)));
                                    amax[2 * s + 0] = fmaxf(amax[2 * s + 0], fabsf(gate(c / 2 + 2)));
                                    amax[2 * s + 1] = fmaxf(amax[2 * s + 1], fabsf(gate(c / 2 + 1)));
                                    amax[2 * s + 1] = fmaxf(amax[2 * s + 1], fabsf(gate(c / 2 + 3)));
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
                                auto gate = accum_C(cute::_, i_m, 0);
                                PRAGMA_UNROLL
                                for (int c = 0, s = 0; c < OP_N; c += 8, ++s) {
                                    gate(c / 2 + 0) *= inv[2 * s + 0];
                                    gate(c / 2 + 1) *= inv[2 * s + 1];
                                    gate(c / 2 + 2) *= inv[2 * s + 0];
                                    gate(c / 2 + 3) *= inv[2 * s + 1];
                                }
                            }

                            // W: one scale per BATCH row (act); n_group along OUT tiles.
                            if (param_W.ptr && (lane_id / 4) % 8 == 0 && (warp_id % 4) == 0) {
                                const int n_group  = tile->offset_n / TILE_N;
                                const int t0       = lane_id % 4;
                                const int batch0   = wg_idx_m * WG_TILE_M;
                                Tw*       W        = reinterpret_cast<Tw*>(param_W.ptr);
                                const int ldW      = param_W.stride;
                                int       row_end  = sched.gemm_shape().x;
                                int       row_base = tile->offset_m + batch0;
                                if constexpr (is_grouped_gemm) {
                                    row_base += tile->m0;
                                    row_end = tile->m1;
                                }
                                PRAGMA_UNROLL
                                for (int c = 0, s = 0; c < OP_N; c += 8, ++s) {
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
                            // Gate and up live in separate WGs.  Exchange one 64-wide
                            // GMMA-M atom at a time instead of materializing the entire
                            // 256xBATCH accumulator tile.  Each WG computes 32 outputs per
                            // pass; the two pass-local fragments stay in registers so amax
                            // still spans the complete 128-wide quantization group.
                            constexpr float kQmax          = 448.f;
                            constexpr int   kPasses        = 2;
                            constexpr int   kWgFusedPass   = OP_M / WG_N;
                            constexpr int   kItemsPass     = kWgFusedPass * TILE_M / WARPGROUP_SIZE;
                            constexpr int   kItems         = kPasses * kItemsPass;
                            float* stage = storage.silu_stage;  // [gate/up][OUT64][BATCH]
                            float  vals[kItems];
                            float  amax = 1e-8f;
                            const int base = thread_idx * kItemsPass;

                            PRAGMA_UNROLL
                            for (int pass = 0; pass < kPasses; ++pass) {
                                auto      C       = accum_C(cute::_, pass, 0);
                                const int m0      = (warp_id % 4) * 16 + lane_id / 4;
                                const int lane_n0 = 2 * (lane_id % 4);
                                PRAGMA_UNROLL
                                for (int c = 0; c < OP_N; c += 8) {
                                    const int n0 = lane_n0 + c;
                                    const int plane_offset = wg_idx_n * OP_M * TILE_M;
                                    stage[plane_offset + m0 * TILE_M + n0] = C(c / 2 + 0);
                                    stage[plane_offset + m0 * TILE_M + n0 + 1] = C(c / 2 + 1);
                                    stage[plane_offset + (m0 + 8) * TILE_M + n0] = C(c / 2 + 2);
                                    stage[plane_offset + (m0 + 8) * TILE_M + n0 + 1] = C(c / 2 + 3);
                                }
                                epi_synchronize();

                                PRAGMA_UNROLL
                                for (int i = 0; i < kItemsPass; ++i) {
                                    const int   idx = base + i;
                                    const int   row = idx / kWgFusedPass;
                                    const int   of  = wg_idx_n * kWgFusedPass + idx % kWgFusedPass;
                                    const float g   = stage[of * TILE_M + row];
                                    const float u   = stage[(OP_M + of) * TILE_M + row];
                                    const float v   = fdividef(g, 1.f + expf(-g)) * u;
                                    vals[pass * kItemsPass + i] = v;
                                    amax = fmaxf(amax, fabsf(v));
                                }
                                // All reads must retire before the next atom overwrites the
                                // bounded exchange tile.
                                epi_synchronize();
                            }
                            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));

                            float*    scratch = storage.fused_amax_scratch[wg_idx];
                            const int row     = base / kWgFusedPass;
                            if ((thread_idx & 1) == 0) {
                                scratch[row] = amax;
                            }
                            epi_synchronize();
                            const float row_amax = fmaxf(
                                fmaxf(storage.fused_amax_scratch[0][row], storage.fused_amax_scratch[1][row]), 1e-8f);
                            const float inv = kQmax / row_amax;

                            PRAGMA_UNROLL
                            for (int pass = 0; pass < kPasses; ++pass) {
                                PRAGMA_UNROLL
                                for (int i = 0; i < kItemsPass; i += 4) {
                                    const int idx = base + i;
                                    const int r   = idx / kWgFusedPass;
                                    const int c   = pass * OP_M + wg_idx_n * kWgFusedPass
                                                  + idx % kWgFusedPass;
                                    Array<OutputT, 4> q;
                                    PRAGMA_UNROLL
                                    for (int j = 0; j < 4; ++j) {
                                        q[j] = OutputT(vals[pass * kItemsPass + i + j] * inv);
                                    }
                                    *reinterpret_cast<uint32_t*>(smem_C + r * kStoreOut + c) =
                                        reinterpret_cast<uint32_t&>(q);
                                }
                            }

                            // W: one scale per BATCH row (spans both WGs); WG0 pair leaders store.
                            if (param_W.ptr && wg_idx == 0 && (thread_idx & 1) == 0) {
                                const int n_group = tile->offset_n / TILE_N;
                                Tw*       W       = reinterpret_cast<Tw*>(param_W.ptr);
                                const int ldW     = param_W.stride;
                                int       row_end = sched.gemm_shape().x;
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
                        else if constexpr (kFuseSilu) {
                            // WA CRegisters (OUT, BATCH) → problem smem C (BATCH, kStoreOut) row-major.
                            static_assert(OP_N % 8 == 0);
                            PRAGMA_UNROLL
                            for (int m_atom = 0; m_atom < 2; ++m_atom) {
                                PRAGMA_UNROLL
                                for (int n_atom = 0; n_atom < cute::size<2>(accum_C); ++n_atom) {
                                    auto      C       = accum_C(cute::_, m_atom, n_atom);
                                    const int out0    = wg_idx_n * (kStoreOut / WG_N) + m_atom * OP_M;
                                    const int batch0  = wg_idx_m * WG_TILE_M + n_atom * OP_N;
                                    const int m0      = out0 + (warp_id % 4) * 16 + lane_id / 4;
                                    const int lane_n0 = 2 * (lane_id % 4);
                                    PRAGMA_UNROLL
                                    for (int c = 0, s = 0; c < OP_N; c += 8, ++s) {
                                        Array<OutputT, 4> q;
                                        q[0] = OutputT(C(c / 2 + 0));
                                        q[1] = OutputT(C(c / 2 + 1));
                                        q[2] = OutputT(C(c / 2 + 2));
                                        q[3] = OutputT(C(c / 2 + 3));

                                        const uint32_t own  = reinterpret_cast<uint32_t&>(q);
                                        const uint32_t peer = __shfl_xor_sync(0xffffffffu, own, 4);

                                        // t1^1 owns the adjacent OUT column.  Even t1
                                        // lanes pack the two columns for both BATCH rows:
                                        //   lo = [own.byte0, peer.byte0],
                                        //   hi = [own.byte1, peer.byte1].
                                        // This halves the shared-store instruction count
                                        // for one SHFL and two PRMTs per four FP8 values.
                                        if (((lane_id / 4) & 1) == 0) {
                                            const int      n0   = batch0 + lane_n0 + c;
                                            const uint32_t lohi = __byte_perm(own, peer, 0x5140);
                                            const uint32_t lohi8 = __byte_perm(own, peer, 0x7362);
                                            *reinterpret_cast<uint16_t*>(smem_C + n0 * kStoreOut + m0) =
                                                static_cast<uint16_t>(lohi);
                                            *reinterpret_cast<uint16_t*>(smem_C + (n0 + 1) * kStoreOut + m0) =
                                                static_cast<uint16_t>(lohi >> 16);
                                            *reinterpret_cast<uint16_t*>(smem_C + n0 * kStoreOut + m0 + 8) =
                                                static_cast<uint16_t>(lohi8);
                                            *reinterpret_cast<uint16_t*>(smem_C + (n0 + 1) * kStoreOut + m0 + 8) =
                                                static_cast<uint16_t>(lohi8 >> 16);
                                        }
                                    }
                                }
                            }
                        }

                        cutlass::arch::fence_view_async_shared();
                        cute::tma_store_fence();
                        epi_synchronize();

                        const int offset_m = tile->offset_m;
                        const int offset_n = tile->offset_n;
                        const void* Cdesc = &tm_c;

                        // One full-tile store after all math WGs publish their packed
                        // row-major output.
                        if (threadIdx.x == 0) {
                            if constexpr (is_grouped_gemm) {
                                Cdesc = tensormap_buf + tile->group_idx * kTmaDescNum + kCdescIdx;
                            }
                            const int store_n = kFuseSilu ? offset_n / 2 : offset_n;
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
                        consumer_wait();
                        consumer_release();
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
