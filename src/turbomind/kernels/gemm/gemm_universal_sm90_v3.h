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
#include "cute/atom/copy_atom.hpp"
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

#include "src/turbomind/kernels/gemm/gmma_bf16_sm90.h"
#include "src/turbomind/kernels/gemm/gmma_fp8_sm90.h"
#include "src/turbomind/kernels/gemm/prepare_moe_tma_descs_sm90_fp8.h"
#include "src/turbomind/kernels/gemm/sm90_utils.h"

namespace turbomind::gemm {

template<class Config_, int Stages_, Order Raster, Striding Mode, bool Silu, class ClusterShape_, int MaxOpN, int EpiStages_>
struct GemmUniversalSm90_v3 {

    static constexpr bool kDebug = false;

    using Arch = Sm90;
    using Tile = typename Config_::Tile;
    using Groups = typename Config_::Groups;
    using RegisterConfig = typename Config_::RegisterConfig;

    static constexpr bool kSupportsFusedSilu = Silu;

    static constexpr int TILE_M = Tile::M;
    static constexpr int TILE_N = Tile::N;
    static constexpr int TILE_K = 128;

    // Fused SiLU pairs OP_N=128 gate/up atoms inside TILE_N=256.
    static_assert(!kSupportsFusedSilu || (TILE_N == 256 && MaxOpN == 128));

    static constexpr int WG_M = Groups::M;
    static constexpr int WG_N = Groups::N;

    static constexpr int WG_TILE_M = TILE_M / WG_M;
    static constexpr int WG_TILE_N = TILE_N / WG_N;
    static_assert(TILE_M % WG_M == 0);
    static_assert(TILE_N % WG_N == 0);
    static_assert(WG_TILE_M % 64 == 0);  // WGMMA atom M

    static constexpr int kSchedWarpGroups = 1;

    static constexpr int WARPGROUPS = WG_M * WG_N;

    static constexpr Order kRasterOrder = Raster;
    static constexpr int   kAlgoFamily  = 1;

    // Each math WG owns one contiguous (M,N) tile.  The WGMMA atom is capped
    // by MaxOpN; any additional N atoms are CuTe rest-N fragments.
    using AtomLayoutMNK = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
    using Traits = GmmaFP8V3Traits<WG_TILE_M, WG_TILE_N, TILE_K, AtomLayoutMNK, MaxOpN>;
    using TiledMma = typename Traits::TiledMma;

    static constexpr int OP_M = Traits::kOpM;
    static constexpr int OP_N = Traits::kOpN;
    static constexpr int OP_K = Traits::kOpK;
    static_assert(Traits::kRestM == 1);
    static_assert(!kSupportsFusedSilu || (OP_M == 64 && OP_N == 128 && Traits::kRestN == 2));

    static constexpr int kMulticastA = ClusterShape_::N;
    static constexpr int kMulticastB = ClusterShape_::M;

    static constexpr int kClusterSize = kMulticastA * kMulticastB;

    static constexpr int Stages = Stages_;

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

    using Cluster = arch::Cluster<ClusterShape_::M, ClusterShape_::N, kRowMajor>;

    static constexpr bool is_grouped_gemm = Mode != Striding::kFlat;

    static constexpr Striding kStridingA = Mode;
    static constexpr Striding kStridingB = is_grouped_gemm ? Striding::kBlocked : Striding::kFlat;
    static constexpr Striding kStridingC = is_grouped_gemm ? Striding::kBlocked : Striding::kFlat;

    // Indexed gather: A/U via cp.async; TMA only B (+ C store descs).
    static constexpr bool kIndexedGather = (Mode == Striding::kIndexed);

    using Scheduler = TileScheduler<Raster, Cluster, true, true, TILE_M, TILE_N, Stages, is_grouped_gemm>;

    static constexpr int kMulticastU = is_grouped_gemm ? 1 : kMulticastA;

    using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
    using PipelineState    = typename MainloopPipeline::PipelineState;
    using PipelineStorage  = typename MainloopPipeline::SharedStorage;
    using ClusterShape     = cute::Shape<cute::Int<kClusterSize>, cute::_1, cute::_1>;

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
    static constexpr int kTmaDescNum = !is_grouped_gemm ? 1 : (kIndexedGather ? 2 : 4);
    static constexpr int kCdescIdx   = kIndexedGather ? 1 : 3;

    // One canonical SW128 K-major layout is shared by the TMA/cp.async
    // producers and CuTe's GMMA descriptor fragments.
    using SmemLayoutA = decltype(cute::tile_to_shape(typename Traits::SmemLayoutAtomA{},
                                                     cute::make_shape(cute::Int<TILE_M>{},
                                                                      cute::Int<TILE_K>{},
                                                                      cute::Int<Stages>{}),
                                                     cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutB = decltype(cute::tile_to_shape(typename Traits::SmemLayoutAtomB{},
                                                     cute::make_shape(cute::Int<TILE_N>{},
                                                                      cute::Int<TILE_K>{},
                                                                      cute::Int<Stages>{}),
                                                     cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutA_2D = decltype(cute::tile_to_shape(typename Traits::SmemLayoutAtomA{},
                                                        cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_K>{}),
                                                        cute::Step<cute::_1, cute::_2>{}));

    static constexpr int kGatherVec      = 16 / (int)sizeof(Ta);
    static constexpr int kGatherThreadsK = TILE_K / kGatherVec;
    static constexpr int kGatherThreadsM = WARPGROUP_SIZE / kGatherThreadsK;
    static constexpr int kGatherSlots    = TILE_M * kGatherThreadsK / WARPGROUP_SIZE;
    using GatherCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>, Ta>;
    using GatherTiledCopy = decltype(cute::make_tiled_copy(
        GatherCopyAtom{},
        cute::Layout<cute::Shape<cute::Int<kGatherThreadsM>, cute::Int<kGatherThreadsK>>,
                     cute::Stride<cute::Int<kGatherThreadsK>, cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1, cute::Int<kGatherVec>>>{}));
    static_assert(kGatherVec * (int)sizeof(Ta) == 16);
    static_assert(kGatherThreadsM * kGatherThreadsK == WARPGROUP_SIZE);
    static_assert(TILE_M % kGatherThreadsM == 0);
    static_assert(TILE_M * kGatherThreadsK % WARPGROUP_SIZE == 0);
    static_assert(cute::size(GatherTiledCopy{}) == WARPGROUP_SIZE);

    // setmaxnreg: each WG ≤ 256, multiples of 8. Config supplies the active budget.
    // Two math WGs pack to 504; one math WG packs to ≤512.
    static constexpr int kProducerRegs = RegisterConfig::Producer;
    static constexpr int kMathRegs = RegisterConfig::Math;
    static_assert(kProducerRegs >= 24 && kProducerRegs % 8 == 0);
    static_assert(kMathRegs >= 24 && kMathRegs % 8 == 0 && kMathRegs <= 256);
    static_assert(WARPGROUPS == 1 || WARPGROUPS == 2);
    static_assert(WARPGROUPS != 2 || kProducerRegs + 2 * kMathRegs == 504);
    static_assert(WARPGROUPS != 1 || kProducerRegs + kMathRegs <= 512);

    // Epilogue ring: every slot is exactly one 128-byte-wide output strip.
    // The tile trades epilogue overlap for mainloop stages under the SMEM cap.
    static constexpr int kEpiStoreBytes    = 128;
    static constexpr int kEpiStorageStages = EpiStages_;
    static constexpr int kEpiStageBytes    = WG_TILE_M * kEpiStoreBytes;
    static_assert(WG_TILE_M == OP_M);

    // ! SMEM addr must be SBO aligned for TMA load/store
    struct SharedStorage {
        __align__(1024) Array<Ta, Stages * TILE_M * TILE_K> A;
        __align__(1024) Array<Tb, Stages * TILE_N * TILE_K> B;
        __align__(128) Tu U[Stages][round_up<int>(kBoxU, 128)];  // at least 128 byte alignment
        __align__(128) Tv V[Stages][2];
        __align__(1024) uint8_t D[WARPGROUPS * kEpiStorageStages * kEpiStageBytes];
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

        using Tc       = std::conditional_t<kFuseSilu, __nv_fp8_e4m3, nv_bfloat16>;
        using ElementC = std::conditional_t<kFuseSilu, cutlass::float_e4m3_t, cutlass::bfloat16_t>;

        static constexpr int kStoreN    = kFuseSilu ? TILE_N / 2 : TILE_N;
        static constexpr int kWgStoreN  = kFuseSilu ? WG_TILE_N / 2 : WG_TILE_N;
        static constexpr int kEpiN      = kEpiStoreBytes / sizeof(Tc);
        static constexpr int kEpiPasses = kWgStoreN / kEpiN;
        static constexpr int kEpiStages = kEpiPasses > 1 ? kEpiStorageStages : 1;

        using SmemLayoutAtomC =
            decltype(gmma_ss_smem_selector<cute::GMMA::Major::K, ElementC, cute::Int<WG_TILE_M>, cute::Int<kEpiN>>());
        using SmemLayoutC =
            decltype(cute::tile_to_shape(SmemLayoutAtomC{},
                                         cute::make_shape(cute::Int<WG_TILE_M>{}, cute::Int<kEpiN>{}),
                                         cute::Step<cute::_1, cute::_2>{}));

        struct LayoutC {
            static constexpr int S0       = WG_TILE_M;
            static constexpr int C0       = kEpiN;
            static constexpr int C1       = 1;
            static constexpr int C1_store = 1;
        };

        static_assert(kWgStoreN % LayoutC::C0 == 0);
        static_assert(kEpiStageBytes == WG_TILE_M * kEpiN * (int)sizeof(Tc));
        static_assert(decltype(cute::size<1>(SmemLayoutAtomC{}))::value == LayoutC::C0);

        static constexpr int kSwizzleC = LayoutC::C0 * (int)sizeof(Tc);
        static_assert(kSwizzleC == kEpiStoreBytes);
    };

    static constexpr int GetSmemSize(bool)
    {
        return sizeof(SharedStorage);
    }

    static constexpr int kSmemSize = GetSmemSize(false);
    static_assert(kSmemSize <= (228 << 10));

    // The epilogue consumes the same single-atom C TV map as the mainloop.
    using EpiTiledMma = TiledMma;

    // Fused SiLU emits an FP8 64x128 C fragment. STSM operates on b16, so its
    // destination uses the native b16 64x64 C map after adjacent N values are packed.
    using EpiPackedOperation = cute::SM90::GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<>;
    using EpiPackedTiledMma  = decltype(cute::make_tiled_mma(
        EpiPackedOperation{}, cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>{}));

    static constexpr int OUTER_N = std::gcd(WG_TILE_N, 128);
    // V-scale predicates span the full WG tile N (not just one MMA atom).
    static constexpr int MMA_SUBTILE_N = WG_TILE_N / OUTER_N;

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
        if constexpr (!is_grouped_gemm) {
            return nullptr;
        }
        int* offsets = reinterpret_cast<int*>(out + num_groups * kTmaDescNum);
        prepare_moe_tma_descs_sm90_fp8<kAlignmentU, Mode><<<num_groups, 32, 0, stream>>>(
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

        const int wg_idx = cutlass::canonical_warp_group_idx();

        if (threadIdx.x == 0) {
            sched.init_dyanmic(storage.sched, kClusterSize * (WARPGROUPS * 4 + 1));
        }

        typename MainloopPipeline::Params pp;
        pp.transaction_bytes = (uint32_t)kTmaTxBytes;
        pp.num_consumers     = (uint32_t)kMathGroupSize;
        // Indexed: leader arrive_and_expect_tx + one cp.async .noinc arrival
        // from every gather thread. TMA: leader + its V cp.async .noinc.
        pp.num_producers     = kIndexedGather ? (1 + WARPGROUP_SIZE) : (1 + 1);
        pp.initializing_warp = 0;

        if (wg_idx == WARPGROUPS) {
            const int warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
            if constexpr (kIndexedGather) {
                pp.role      = MainloopPipeline::ThreadCategory::Producer;
                pp.is_leader = warp_in_wg == 0 && threadIdx.x % WARP_SIZE == 0;
            }
            else {
                pp.role      = warp_in_wg == 0 ? MainloopPipeline::ThreadCategory::Producer :
                                                MainloopPipeline::ThreadCategory::NonParticipant;
                pp.is_leader = warp_in_wg == 0 && threadIdx.x % WARP_SIZE == 0;
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
            const int  warp_in_wg = warp_id % 4;
            const bool cta_0      = cute::block_id_in_cluster().x == 0;

            if constexpr (kIndexedGather) {
                // Full producer WG gather. Scheduler folded onto warp0.
                cutlass::arch::NamedBarrier gather_bar(
                    /*num_threads=*/WARPGROUP_SIZE, cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);

                Cluster cluster(cute::block_id_in_cluster().x);

                const int mc_offset_n = cluster.cta_m() * (TILE_N / kMulticastB);

                auto* smem_B = storage.B.data() + mc_offset_n * TILE_K;
                auto& smem_U = storage.U;
                auto& smem_V = storage.V;

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
                    const Tu* src_u_base;
                    int       m_u;
                    bool      pred_u;
                    const int u_pad = m0 % kAlignmentU;

                    auto gather_thr = GatherTiledCopy{}.get_slice(prod_tid);
                    auto gather_smem = cute::make_tensor(cute::make_smem_ptr(storage.A.data()), SmemLayoutA_2D{});
                    auto gather_dst_part = gather_thr.partition_D(gather_smem);
                    auto gather_identity =
                        cute::make_identity_tensor(cute::Shape<cute::Int<TILE_M>, cute::Int<TILE_K>>{});
                    auto gather_coord = gather_thr.partition_D(gather_identity);
                    static_assert(cute::size<0>(gather_coord) == kGatherVec);
                    static_assert(cute::size<1>(gather_coord) == kGatherSlots);
                    static_assert(cute::size<2>(gather_coord) == 1);
                    const int gather_k =
                        cute::get<1>(gather_coord(cute::make_coord(0, 0), 0, 0));

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
                            gmem_B.Step(bar, &smem_B[pipe * TILE_N * TILE_K], mask_B);
                            uint32_t uint_ptr_V = cast_smem_ptr_to_uint(smem_V[pipe]);
                            CP_ASYNC<CacheOp::kAlways, 4, 0>::apply(uint_ptr_V, gmem_V0, true);
                            CP_ASYNC<CacheOp::kAlways, 4, 0>::apply(uint_ptr_V + sizeof(Tv), gmem_V1, true);
                            ++gmem_V0;
                            ++gmem_V1;
                        }

                        {
                            PRAGMA_UNROLL
                            for (int slot = 0; slot < kGatherSlots; ++slot) {
                                const bool pred = gather_pred[slot] && coord_k + gather_k < K;
                                auto*      dst  = gather_dst[slot] + pipe * TILE_M * TILE_K;
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

                auto  smem_A = storage.A.data() + mc_offset_m * TILE_K;
                auto  smem_B = storage.B.data() + mc_offset_n * TILE_K;
                auto& smem_U = storage.U;
                auto& smem_V = storage.V;

                PipelineState write_state = cutlass::make_producer_start_state<MainloopPipeline>();

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
                                pipeline.producer_acquire(write_state);
                                auto*     bar  = pipeline.producer_get_barrier(write_state);
                                const int pipe = write_state.index();
                                gmem_A.Step(bar, &smem_A[pipe * TILE_M * TILE_K], mask_A);
                                gmem_B.Step(bar, &smem_B[pipe * TILE_N * TILE_K], mask_B);
                                gmem_U.Step(bar, smem_U[pipe] + mc_offset_u, mask_A);
                                uint32_t uint_ptr_V = cast_smem_ptr_to_uint(smem_V[pipe]);
                                CP_ASYNC<CacheOp::kAlways, 4, 0>::apply(uint_ptr_V, gmem_V0, true);
                                CP_ASYNC<CacheOp::kAlways, 4, 0>::apply(uint_ptr_V + sizeof(Tv), gmem_V1, true);
                                ++gmem_V0;
                                ++gmem_V1;
                                cutlass::arch::cpasync_barrier_arrive_noinc(bar);
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

                if (lane_predicate) {
                    pipeline.producer_tail(write_state);
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

            auto& smem_A = storage.A;
            auto& smem_B = storage.B;
            auto& smem_U = storage.U;
            auto& smem_V = storage.V;

            const int wg_idx_m = WG_M > 1 ? wg_idx % WG_M : 0;
            const int wg_idx_n = WG_N > 1 ? wg_idx / WG_M : 0;

            auto sA_full = cute::make_tensor(cute::make_smem_ptr(smem_A.data()), SmemLayoutA{});
            auto sB_full = cute::make_tensor(cute::make_smem_ptr(smem_B.data()), SmemLayoutB{});
            auto sA = cute::local_tile(
                sA_full,
                cute::make_shape(cute::Int<WG_TILE_M>{}, cute::Int<TILE_K>{}, cute::Int<Stages>{}),
                cute::make_coord(wg_idx_m, 0, 0));
            auto sB = cute::local_tile(
                sB_full,
                cute::make_shape(cute::Int<WG_TILE_N>{}, cute::Int<TILE_K>{}, cute::Int<Stages>{}),
                cute::make_coord(wg_idx_n, 0, 0));

            TiledMma tiled_mma;
            auto     thr_mma = tiled_mma.get_thread_slice(threadIdx.x % WARPGROUP_SIZE);
            auto     tCrA    = thr_mma.make_fragment_A(thr_mma.partition_A(sA));
            auto     tCrB    = thr_mma.make_fragment_B(thr_mma.partition_B(sB));

            CUTE_STATIC_ASSERT_V(cute::size<1>(tCrA) == cute::Int<Traits::kRestM>{});
            CUTE_STATIC_ASSERT_V(cute::size<1>(tCrB) == cute::Int<Traits::kRestN>{});
            CUTE_STATIC_ASSERT_V(cute::size<2>(tCrA) == cute::Int<Traits::kKBlocks>{});
            CUTE_STATIC_ASSERT_V(cute::size<2>(tCrB) == cute::Int<Traits::kKBlocks>{});

            cutlass::arch::NamedBarrier barrier(WARPGROUP_SIZE, 2 + wg_idx);  // 0, 1

            PipelineState pipe_state{};
            int           epi_store_count = 0;

            const int warp_id = cutlass::canonical_warp_idx_sync();
            const int lane_id = cutlass::canonical_lane_idx();

            auto consumer_wait = [&] {
                auto token = pipeline.consumer_try_wait(pipe_state);
                pipeline.consumer_wait(pipe_state, token);
            };

            auto sched_state = sched.init_consumer(storage.sched);

            typename Scheduler::Tile* tile;

            sched_state.acquire(tile);

            while (tile->alive) {

                if (tile->is_valid_cta) {
                    auto accum_C = cute::partition_fragment_C(
                        tiled_mma, cute::take<0, 2>(typename Traits::TileShape{}));
                    auto frag_C = cute::make_fragment_like(accum_C(cute::_, cute::Int<0>{}, cute::Int<0>{}));
                    cute::clear(accum_C);

                    CUTE_STATIC_ASSERT_V(cute::size<1>(accum_C) == cute::_1{});
                    CUTE_STATIC_ASSERT_V(cute::size<2>(accum_C) == cute::Int<Traits::kRestN>{});

                    auto pred_V = Fetch_V(tile, wg_idx_n);

                    float scale_V[2];
                    auto  Load_V = [&] {
                        scale_V[0] = smem_V[pipe_state.index()][0];
                        scale_V[1] = smem_V[pipe_state.index()][1];
                    };

                    int offset_U = wg_idx_m * WG_TILE_M + warp_id % 4 * 16 + lane_id / 4;
                    if constexpr (is_grouped_gemm) {
                        offset_U += tile->m0 % kAlignmentU;
                    }
                    float frag_U[2];
                    auto  Load_U = [&] {
                        frag_U[0] = smem_U[pipe_state.index()][offset_U];
                        frag_U[1] = smem_U[pipe_state.index()][offset_U + 8];
                    };

                    auto gmma = [&](auto prefetch_next) {
                        constexpr bool kPrefetchNext = decltype(prefetch_next)::value;
                        const int read       = pipe_state.index();
                        auto      tCrA_stage = tCrA(cute::_, cute::_, cute::_, read);
                        auto      tCrB_stage = tCrB(cute::_, cute::_, cute::_, read);
                        cutlass::ConsumerToken next_token{cutlass::BarrierStatus::WaitAgain};

                        // Preserve the established peak schedule exactly for
                        // every CuTe rest-N atom: issue the complete K=128
                        // batch, commit, wait<0>, then apply U×V scales.
                        cute::for_each(
                            cute::make_seq<cute::size<2>(typename decltype(accum_C)::layout_type{})>{},
                            [&](auto n) {
                                auto rA = tCrA_stage(cute::_, cute::Int<0>{}, cute::Int<0>{});
                                auto rB = tCrB_stage(cute::_, n, cute::Int<0>{});

                                cute::warpgroup_fence_operand(frag_C);
                                tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
                                cute::warpgroup_arrive();
                                cute::for_each(
                                    cute::make_seq<cute::size<2>(typename decltype(tCrA)::layout_type{})>{},
                                    [&](auto k) {
                                        cute::gemm(tiled_mma, rA, rB, frag_C);
                                        tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
                                        if constexpr (decltype(k)::value + 1 < Traits::kKBlocks) {
                                            // DescriptorIterator::operator+ advances only the 32-bit
                                            // address field; the descriptor control word is invariant.
                                            rA.data() = rA.data() + cute::stride<2>(tCrA_stage.layout());
                                            rB.data() = rB.data() + cute::stride<2>(tCrB_stage.layout());
                                        }
                                    });
                                cute::warpgroup_commit_batch();
                                if constexpr (kPrefetchNext && decltype(n)::value + 1 == Traits::kRestN) {
                                    auto next_state = pipe_state;
                                    ++next_state;
                                    next_token = pipeline.consumer_try_wait(next_state);
                                }
                                cute::warpgroup_wait<0>();
                                cute::warpgroup_fence_operand(frag_C);

                                auto        accum    = accum_C(cute::_, cute::Int<0>{}, n);
                                const int   offset_V = n * OP_N;
                                PRAGMA_UNROLL
                                for (int c = 0; c < OP_N; c += 8) {
                                    const float sv = pred_V[(offset_V + c) / OUTER_N] ? scale_V[1] : scale_V[0];
                                    const float s0 = frag_U[0] * sv;
                                    const float s1 = frag_U[1] * sv;
                                    accum(c / 2 + 0) += s0 * frag_C(c / 2 + 0);
                                    accum(c / 2 + 1) += s0 * frag_C(c / 2 + 1);
                                    accum(c / 2 + 2) += s1 * frag_C(c / 2 + 2);
                                    accum(c / 2 + 3) += s1 * frag_C(c / 2 + 3);
                                }
                            });
                        return next_token;
                    };

                    int k_iter = sched.k_iters_;

                    consumer_wait();
                    Load_V();
                    Load_U();
                    auto next_token = gmma(cute::bool_constant<is_grouped_gemm>{});
                    pipeline.consumer_release(pipe_state);
                    ++pipe_state;
                    --k_iter;

                    if constexpr (is_grouped_gemm) {
                        pipeline.consumer_wait(pipe_state, next_token);
                    }
                    else {
                        consumer_wait();
                    }
                    Load_V();
                    Load_U();

                    PRAGMA_NO_UNROLL
                    for (; k_iter > 1; --k_iter) {
                        next_token = gmma(cute::bool_constant<is_grouped_gemm>{});
                        pipeline.consumer_release(pipe_state);
                        ++pipe_state;
                        if constexpr (is_grouped_gemm) {
                            pipeline.consumer_wait(pipe_state, next_token);
                        }
                        else {
                            consumer_wait();
                        }
                        Load_V();
                        Load_U();
                    }

                    gmma(cute::false_type{});

                    const int thread_idx = threadIdx.x % WARPGROUP_SIZE;

                    pipeline.consumer_release(pipe_state);
                    ++pipe_state;

                    auto run_epilogue = [&](auto fused_silu) {
                        constexpr bool kFuseSilu = decltype(fused_silu)::value;
                        using OutputTraits       = Output<kFuseSilu>;
                        using OutputT            = typename OutputTraits::Tc;
                        using ElementC           = typename OutputTraits::ElementC;
                        using LayoutC            = typename OutputTraits::LayoutC;
                        using SmemLayoutC        = typename OutputTraits::SmemLayoutC;

                        // CUTLASS RowMajor epi: STSM_N into K-major swizzle panels.
                        // Fused SiLU: silu(gate)*up → per-row amax/quant (gs=128) → FP8 + W scales.
                        static_assert(OP_N % 16 == 0);
                        static_assert(OP_N % LayoutC::C0 == 0);

                        if constexpr (kFuseSilu) {
                            static_assert(!kFuseSilu || Traits::kRestN == 2);
                            static_assert(!kFuseSilu || OP_N == 128);
                            constexpr float kQmax = 448.f;
                            // CRegisters: every 4 floats = [u0,u0,u1,u1] for 8 N-cols;
                            // each thread owns 2 M-rows (U[0]/U[1]); reduce amax across lane%4.
                            auto          gate     = accum_C(cute::_, cute::Int<0>{}, cute::Int<0>{});
                            auto          up       = accum_C(cute::_, cute::Int<0>{}, cute::Int<1>{});
                            constexpr int kNumRegs = cute::size<0>(typename decltype(gate)::layout_type{});
                            static_assert(!kFuseSilu || kNumRegs == 64);
                            PRAGMA_UNROLL
                            for (int i = 0; i < kNumRegs; ++i) {
                                const float g = gate(i);
                                const float u = up(i);
                                gate(i)       = fdividef(g, 1.f + expf(-g)) * u;
                            }
                            float amax0 = 0.f;
                            float amax1 = 0.f;
                            PRAGMA_UNROLL
                            for (int i = 0; i < kNumRegs; i += 4) {
                                amax0 = fmaxf(amax0, fabsf(gate(i + 0)));
                                amax0 = fmaxf(amax0, fabsf(gate(i + 1)));
                                amax1 = fmaxf(amax1, fabsf(gate(i + 2)));
                                amax1 = fmaxf(amax1, fabsf(gate(i + 3)));
                            }
                            amax0              = fmaxf(amax0, __shfl_xor_sync(0xffffffffu, amax0, 1));
                            amax0              = fmaxf(amax0, __shfl_xor_sync(0xffffffffu, amax0, 2));
                            amax1              = fmaxf(amax1, __shfl_xor_sync(0xffffffffu, amax1, 1));
                            amax1              = fmaxf(amax1, __shfl_xor_sync(0xffffffffu, amax1, 2));
                            amax0              = fmaxf(amax0, 1e-8f);
                            amax1              = fmaxf(amax1, 1e-8f);
                            const float scale0 = amax0 / kQmax;
                            const float scale1 = amax1 / kQmax;
                            const float inv0   = kQmax / amax0;
                            const float inv1   = kQmax / amax1;
                            PRAGMA_UNROLL
                            for (int i = 0; i < kNumRegs; i += 4) {
                                gate(i + 0) *= inv0;
                                gate(i + 1) *= inv0;
                                gate(i + 2) *= inv1;
                                gate(i + 3) *= inv1;
                            }
                            // W is a flat packed buffer (no per-expert TMA rebase). Global row
                            // is m0 + tile-local offset (C TMA is rebased; W is not).
                            if (param_W.ptr && (lane_id % 4) == 0) {
                                const int n_group = tile->offset_n / TILE_N;
                                int row0 = tile->offset_m + wg_idx_m * WG_TILE_M + (warp_id % 4) * 16 + lane_id / 4;
                                int row_end = sched.gemm_shape().x;
                                if constexpr (is_grouped_gemm) {
                                    row0 += tile->m0;
                                    row_end = tile->m1;
                                }
                                Tw*       W   = reinterpret_cast<Tw*>(param_W.ptr);
                                const int ldW = param_W.stride;
                                if (row0 < row_end) {
                                    W[(int64_t)n_group * ldW + row0] = scale0;
                                }
                                if (row0 + 8 < row_end) {
                                    W[(int64_t)n_group * ldW + row0 + 8] = scale1;
                                }
                            }
                        }

                        EpiTiledMma epi_tiled_mma{};
                        using PackedElementC = uint16_t;
                        using CopyElementC   = std::conditional_t<kFuseSilu, PackedElementC, ElementC>;
                        using CopyOperationC = std::conditional_t<
                            kFuseSilu, cute::SM90_U32x2_STSM_N, cute::SM90_U32x4_STSM_N>;
                        using CopyAtomC      = cute::Copy_Atom<CopyOperationC, CopyElementC>;
                        using CopyTiledMma   = std::conditional_t<kFuseSilu, EpiPackedTiledMma, EpiTiledMma>;
                        auto tiled_copy_C    = cute::make_tiled_copy_C(CopyAtomC{}, CopyTiledMma{});
                        auto thr_copy        = tiled_copy_C.get_thread_slice(thread_idx);

                        auto tCr_layout = cute::layout(cute::partition_fragment_C(
                            epi_tiled_mma, cute::make_shape(cute::Int<OP_M>{}, cute::Int<OP_N>{})));

                        cute::for_each(
                            cute::make_seq<cute::size<2>(typename decltype(accum_C)::layout_type{})>{},
                            [&](auto n) {
                            constexpr auto m = cute::Int<0>{};
                            if constexpr (kFuseSilu) {
                                if constexpr (decltype(n)::value != 0) {
                                    return;  // up atom consumed; store fused gate only
                                }
                            }
                            auto C = accum_C(cute::_, m, n);
                            cute::Tensor tCr_f32 =
                                cute::make_tensor(C.data(), tCr_layout);

                            cute::Tensor tCr_out = cute::make_tensor_like<ElementC>(tCr_f32);
                            CUTE_UNROLL
                            for (int i = 0; i < cute::size(tCr_f32); ++i) {
                                tCr_out(i) = ElementC(tCr_f32(i));
                            }

                            auto tCrS = [&]() {
                                if constexpr (kFuseSilu) {
                                    auto packed = cute::recast<PackedElementC>(tCr_out);
                                    auto moved  = cute::make_tensor_like<PackedElementC>(packed);

                                    // Packed FP8 ownership differs from native b16 STSM ownership by
                                    // exchanging t0.bit0 with v2.bit0. MOVM supplies that bit exchange,
                                    // while the two SHFLs permute the remaining lane bits around it:
                                    //   source lane = [A,B,X0,X1,X2], half = C
                                    //   after MOVM  = [B,X0,C,X1,X2], half = A
                                    //   STSM lane   = [B,C,X0,X1,X2], half = A.
                                    const int lane     = thread_idx & 31;
                                    const int pre_src  = ((lane & 0x1c) >> 2) | ((lane & 0x03) << 3);
                                    const int post_src = (lane & 0x19) | ((lane & 0x04) >> 1)
                                                         | ((lane & 0x02) << 1);
                                    CUTE_UNROLL
                                    for (int v2_hi = 0; v2_hi < 8; ++v2_hi) {
                                        CUTE_UNROLL
                                        for (int v1 = 0; v1 < 2; ++v1) {
                                            const int packed_idx = 4 * v2_hi + v1;
                                            const int moved_idx  = 2 * (2 * v2_hi + v1);
                                            uint32_t src = uint32_t(packed(packed_idx))
                                                           | (uint32_t(packed(packed_idx + 2)) << 16);
                                            src = __shfl_sync(0xffffffffu, src, pre_src);
                                            src = transpose_m8n8_b16(src);
                                            src = __shfl_sync(0xffffffffu, src, post_src);
                                            moved(moved_idx)     = PackedElementC(src);
                                            moved(moved_idx + 1) = PackedElementC(src >> 16);
                                        }
                                    }
                                    return thr_copy.retile_S(moved);
                                }
                                else {
                                    return thr_copy.retile_S(tCr_out);
                                }
                            }();

                            constexpr int kStripsPerAtom = OP_N / LayoutC::C0;
                            static_assert(OP_N % LayoutC::C0 == 0);

                            PRAGMA_UNROLL
                            for (int epi_strip = 0; epi_strip < kStripsPerAtom; ++epi_strip) {
                                const int epi_stage = epi_store_count % OutputTraits::kEpiStages;
                                auto* smem_C = reinterpret_cast<OutputT*>(
                                    storage.D + (wg_idx * kEpiStorageStages + epi_stage) * kEpiStageBytes);

                                // The issuer owns the TMA store-group sequence. Wait immediately
                                // before the WG overwrites a ring slot.
                                if (thread_idx == 0) {
                                    cute::tma_store_wait<OutputTraits::kEpiStages - 1>();
                                }
                                barrier.sync();

                                cute::Tensor sC = cute::as_position_independent_swizzle_tensor(cute::make_tensor(
                                    cute::make_smem_ptr(reinterpret_cast<ElementC*>(smem_C)), SmemLayoutC{}));
                                auto tCsC = [&]() {
                                    if constexpr (kFuseSilu) {
                                        return thr_copy.partition_D(cute::recast<PackedElementC>(sC));
                                    }
                                    else {
                                        return thr_copy.partition_D(sC);
                                    }
                                }();

                                if constexpr (kFuseSilu) {
                                    static_assert(kStripsPerAtom == 1);
                                    static_assert(cute::size(tCrS) == 32);
                                    cute::copy(tiled_copy_C, tCrS, tCsC);
                                }
                                else {
                                    // STSM_N exposes 8 values x 4 M-groups x N-strips per thread.
                                    // Recast only the register view; no fragment or pointer array is added.
                                    using StripSrcLayout = cute::Layout<
                                        cute::Shape<cute::Shape<cute::_8,
                                                                cute::Shape<cute::_4, cute::Int<kStripsPerAtom>>>,
                                                    cute::_1,
                                                    cute::_1>,
                                        cute::Stride<cute::Stride<cute::_1, cute::Stride<cute::_8, cute::_32>>,
                                                     cute::_0,
                                                     cute::_0>>;
                                    cute::Tensor strip_src = cute::make_tensor(tCrS.data(), StripSrcLayout{});
                                    cute::Tensor src = strip_src(
                                        cute::make_coord(cute::_, cute::make_coord(cute::_, epi_strip)),
                                        cute::_,
                                        cute::_);
                                    cute::Tensor dst = tCsC(
                                        cute::make_coord(cute::_, cute::make_coord(cute::_, epi_strip)),
                                        cute::_,
                                        cute::_);
                                    cute::copy(tiled_copy_C, src, dst);
                                }

                                cutlass::arch::fence_view_async_shared();
                                cute::tma_store_fence();
                                barrier.sync();

                                if (thread_idx == 0) {
                                    const void* Cdesc = &tm_c;
                                    if constexpr (is_grouped_gemm) {
                                        Cdesc = tensormap_buf + tile->group_idx * kTmaDescNum + kCdescIdx;
                                    }
                                    const int store_n_base = kFuseSilu ? tile->offset_n / 2 : tile->offset_n;
                                    const int store_n = store_n_base + wg_idx_n * OutputTraits::kWgStoreN
                                                      + n * OP_N + epi_strip * LayoutC::C0;
                                    cute::SM90_TMA_STORE::copy(Cdesc,
                                                               smem_C,
                                                               store_n,
                                                               tile->offset_m + wg_idx_m * WG_TILE_M
                                                                   + m * OP_M);
                                    cute::tma_store_arrive();
                                }
                                ++epi_store_count;
                            }
                        });
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
                        pipeline.consumer_release(pipe_state);
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

    __device__ auto Fetch_V(typename Scheduler::Tile* tile, int wg_idx_n)
    {
        constexpr int BLK_SUBTILE_N = 128 / OUTER_N;
        static_assert(MMA_SUBTILE_N - 1 < BLK_SUBTILE_N + 1);  // n1 - 1 + n0 - 1 < 2 * n0

        Array<bool, MMA_SUBTILE_N> pred_V{};
        if constexpr (MMA_SUBTILE_N != 1) {
            int offset = tile->offset_n % 128 + wg_idx_n * WG_TILE_N;
            static_assert(WG_N == 1);
            // Safely skip pred_V_0 when distributing WGs along M
            PRAGMA_UNROLL
            for (int i = 1; i < MMA_SUBTILE_N; ++i) {
                pred_V[i] = (i * OUTER_N + offset) >= 128;
            }
        }

        return pred_V;
    }
};

}  // namespace turbomind::gemm
