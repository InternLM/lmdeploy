// Inspired by
// https://github.com/QwenLM/FlashQLA/blob/60f81453143e724bcaf3fc7921e71e7328f6ebcd/flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py

#pragma once

#include "src/turbomind/kernels/linear_attn/kernel/sm_90/common.h"

#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>

namespace turbomind::linear_attn::delta_rule {
namespace {

template<class T, class StateT, int BlockDv, bool ContextParallel>
struct Sm90FusedGdrFwd {
    static constexpr int kContextParallelGdrBlockDv = 32;

    enum TmaDescIndex : int
    {
        kFusedGdrQDesc = 0,
        kFusedGdrKDesc,
        kFusedGdrVDesc,
        kFusedGdrResolventDesc,
        kFusedGdrOutDesc,
    };

    static constexpr int kFusedGdrDataDescCount = 5;

    static __device__ __forceinline__ void FusedGdrFenceProxyAsyncShared()
    {
        // Release generic-proxy shared writes to WGMMA's async proxy. This is not
        // a thread rendezvous or a WGMMA completion wait.
        cutlass::arch::fence_view_async_shared();
    }

    static __device__ __forceinline__ __nv_bfloat16 CastFromFloat(float value)
    {
        return __float2bfloat16(value);
    }

    template<class Element>
    static CUTE_HOST_DEVICE constexpr auto GmmaSquareKLayout()
    {
        return cute::tile_to_shape(cute::SM90::GMMA::Layout_K_SW128_Atom<Element>{},
                                   cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<kChunkSize>{}));
    }

    static constexpr int kHStateReadThreads = 3 * kFusedGdrRoleThreads;
    static constexpr int kGateVdReadThreads = 3 * kFusedGdrRoleThreads;

    static __device__ __forceinline__ int CeilDivDevice(int value, int divisor)
    {
        return (value + divisor - 1) / divisor;
    }

    // Forward-only named barriers. CUTLASS maps user IDs 0-7 to
    // SM90 hardware barrier IDs 8-15.
    static constexpr int kBarrierStateUpdate    = 1;
    static constexpr int kBarrierValueU         = 2;
    static constexpr int kBarrierVdReadDone     = 3;
    static constexpr int kBarrierOutputAg       = 4;
    static constexpr int kBarrierOutputLocal    = 5;
    static constexpr int kBarrierHStateReadDone = 6;

    static_assert(kBarrierHStateReadDone + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount
                  < cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers);

    template<int BarrierId>
    static __device__ __forceinline__ void MmaSyncNamed()
    {
        cutlass::arch::NamedBarrier::sync(kFusedGdrRoleThreads, BarrierId);
    }

    static __device__ __forceinline__ void StateWaitForHReaders()
    {
        cutlass::arch::NamedBarrier::sync(kHStateReadThreads, kBarrierHStateReadDone);
    }

    static __device__ __forceinline__ void HReaderArrive()
    {
        cutlass::arch::NamedBarrier::arrive(kHStateReadThreads, kBarrierHStateReadDone);
    }

    static __device__ __forceinline__ void ValueWaitForGateVdReaders()
    {
        cutlass::arch::NamedBarrier::sync(kGateVdReadThreads, kBarrierVdReadDone);
    }

    static __device__ __forceinline__ void StateGateReadArrive()
    {
        cutlass::arch::NamedBarrier::arrive(kGateVdReadThreads, kBarrierVdReadDone);
    }

    static __device__ __forceinline__ void OutputVdReadArrive()
    {
        cutlass::arch::NamedBarrier::arrive(kGateVdReadThreads, kBarrierVdReadDone);
    }

    static CUTE_HOST_DEVICE constexpr auto SwizzledA64Layout()
    {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::_64, cute::_64>, cute::Stride<cute::_64, cute::_1>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto SwizzledV128RowLayout()
    {
        return cute::composition(
            cute::Swizzle<3, 3, 3>{},
            cute::Layout<cute::Shape<cute::Int<kChunkSize>, cute::Shape<cute::Int<kFusedGdrBlockDv>, cute::_2>>,
                         cute::Stride<cute::Int<kFusedGdrBlockDv>,
                                      cute::Stride<cute::_1, cute::Int<kFusedGdrBlockDv * kChunkSize>>>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto SwizzledV32RowLayout()
    {
        return cute::composition(cute::Swizzle<2, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::_64, cute::_32>, cute::Stride<cute::_32, cute::_1>>{});
    }

    template<int LayoutBlockDv>
    static CUTE_HOST_DEVICE constexpr auto SwizzledVRowLayout()
    {
        static_assert(LayoutBlockDv == kContextParallelGdrBlockDv || LayoutBlockDv == kFusedGdrBlockDv
                      || LayoutBlockDv == kWideGdrBlockDv);
        if constexpr (LayoutBlockDv == kContextParallelGdrBlockDv) {
            return SwizzledV32RowLayout();
        }
        else if constexpr (LayoutBlockDv == kFusedGdrBlockDv) {
            return SwizzledA64Layout();
        }
        else {
            return SwizzledV128RowLayout();
        }
    }

    template<class TiledMma, class AFragment, class TB, class BLayout, class Accumulator>
    static __device__ __forceinline__ void GmmaRs(TiledMma&                        tiled_mma,
                                                  uint32_t                         thread_idx,
                                                  AFragment const&                 tCrA,
                                                  cute::Tensor<TB, BLayout> const& sB,
                                                  Accumulator&                     tCrC,
                                                  cute::SM90::GMMA::ScaleOut       scale)
    {
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        auto tCsB    = thr_mma.partition_B(sB);
        auto tCrB    = thr_mma.make_fragment_B(tCsB);

        constexpr int K_BLOCK_MAX = cute::size<2>(AFragment{});
        static_assert(K_BLOCK_MAX == cute::size<2>(decltype(tCrB){}));

        FusedGdrFenceProxyAsyncShared();
        cute::warpgroup_fence_operand(tCrC);
        cute::warpgroup_arrive();
        tiled_mma.accumulate_ = scale;
#pragma unroll
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), tCrC);
            tiled_mma.accumulate_ = cute::SM90::GMMA::ScaleOut::One;
        }
        cute::warpgroup_commit_batch();
        cute::warpgroup_wait<0>();
        cute::warpgroup_fence_operand(tCrC);
    }

    static constexpr float kHeadScale = 0.08838834764831845f;

    static constexpr int kFusedGdrGateRowsPerWarp = 8;
    static constexpr int kFusedGdrGateWriterThreads =
        (kFusedGdrRoleThreads / kCudaWarpThreads) * kFusedGdrGateRowsPerWarp;
    static constexpr int kFusedGdrGatePasses        = kChunkSize / kFusedGdrGateWriterThreads;
    static constexpr int kFusedGdrStateRegisters    = 160;
    static constexpr int kFusedGdrValueRegisters    = 168;
    static constexpr int kFusedGdrOutputRegisters   = 160;
    static constexpr int kFusedGdrProducerRegisters = 24;
    static constexpr int kFusedGdrOutputStoreStages = 2;

    static_assert(kFusedGdrGateRowsPerWarp <= kCudaWarpThreads);
    static_assert(kChunkSize % kFusedGdrGateWriterThreads == 0);
    static_assert(kFusedGdrGateWriterThreads * kFusedGdrGatePasses == kChunkSize);
    static_assert(kFusedGdrRoleThreads
                      * (kFusedGdrStateRegisters + kFusedGdrValueRegisters + kFusedGdrOutputRegisters
                         + kFusedGdrProducerRegisters)
                  <= 65536);

    static constexpr const char* kUnsupportedMessage =
        "fused GDR forward supports only the SM90 bf16 chunked target shape "
        "(int32 q_offsets, bool finished mask, head_dim=128, chunk_size=64, Hv % Hq == 0)";

    // Active SM90 fused-forward actor/barrier contract:
    // - Producer WG3 warp 0 loads Q/K, warp 1 loads V/beta, warp 2 loads resolvent/g,
    //   and warp 3 stores output. Consumer WG0 owns state, WG1 value/update, and WG2 output.
    // - stage_ready_mbar[2] has count 96 plus transaction bytes. Warp 0 releases 32
    //   generic arrivals after issuing Q/K TMA. Warps 1 and 2 issue V/resolvent TMA plus
    //   per-head beta/g cp.async and attach 32 deferred arrivals per warp to cp.async
    //   completion. TMA completion releases the leaders' expected matrix bytes. Consumer
    //   WGs 0, 1, and 2 acquire the complete matrix and gate payload; the phase advances
    //   on each reuse of a double-buffer stage.
    // - stage_free_bar[2] has count 384. WGs 0, 1, and 2 each release 128 arrivals after
    //   their last stage read. Each producer warp acquires stage reuse with the complementary
    //   phase, so generation 0 is initially free without artificial arrivals; the phase
    //   advances after every 384 consumer releases.
    // - The single-slot handoffs below have count 128 and phase (chunk & 1): WG0 releases
    //   h_shared through state_snapshot_bar to WGs 1/2; WG1 releases g/g_exp/g_rev_exp through
    //   gate_ready_bar to WGs 0/2, VD through vd_ready_bar to WG2, and VN through
    //   update_ready_bar to WG0; WG2 releases packed AG through ag_ready_bar and WG1 acquires
    //   it before its second GMMA.
    // - Each o_shared slot has a count-128 out_ready_bar and phase ((chunk / 2) & 1).
    //   WG2 publishes a complete STSM tile and the store leader acquires that slot.
    // - Named barrier 6 has count 384: WG0 waits until WGs 1/2 have drained their
    //   h_shared WGMMA reads. Named barrier 3 has count 384: WG1 waits until WG0 has
    //   consumed the single-slot gate vectors and WG2 has drained its vd_shared WGMMA
    //   read.
    // - Named barriers 1, 2, 4, and 5 remain WG-local thread rendezvous for state phases,
    //   V/W/VD/VN handoffs, AG publication, and output STSM publication, respectively.
    //   Async WGMMA completion remains separately owned by the warpgroup wait operations.
    // - out_free_bar[2] has count 1. The store leader keeps at most one committed TMA
    //   store group in flight; after its acquire completes the older group, it releases
    //   that slot for WG2. Complementary parity makes both slots initially free.
    // The output-store leader owns the async-proxy fence, store commits, steady wait<1>,
    // and final wait<0> tail drain.
    template<class StorageT, int StorageBlockDv>
    struct SharedStorageFor {
        // Fused arena order for the BlockDv=128 path: H, K, Q, V/W, A/AG, O, VD, VN.
        union {
            struct {
                alignas(1024) StorageT h_shared[StorageBlockDv][kHeadDim];
                alignas(1024) StorageT k_stage[2][kChunkSize][kHeadDim];
                alignas(1024) StorageT q_stage[2][kChunkSize][kHeadDim];
            };
        };
        alignas(1024) StorageT v_stage[2][StorageBlockDv][kChunkSize];
        alignas(1024) StorageT a_stage[2][kChunkSize][kChunkSize];
        alignas(1024) StorageT o_shared[kFusedGdrOutputStoreStages][StorageBlockDv][kChunkSize];
        alignas(1024) StorageT vd_shared[StorageBlockDv][kChunkSize];
        alignas(1024) StorageT vn_shared[StorageBlockDv][kChunkSize];
        alignas(1024) float gate_stage[2][2][kChunkSize];
        float          g[kChunkSize];
        float          g_exp[kChunkSize];
        float          g_rev_exp[kChunkSize];
        cute::uint64_t stage_ready_mbar[2];
        cute::uint64_t stage_free_bar[2];
        cute::uint64_t out_ready_bar[kFusedGdrOutputStoreStages];
        cute::uint64_t out_free_bar[kFusedGdrOutputStoreStages];
        cute::uint64_t state_snapshot_bar;
        cute::uint64_t gate_ready_bar;
        cute::uint64_t ag_ready_bar;
        cute::uint64_t vd_ready_bar;
        cute::uint64_t update_ready_bar;
    };

    template<class StorageT>
    struct SharedStorageFor<StorageT, kFusedGdrBlockDv> {
        // BlockDv=64 arena for the high-through direct fused kernel:
        // K, Q, A/AG, H, V/W, O, VD, VN.
        union {
            struct {
                alignas(1024) StorageT k_stage[2][kChunkSize][kHeadDim];
                alignas(1024) StorageT q_stage[2][kChunkSize][kHeadDim];
                alignas(1024) StorageT a_stage[2][kChunkSize][kChunkSize];
                alignas(1024) StorageT h_shared[kFusedGdrBlockDv][kHeadDim];
            };
        };
        alignas(1024) StorageT v_stage[2][kFusedGdrBlockDv][kChunkSize];
        alignas(1024) StorageT o_shared[kFusedGdrOutputStoreStages][kFusedGdrBlockDv][kChunkSize];
        alignas(1024) StorageT vd_shared[kFusedGdrBlockDv][kChunkSize];
        alignas(1024) StorageT vn_shared[kFusedGdrBlockDv][kChunkSize];
        alignas(1024) float gate_stage[2][2][kChunkSize];
        float          g[kChunkSize];
        float          g_exp[kChunkSize];
        float          g_rev_exp[kChunkSize];
        cute::uint64_t stage_ready_mbar[2];
        cute::uint64_t stage_free_bar[2];
        cute::uint64_t out_ready_bar[kFusedGdrOutputStoreStages];
        cute::uint64_t out_free_bar[kFusedGdrOutputStoreStages];
        cute::uint64_t state_snapshot_bar;
        cute::uint64_t gate_ready_bar;
        cute::uint64_t ag_ready_bar;
        cute::uint64_t vd_ready_bar;
        cute::uint64_t update_ready_bar;
    };

    template<class StorageT, int StorageBlockDv>
    static constexpr size_t SharedBytesFor()
    {
        static_assert(StorageBlockDv == kContextParallelGdrBlockDv || StorageBlockDv == kFusedGdrBlockDv
                      || StorageBlockDv == kWideGdrBlockDv);
        return sizeof(SharedStorageFor<StorageT, StorageBlockDv>);
    }

    static_assert(SharedBytesFor<__nv_bfloat16, kFusedGdrBlockDv>() <= kFusedGdrMaxDynamicSharedBytes);
    static_assert(SharedBytesFor<__nv_bfloat16, kContextParallelGdrBlockDv>() <= kFusedGdrMaxDynamicSharedBytes);
    static_assert(SharedBytesFor<__nv_bfloat16, kWideGdrBlockDv>() <= kFusedGdrMaxDynamicSharedBytes);

    template<class StorageT, int StorageBlockDv>
    static constexpr size_t VStageOffset()
    {
        using Storage = SharedStorageFor<StorageT, StorageBlockDv>;
        return offsetof(Storage, v_stage);
    }

    template<class StorageT, int StorageBlockDv>
    static constexpr size_t HSharedOffset()
    {
        using Storage = SharedStorageFor<StorageT, StorageBlockDv>;
        return offsetof(Storage, h_shared);
    }

    template<class StorageT, int StorageBlockDv>
    static constexpr size_t KStageOffset()
    {
        using Storage = SharedStorageFor<StorageT, StorageBlockDv>;
        return offsetof(Storage, k_stage);
    }

    template<class StorageT, int StorageBlockDv>
    static constexpr size_t QStageOffset()
    {
        using Storage = SharedStorageFor<StorageT, StorageBlockDv>;
        return offsetof(Storage, q_stage);
    }

    template<class StorageT, int StorageBlockDv>
    static constexpr size_t AStageOffset()
    {
        using Storage = SharedStorageFor<StorageT, StorageBlockDv>;
        return offsetof(Storage, a_stage);
    }

    template<class StorageT, int StorageBlockDv>
    static constexpr size_t OSharedOffset()
    {
        using Storage = SharedStorageFor<StorageT, StorageBlockDv>;
        return offsetof(Storage, o_shared);
    }

    template<class StorageT, int StorageBlockDv>
    static constexpr size_t OSharedStageStride()
    {
        return StorageBlockDv * kChunkSize * sizeof(StorageT);
    }

    template<class StorageT, int StorageBlockDv>
    static constexpr size_t VdSharedOffset()
    {
        using Storage = SharedStorageFor<StorageT, StorageBlockDv>;
        return offsetof(Storage, vd_shared);
    }

    template<class StorageT, int StorageBlockDv>
    static constexpr size_t VnSharedOffset()
    {
        using Storage = SharedStorageFor<StorageT, StorageBlockDv>;
        return offsetof(Storage, vn_shared);
    }

    static_assert(VStageOffset<__nv_bfloat16, kFusedGdrBlockDv>() % alignof(__nv_bfloat162) == 0);
    static_assert(VdSharedOffset<__nv_bfloat16, kFusedGdrBlockDv>() % alignof(__nv_bfloat162) == 0);
    static_assert(KStageOffset<__nv_bfloat16, kFusedGdrBlockDv>() == 0);
    static_assert(QStageOffset<__nv_bfloat16, kFusedGdrBlockDv>() == 32768);
    static_assert(AStageOffset<__nv_bfloat16, kFusedGdrBlockDv>() == 65536);
    static_assert(HSharedOffset<__nv_bfloat16, kFusedGdrBlockDv>() == 81920);
    static_assert(VStageOffset<__nv_bfloat16, kFusedGdrBlockDv>() == 98304);
    static_assert(OSharedOffset<__nv_bfloat16, kFusedGdrBlockDv>() == 114688);
    static_assert(OSharedStageStride<__nv_bfloat16, kFusedGdrBlockDv>() == 8192);
    static_assert(VdSharedOffset<__nv_bfloat16, kFusedGdrBlockDv>() == 131072);
    static_assert(VnSharedOffset<__nv_bfloat16, kFusedGdrBlockDv>() == 139264);
    static_assert(HSharedOffset<__nv_bfloat16, kWideGdrBlockDv>() == 0);
    static_assert(KStageOffset<__nv_bfloat16, kWideGdrBlockDv>() == 32768);
    static_assert(QStageOffset<__nv_bfloat16, kWideGdrBlockDv>() == 65536);
    static_assert(VStageOffset<__nv_bfloat16, kWideGdrBlockDv>() == 98304);
    static_assert(AStageOffset<__nv_bfloat16, kWideGdrBlockDv>() == 131072);
    static_assert(OSharedOffset<__nv_bfloat16, kWideGdrBlockDv>() == 147456);
    static_assert(OSharedStageStride<__nv_bfloat16, kWideGdrBlockDv>() == 16384);
    static_assert(VdSharedOffset<__nv_bfloat16, kWideGdrBlockDv>() == 180224);
    static_assert(VnSharedOffset<__nv_bfloat16, kWideGdrBlockDv>() == 196608);
    static_assert(OSharedStageStride<__nv_bfloat16, kContextParallelGdrBlockDv>() == 4096);
    static_assert(OSharedStageStride<__nv_bfloat16, kContextParallelGdrBlockDv>() % 1024 == 0);

    static inline bool CanUseFusedGdrFwd(const Problem& problem)
    {
        return problem.arch == 900 && problem.input_dtype == kBfloat16 && problem.batch == problem.sequence_num
               && problem.hv % problem.hq == 0 && problem.head_dim == kHeadDim && problem.chunk_size == kChunkSize;
    }

    static __device__ __forceinline__ void AcquireAndPrefetchDataTmaDescriptors(const CUtensorMap* desc, int tid)
    {
        if (tid < 32) {
            for (int idx = tid; idx < kFusedGdrDataDescCount; idx += 32) {
                cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(&desc[idx]));
                cute::prefetch_tma_descriptor(&desc[idx]);
            }
        }
    }

    static __device__ __forceinline__ void AcquireAndPrefetchStateTmaDescriptor(const CUtensorMap* desc, int tid)
    {
        if (tid == 0) {
            cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(desc));
            cute::prefetch_tma_descriptor(desc);
        }
    }

    template<int ChunkSize>
    static __device__ __forceinline__ int ResolventTmaCoord(int value_head)
    {
        static_assert(ChunkSize == kChunkSize);
        return value_head * ChunkSize;
    }

    static __device__ __forceinline__ int ResolventTmaCoord(int value_head)
    {
        return ResolventTmaCoord<kChunkSize>(value_head);
    }

    template<class Element, class Fragment, class SmemTensor, class ThrMma>
    static __device__ __forceinline__ void
    StorePackedBf16Stsm(Fragment const& fragment, SmemTensor const& smem_tensor, ThrMma const& thr_mma, int role_tid)
    {
        static_assert(std::is_same_v<Element, cute::bfloat16_t>);

        auto s_pack            = cute::as_position_independent_swizzle_tensor(smem_tensor);
        auto smem_tiled_copy_C = cute::make_tiled_copy_C(cute::Copy_Atom<cute::SM90_U32x4_STSM_N, Element>{}, thr_mma);
        auto smem_thr_copy_C   = smem_tiled_copy_C.get_thread_slice(role_tid);
        auto tCsPack           = smem_thr_copy_C.partition_D(s_pack);
        auto tCrPackView       = smem_thr_copy_C.retile_S(fragment);
        cute::copy(smem_tiled_copy_C, tCrPackView, tCsPack);
    }

    struct GmmaCThreadProjection {
        int row_base;
        int column_base;
    };

    template<class TiledMma>
    using GmmaCLayout = typename TiledMma::AtomLayoutC_TV;

    template<class TiledMma>
    using GmmaCThreadLayout = decltype(cute::make_layout(cute::get<0>(cute::shape(GmmaCLayout<TiledMma>{})),
                                                         cute::get<0>(cute::stride(GmmaCLayout<TiledMma>{}))));

    template<class TiledMma>
    static CUTE_HOST_DEVICE constexpr auto GmmaCIdentityTensor()
    {
        using AtomShape = typename TiledMma::AtomShape_MNK;
        using CLayout   = GmmaCLayout<TiledMma>;

        static_assert(decltype(cute::size<0>(CLayout{}))::value == kFusedGdrRoleThreads);
        static_assert(decltype(cute::size(CLayout{}))::value
                      == decltype(cute::size<0>(AtomShape{}))::value* decltype(cute::size<1>(AtomShape{}))::value);

        return cute::make_identity_tensor(cute::make_shape(cute::get<0>(AtomShape{}), cute::get<1>(AtomShape{})));
    }

    template<class TiledMma>
    static __device__ __forceinline__ GmmaCThreadProjection MakeGmmaCThreadProjection(int role_tid)
    {
        using CLayout = GmmaCLayout<TiledMma>;

        auto c_tv = GmmaCIdentityTensor<TiledMma>().compose(CLayout{});
        auto base = c_tv(role_tid, cute::make_coord(cute::Int<0>{}, cute::Int<0>{}, cute::Int<0>{}));
        return {static_cast<int>(cute::get<0>(base)), static_cast<int>(cute::get<1>(base))};
    }

    template<class TiledMma, class Pair, class RowSelect, class Stripe>
    static CUTE_HOST_DEVICE constexpr auto GmmaCStaticOffset(Pair pair, RowSelect row_select, Stripe stripe)
    {
        using CLayout = GmmaCLayout<TiledMma>;

        auto c_tv = GmmaCIdentityTensor<TiledMma>().compose(CLayout{});
        return c_tv(cute::Int<0>{}, cute::make_coord(pair, row_select, stripe));
    }

    template<class Pair, class RowSelect, class Stripe>
    static CUTE_HOST_DEVICE constexpr auto GmmaCFragmentCoord(Pair pair, RowSelect row_select, Stripe stripe)
    {
        return cute::make_coord(cute::make_coord(pair, row_select, stripe), cute::Int<0>{}, cute::Int<0>{});
    }

    template<class TiledMma, class RowMap, class ColMap, class Body>
    static __device__ __forceinline__ void
    ForEachProjectedGmmaC(GmmaCThreadProjection projection, RowMap row_map, ColMap col_map, Body body)
    {
        using CLayout = GmmaCLayout<TiledMma>;

        static_assert(decltype(cute::size<1, 0>(CLayout{}))::value == 2);
        static_assert(decltype(cute::size<1, 1>(CLayout{}))::value == 2);
        constexpr int kStripes = decltype(cute::size<1, 2>(CLayout{}))::value;

        auto rows = cute::transform(cute::make_seq<2>{}, [&](auto row_select) {
            auto offset = GmmaCStaticOffset<TiledMma>(cute::Int<0>{}, row_select, cute::Int<0>{});
            return row_map(projection.row_base + static_cast<int>(cute::get<0>(offset)));
        });

        cute::for_each(cute::make_seq<kStripes>{}, [&](auto stripe) {
            auto columns = cute::transform(cute::make_seq<2>{}, [&](auto pair) {
                auto offset = GmmaCStaticOffset<TiledMma>(pair, cute::Int<0>{}, stripe);
                return col_map(projection.column_base + static_cast<int>(cute::get<1>(offset)));
            });

            cute::for_each(cute::make_seq<2>{}, [&](auto row_select) {
                cute::for_each(cute::make_seq<2>{}, [&](auto pair) {
                    auto offset = GmmaCStaticOffset<TiledMma>(pair, row_select, stripe);
                    body(GmmaCFragmentCoord(pair, row_select, stripe),
                         projection.row_base + static_cast<int>(cute::get<0>(offset)),
                         projection.column_base + static_cast<int>(cute::get<1>(offset)),
                         cute::get<row_select>(rows),
                         cute::get<pair>(columns));
                });
            });
        });
    }

    template<class TiledMma, class RowMap, class Body>
    static __device__ __forceinline__ void
    ForEachProjectedGmmaC(GmmaCThreadProjection projection, RowMap row_map, Body body)
    {
        using CLayout = GmmaCLayout<TiledMma>;

        static_assert(decltype(cute::size<1, 0>(CLayout{}))::value == 2);
        static_assert(decltype(cute::size<1, 1>(CLayout{}))::value == 2);
        constexpr int kStripes = decltype(cute::size<1, 2>(CLayout{}))::value;

        auto rows = cute::transform(cute::make_seq<2>{}, [&](auto row_select) {
            auto offset = GmmaCStaticOffset<TiledMma>(cute::Int<0>{}, row_select, cute::Int<0>{});
            return row_map(projection.row_base + static_cast<int>(cute::get<0>(offset)));
        });

        cute::for_each(cute::make_seq<kStripes>{}, [&](auto stripe) {
            cute::for_each(cute::make_seq<2>{}, [&](auto row_select) {
                cute::for_each(cute::make_seq<2>{}, [&](auto pair) {
                    auto offset = GmmaCStaticOffset<TiledMma>(pair, row_select, stripe);
                    body(GmmaCFragmentCoord(pair, row_select, stripe),
                         projection.row_base + static_cast<int>(cute::get<0>(offset)),
                         projection.column_base + static_cast<int>(cute::get<1>(offset)),
                         cute::get<row_select>(rows));
                });
            });
        });
    }

    template<class Element, class SquareMma, class OutputRsMma, class SquareFragment, class RsFragment>
    static __device__ __forceinline__ void PackSquareCAsRsA(SquareFragment const& tCrP, RsFragment& tCrPrs)
    {
        using CLayout = GmmaCLayout<SquareMma>;
        using ALayout = typename OutputRsMma::AtomLayoutA_TV;

        static_assert(std::is_same_v<T, __nv_bfloat16>);
        static_assert(std::is_same_v<Element, cute::bfloat16_t>);
        static_assert(std::is_same_v<CLayout, cute::SM90::GMMA::CLayout_64x64>);
        static_assert(std::is_same_v<ALayout, cute::SM90::GMMA::ALayout_64x16>);
        static_assert(decltype(cute::size<0, 0>(SquareFragment{}))::value == 2);
        static_assert(decltype(cute::size<0, 1>(SquareFragment{}))::value == 2);
        static_assert(decltype(cute::size<0, 2>(SquareFragment{}))::value == 8);
        static_assert(decltype(cute::size<0, 0>(RsFragment{}))::value == 2);
        static_assert(decltype(cute::size<0, 1>(RsFragment{}))::value == 2);
        static_assert(decltype(cute::size<0, 2>(RsFragment{}))::value == 2);
        static_assert(decltype(cute::size<2>(RsFragment{}))::value == 4);

        cute::for_each(cute::make_seq<8>{}, [&](auto stripe) {
            constexpr int kBlock = decltype(stripe)::value / 2;
            constexpr int innerK = decltype(stripe)::value % 2;
            cute::for_each(cute::make_seq<2>{}, [&](auto row_select) {
                cute::for_each(cute::make_seq<2>{}, [&](auto pair) {
                    auto c_coord = GmmaCFragmentCoord(pair, row_select, stripe);
                    auto a_coord = cute::make_coord(
                        cute::make_coord(pair, row_select, cute::Int<innerK>{}), cute::Int<0>{}, cute::Int<kBlock>{});
                    tCrPrs(a_coord) = Element(CastFromFloat(static_cast<float>(tCrP(c_coord))));
                });
            });
        });
    }

    static constexpr int kThreads   = kFusedGdrThreads;
    static constexpr int kMinBlocks = 1;

    using SharedStorage = SharedStorageFor<T, BlockDv>;

    static constexpr size_t SharedBytes()
    {
        return SharedBytesFor<T, BlockDv>();
    }

    static __device__ __forceinline__ void IssueGateStageCpAsync(float* __restrict__ gate_stage,
                                                                 const float* __restrict__ gate,
                                                                 int     lane,
                                                                 int     valid,
                                                                 int     token0,
                                                                 int64_t gate_sequence_offset,
                                                                 int64_t gate_stride)
    {
#pragma unroll 1
        for (int row = lane; row < kChunkSize; row += kCudaWarpThreads) {
            const bool    in_bounds  = row < valid;
            const int     source_row = in_bounds ? row : valid - 1;
            const int64_t source_offset =
                gate_sequence_offset + static_cast<int64_t>(token0 + source_row) * gate_stride;
            cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<float>::copy(gate[source_offset], gate_stage[row], in_bounds);
        }
    }

    static __device__ __forceinline__ void Run(const CUtensorMap* __restrict__ tma_desc_workspace,
                                               const float* __restrict__ g_cumsum,
                                               const float* __restrict__ beta,
                                               int64_t gate_batch_stride,
                                               int64_t gate_stride,
                                               int64_t beta_batch_stride,
                                               int64_t beta_stride,
                                               int     token_num,
                                               const int32_t* __restrict__ q_offsets,
                                               const bool* __restrict__ finished,
                                               const int32_t* __restrict__ data_q_offsets,
                                               const int32_t* __restrict__ cp_source_indices,
                                               const int64_t* __restrict__ cp_state_ptrs,
                                               const int64_t* __restrict__ state_ptrs,
                                               int64_t        state_layer_offset,
                                               int            data_sequence_num,
                                               int            hq,
                                               int            hv,
                                               int            num_head_groups,
                                               int            heads_per_block,
                                               unsigned char* smem_raw)
    {
        static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv
                      || BlockDv == kWideGdrBlockDv);
        static_assert(kFusedGdrValidStateT<StateT>, "fused chunk GDR StateT must be float or bfloat16");
        auto& smem       = *reinterpret_cast<SharedStorage*>(smem_raw);
        using MmaElement = typename FusedGdrMmaTraits<T>::Element;

        const int tid         = static_cast<int>(threadIdx.x);
        const int wg_idx      = cutlass::canonical_warp_group_idx();
        const int role_tid    = tid % kFusedGdrRoleThreads;
        const int batch_id    = static_cast<int>(blockIdx.x);
        const int value_head  = static_cast<int>(blockIdx.y);
        const int dv_tile     = static_cast<int>(blockIdx.z);
        const int dv0         = dv_tile * BlockDv;
        const int qk_head     = value_head / (hv / hq);
        const int segment_id  = batch_id;
        int       sequence_id = batch_id;
        if constexpr (ContextParallel) {
            sequence_id = cp_source_indices[segment_id];
            if (sequence_id < 0 || sequence_id >= data_sequence_num) {
                return;
            }
        }
        const int seq_start = q_offsets[batch_id];
        const int seq_end   = q_offsets[batch_id + 1];
        const int seq_len   = seq_end - seq_start;
        if (seq_len <= 0) {
            return;
        }
        int  sequence_begin    = seq_start;
        int  raw_sequence_end  = seq_end;
        bool sequence_terminal = false;
        if constexpr (ContextParallel) {
            sequence_begin    = data_q_offsets[sequence_id];
            raw_sequence_end  = data_q_offsets[sequence_id + 1];
            sequence_terminal = seq_end == raw_sequence_end;
            if (seq_start < sequence_begin || seq_end > raw_sequence_end) {
                return;
            }
        }
        const int     token_base                = ContextParallel ? seq_start - sequence_begin : 0;
        const int     gate_physical_batch       = sequence_begin / token_num;
        const int     gate_local_sequence_begin = sequence_begin - gate_physical_batch * token_num;
        const int64_t gate_sequence_offset      = static_cast<int64_t>(gate_physical_batch) * gate_batch_stride
                                             + static_cast<int64_t>(gate_local_sequence_begin) * gate_stride
                                             + value_head;
        const int64_t beta_sequence_offset = static_cast<int64_t>(gate_physical_batch) * beta_batch_stride
                                             + static_cast<int64_t>(gate_local_sequence_begin) * beta_stride
                                             + value_head;
        const int          qk_tma_head_coord   = qk_head;
        constexpr int      qk_tma_batch_coord  = 0;
        const int          chunks              = CeilDivDevice(seq_len, kChunkSize);
        const int          descriptor_sequence = ContextParallel ? sequence_id : batch_id;
        const CUtensorMap* data_desc           = tma_desc_workspace + descriptor_sequence * kFusedGdrDataDescCount;
        AcquireAndPrefetchDataTmaDescriptors(data_desc, tid);
        const CUtensorMap* q_desc         = &data_desc[kFusedGdrQDesc];
        const CUtensorMap* k_desc         = &data_desc[kFusedGdrKDesc];
        const CUtensorMap* v_desc         = &data_desc[kFusedGdrVDesc];
        const CUtensorMap* resolvent_desc = &data_desc[kFusedGdrResolventDesc];
        const CUtensorMap* out_desc       = &data_desc[kFusedGdrOutDesc];
        // Thread 0 initializes the mbarriers before any producer or consumer role uses them.
        if (tid == 0) {
            cute::initialize_barrier(smem.state_snapshot_bar, kFusedGdrRoleThreads);
            cute::initialize_barrier(smem.gate_ready_bar, kFusedGdrRoleThreads);
            cute::initialize_barrier(smem.ag_ready_bar, kFusedGdrRoleThreads);
            cute::initialize_barrier(smem.vd_ready_bar, kFusedGdrRoleThreads);
            cute::initialize_barrier(smem.update_ready_bar, kFusedGdrRoleThreads);
#pragma unroll
            for (int stage = 0; stage < 2; ++stage) {
                cute::initialize_barrier(smem.stage_ready_mbar[stage], 96);
                cute::initialize_barrier(smem.stage_free_bar[stage], kFusedGdrConsumerThreads);
            }
#pragma unroll
            for (int stage = 0; stage < kFusedGdrOutputStoreStages; ++stage) {
                cute::initialize_barrier(smem.out_ready_bar[stage], kFusedGdrRoleThreads);
                cute::initialize_barrier(smem.out_free_bar[stage], 1);
            }
            // Release: publish thread 0's barrier initialization before the CTA rendezvous.
            cutlass::arch::fence_barrier_init();
        }
        // Acquire: the CTA rendezvous lets every role observe initialized barrier state.
        __syncthreads();

        if (wg_idx == 3) {
            cutlass::arch::warpgroup_reg_dealloc<kFusedGdrProducerRegisters>();

            constexpr int kQkTmaBytes     = kChunkSize * kHeadDim * static_cast<int>(sizeof(T));
            constexpr int kQkHalfElements = kChunkSize * (kHeadDim / 2);
            constexpr int kValueTmaBytes  = kChunkSize * BlockDv * static_cast<int>(sizeof(T));
            constexpr int kSquareTmaBytes = kChunkSize * kChunkSize * static_cast<int>(sizeof(T));

            if (role_tid < 32) {
                for (int chunk = 0; chunk < chunks; ++chunk) {
                    const int stage      = chunk & 1;
                    const int free_phase = ((chunk >> 1) & 1) ^ 1;
                    const int token0     = token_base + chunk * kChunkSize;
                    // Acquire: the Q/K producer warp waits for WGs 0-2 to release all
                    // reads of this stage. Complementary parity makes generation 0 free;
                    // each later phase toggles after 384 consumer arrivals.
                    cute::wait_barrier(smem.stage_free_bar[stage], free_phase);
                    if (role_tid == 0) {
                        // Release: the warp-0 leader registers the Q/K transaction bytes;
                        // their TMA completion releases those bytes to stage_ready_mbar.
                        cutlass::arch::ClusterTransactionBarrier::expect_transaction(&smem.stage_ready_mbar[stage],
                                                                                     2 * kQkTmaBytes);
                        cute::SM90_TMA_LOAD_5D::copy(q_desc,
                                                     &smem.stage_ready_mbar[stage],
                                                     kTmaNoCacheHint,
                                                     &smem.q_stage[stage][0][0],
                                                     0,
                                                     0,
                                                     qk_tma_head_coord,
                                                     token0,
                                                     qk_tma_batch_coord);
                        cute::SM90_TMA_LOAD_5D::copy(q_desc,
                                                     &smem.stage_ready_mbar[stage],
                                                     kTmaNoCacheHint,
                                                     &smem.q_stage[stage][0][0] + kQkHalfElements,
                                                     0,
                                                     1,
                                                     qk_tma_head_coord,
                                                     token0,
                                                     qk_tma_batch_coord);
                        cute::SM90_TMA_LOAD_5D::copy(k_desc,
                                                     &smem.stage_ready_mbar[stage],
                                                     kTmaNoCacheHint,
                                                     &smem.k_stage[stage][0][0],
                                                     0,
                                                     0,
                                                     qk_tma_head_coord,
                                                     token0,
                                                     qk_tma_batch_coord);
                        cute::SM90_TMA_LOAD_5D::copy(k_desc,
                                                     &smem.stage_ready_mbar[stage],
                                                     kTmaNoCacheHint,
                                                     &smem.k_stage[stage][0][0] + kQkHalfElements,
                                                     0,
                                                     1,
                                                     qk_tma_head_coord,
                                                     token0,
                                                     qk_tma_batch_coord);
                    }
                    // Release: all 32 Q/K producer lanes arrive after warp role work.
                    // Consumers acquire only after these arrivals and the registered
                    // Q/K transaction bytes have completed.
                    cute::arrive_barrier(smem.stage_ready_mbar[stage]);
                }
                return;
            }

            if (role_tid < 64) {
                for (int chunk = 0; chunk < chunks; ++chunk) {
                    const int stage      = chunk & 1;
                    const int free_phase = ((chunk >> 1) & 1) ^ 1;
                    const int token0     = token_base + chunk * kChunkSize;
                    const int valid      = min(seq_len - chunk * kChunkSize, kChunkSize);
                    // Acquire: the V/beta producer warp waits for WGs 0-2 to release all
                    // reads of this stage. Complementary parity makes generation 0 free;
                    // each later phase toggles after 384 consumer arrivals.
                    cute::wait_barrier(smem.stage_free_bar[stage], free_phase);
                    if (role_tid == 32) {
                        // Release: the warp-1 leader registers only the V transaction bytes;
                        // V TMA completion releases those bytes to stage_ready_mbar.
                        cutlass::arch::ClusterTransactionBarrier::expect_transaction(&smem.stage_ready_mbar[stage],
                                                                                     kValueTmaBytes);
                        if constexpr (BlockDv == kWideGdrBlockDv) {
                            cute::SM90_TMA_LOAD_4D::copy(v_desc,
                                                         &smem.stage_ready_mbar[stage],
                                                         kTmaNoCacheHint,
                                                         &smem.v_stage[stage][0][0],
                                                         dv0,
                                                         value_head,
                                                         token0,
                                                         0);
                            cute::SM90_TMA_LOAD_4D::copy(v_desc,
                                                         &smem.stage_ready_mbar[stage],
                                                         kTmaNoCacheHint,
                                                         &smem.v_stage[stage][kFusedGdrBlockDv][0],
                                                         dv0 + kFusedGdrBlockDv,
                                                         value_head,
                                                         token0,
                                                         0);
                        }
                        else {
                            cute::SM90_TMA_LOAD_4D::copy(v_desc,
                                                         &smem.stage_ready_mbar[stage],
                                                         kTmaNoCacheHint,
                                                         &smem.v_stage[stage][0][0],
                                                         dv0,
                                                         value_head,
                                                         token0,
                                                         0);
                        }
                    }
                    IssueGateStageCpAsync(smem.gate_stage[stage][1],
                                          beta,
                                          role_tid - 32,
                                          valid,
                                          token0,
                                          beta_sequence_offset,
                                          beta_stride);
                    // Release: each V/beta lane attaches its pre-counted arrival to
                    // completion of both per-head beta cp.async copies. Consumers
                    // acquire beta together with completion of the V transaction bytes.
                    cutlass::arch::cpasync_barrier_arrive_noinc(&smem.stage_ready_mbar[stage]);
                }
                return;
            }

            if (role_tid < 96) {
                for (int chunk = 0; chunk < chunks; ++chunk) {
                    const int stage      = chunk & 1;
                    const int free_phase = ((chunk >> 1) & 1) ^ 1;
                    const int token0     = token_base + chunk * kChunkSize;
                    const int valid      = min(seq_len - chunk * kChunkSize, kChunkSize);
                    // Acquire: the resolvent/g producer warp waits for WGs 0-2 to release
                    // all reads of this stage. Complementary parity makes generation 0
                    // free; each later phase toggles after 384 consumer arrivals.
                    cute::wait_barrier(smem.stage_free_bar[stage], free_phase);
                    if (role_tid == 64) {
                        // Release: the warp-2 leader registers only the resolvent transaction
                        // bytes; resolvent TMA completion releases them to stage_ready_mbar.
                        cutlass::arch::ClusterTransactionBarrier::expect_transaction(&smem.stage_ready_mbar[stage],
                                                                                     kSquareTmaBytes);
                        cute::SM90_TMA_LOAD_4D::copy(resolvent_desc,
                                                     &smem.stage_ready_mbar[stage],
                                                     kTmaNoCacheHint,
                                                     &smem.a_stage[stage][0][0],
                                                     0,
                                                     value_head,
                                                     token0,
                                                     0);
                    }
                    IssueGateStageCpAsync(smem.gate_stage[stage][0],
                                          g_cumsum,
                                          role_tid - 64,
                                          valid,
                                          token0,
                                          gate_sequence_offset,
                                          gate_stride);
                    // Release: each resolvent/g lane attaches its pre-counted arrival to
                    // completion of both per-head g cp.async copies. Consumers acquire
                    // g together with completion of the resolvent transaction bytes.
                    cutlass::arch::cpasync_barrier_arrive_noinc(&smem.stage_ready_mbar[stage]);
                }
                return;
            }

            using OutputStorePipeline = cutlass::PipelineTmaStore<kFusedGdrOutputStoreStages>;
            OutputStorePipeline                         output_store_pipeline{};
            typename OutputStorePipeline::PipelineState output_store_state{};
            const int                                   output_store_lane = role_tid - 96;

            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int output_stage = chunk % kFusedGdrOutputStoreStages;
                const int output_phase = (chunk / kFusedGdrOutputStoreStages) & 1;
                const int token0       = token_base + chunk * kChunkSize;

                if (output_store_lane == 0) {
                    // Acquire WG2's complete STSM tile, then release its generic-proxy
                    // writes to the TMA async proxy before issuing this store group.
                    cute::wait_barrier(smem.out_ready_bar[output_stage], output_phase);
                    cute::tma_store_fence();
                    if constexpr (BlockDv == kWideGdrBlockDv) {
                        cute::SM90_TMA_STORE_4D::copy(
                            out_desc, &smem.o_shared[output_stage][0][0], dv0, value_head, token0, 0);
                        cute::SM90_TMA_STORE_4D::copy(out_desc,
                                                      &smem.o_shared[output_stage][kFusedGdrBlockDv][0],
                                                      dv0 + kFusedGdrBlockDv,
                                                      value_head,
                                                      token0,
                                                      0);
                    }
                    else {
                        cute::SM90_TMA_STORE_4D::copy(
                            out_desc, &smem.o_shared[output_stage][0][0], dv0, value_head, token0, 0);
                    }

                    output_store_pipeline.producer_commit(output_store_state);
                    ++output_store_state;
                    // Keep one committed store group in flight. Once this returns, the
                    // older of the two shared slots is no longer read by TMA.
                    output_store_pipeline.producer_acquire(output_store_state);
                    if (chunk > 0) {
                        const int completed_stage = (chunk - 1) % kFusedGdrOutputStoreStages;
                        // Release the oldest slot only after the store-pipeline acquire
                        // has established TMA read completion.
                        cute::arrive_barrier(smem.out_free_bar[completed_stage]);
                    }
                }
                __syncwarp();
            }
            if (output_store_lane == 0) {
                output_store_pipeline.producer_tail(output_store_state);
            }
            return;
        }

        if (wg_idx == 0) {
            cutlass::arch::warpgroup_reg_alloc<kFusedGdrStateRegisters>();
            using Element = typename FusedGdrMmaTraits<T>::Element;

            using StateTileShape   = cute::Shape<cute::Int<64>, cute::Int<BlockDv>, cute::Int<64>>;
            using StateGmmaAtom    = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                                            Element,
                                                                            float,
                                                                            StateTileShape,
                                                                            cute::SM90::GMMA::Major::MN,
                                                                            cute::SM90::GMMA::Major::MN>());
            auto state_mma         = cute::make_tiled_mma(StateGmmaAtom{});
            auto thr_mma           = state_mma.get_thread_slice(role_tid);
            auto state_tile_layout = cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<BlockDv>{}),
                                                       cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{}));
            auto g_state_for_fragment =
                cute::make_tensor(cute::make_gmem_ptr(static_cast<float*>(nullptr)), state_tile_layout);
            auto tCgStateForFragment = thr_mma.partition_C(g_state_for_fragment);
            auto tCrState            = thr_mma.make_fragment_C(tCgStateForFragment);
            if constexpr (ContextParallel) {
                auto* cp_state_base = reinterpret_cast<float*>(static_cast<uintptr_t>(cp_state_ptrs[segment_id]));
                auto* state_base    = cp_state_base + static_cast<int64_t>(value_head) * kHeadDim * kHeadDim + dv0;
                auto  g_state       = cute::make_tensor(cute::make_gmem_ptr(state_base), state_tile_layout);
                FusedGdrLoadStateFragmentGlobal<float>(tCrState, g_state, thr_mma, role_tid);
            }
            else {
                auto* state_base =
                    GroupedStateBase<StateT>(
                        state_ptrs, batch_id, value_head, num_head_groups, heads_per_block, state_layer_offset)
                    + dv0;
                auto g_state = cute::make_tensor(cute::make_gmem_ptr(state_base), state_tile_layout);
                FusedGdrLoadStateFragmentGlobal<StateT>(tCrState, g_state, thr_mma, role_tid);
            }
            // Rendezvous: named barrier 1 aligns WG0 after loading the state fragment
            // before the first state-snapshot phase.
            MmaSyncNamed<kBarrierStateUpdate>();

            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int token0      = chunk * kChunkSize;
                const int valid       = min(seq_len - token0, kChunkSize);
                const int stage       = chunk & 1;
                const int phase       = chunk & 1;
                const int stage_phase = (chunk >> 1) & 1;

                // Acquire: WG0 waits for 32 ordinary and 64 cp.async-completion arrivals
                // plus completion of the Q/K, V, and resolvent TMA bytes. This makes the
                // matrix tiles and per-head g/beta cp.async copies visible to WG0.
                cute::wait_barrier(smem.stage_ready_mbar[stage], stage_phase);

                auto s_h = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.h_shared[0][0])),
                                             FusedGdrGmmaStateRowLayout<Element, BlockDv>());
                FusedGdrStoreFragmentBf16Stsm<T, Element>(tCrState, s_h, thr_mma, role_tid);
                // Rendezvous: named barrier 1 completes WG0's h_shared STSM
                // publication before the snapshot is released.
                MmaSyncNamed<kBarrierStateUpdate>();
                // Release: WG0 publishes h_shared to WGs 1 and 2 for this chunk phase.
                cute::arrive_barrier(smem.state_snapshot_bar);

                // Acquire: WG1 has published g, g_exp, and g_rev_exp for state decay.
                cute::wait_barrier(smem.gate_ready_bar, phase);
                FusedGdrDecayStateFragment(tCrState, smem.g_exp[kChunkSize - 1]);
                // Release: WG0 has consumed the current single-slot gate generation.
                StateGateReadArrive();

                // Rendezvous: WG0 waits only until WGs 1/2 have drained their
                // h_shared WGMMA reads; output and VD tails proceed independently.
                StateWaitForHReaders();
                // Acquire: WG1 has published VN for WG0's state-update GMMA.
                cute::wait_barrier(smem.update_ready_bar, phase);
                auto s_k_t =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.k_stage[stage][0][0])),
                                      FusedGdrGmmaQkTransposeALayout<Element>());
                auto s_vn = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.vn_shared[0][0])),
                                              FusedGdrGmmaVdTLayout<Element, BlockDv>());
                // Rendezvous: named barrier 1 aligns WG0 before the state update reads VN.
                MmaSyncNamed<kBarrierStateUpdate>();
                FusedGdrStateUpdateFragmentGmmaBf16Vd<BlockDv, T>(role_tid, s_k_t, s_vn, tCrState);
                // Rendezvous: named barrier 1 aligns WG0 after the separately drained
                // update WGMMA before the state fragment is stored or republished.
                MmaSyncNamed<kBarrierStateUpdate>();

                const bool store_final_state =
                    chunk == chunks - 1 && (!ContextParallel || sequence_terminal) && !finished[batch_id];
                if (store_final_state) {
                    auto* state_base =
                        GroupedStateBase<StateT>(
                            state_ptrs, sequence_id, value_head, num_head_groups, heads_per_block, state_layer_offset)
                        + dv0;
                    auto g_state = cute::make_tensor(cute::make_gmem_ptr(state_base), state_tile_layout);
                    FusedGdrStoreStateFragmentGlobal<StateT>(tCrState, g_state, thr_mma, role_tid);
                    // Rendezvous: named barrier 1 completes WG0's final-state store phase.
                    MmaSyncNamed<kBarrierStateUpdate>();
                }
                // Release WG0's read ownership to the producer warps. The stage becomes
                // reusable only after WGs 1 and 2 contribute their 128 arrivals.
                cute::arrive_barrier(smem.stage_free_bar[stage]);
            }
            return;
        }
        else if (wg_idx == 1) {
            cutlass::arch::warpgroup_reg_alloc<kFusedGdrValueRegisters>();
            using Element   = typename FusedGdrMmaTraits<T>::Element;
            using TileShape = cute::Shape<cute::Int<64>, cute::Int<BlockDv>, cute::Int<64>>;
            using GmmaAtom  = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                                       Element,
                                                                       float,
                                                                       TileShape,
                                                                       cute::SM90::GMMA::Major::K,
                                                                       cute::SM90::GMMA::Major::MN>());
            auto tiled_mma  = cute::make_tiled_mma(GmmaAtom{});
            auto thr_mma    = tiled_mma.get_thread_slice(role_tid);
            using ValueMma  = decltype(tiled_mma);
            auto s_sv_c =
                cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.o_shared[0][0][0])),
                                  cute::make_layout(cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<BlockDv>{}),
                                                    cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{})));
            auto       tCsSv        = thr_mma.partition_C(s_sv_c);
            auto       tCrU         = thr_mma.make_fragment_C(tCsSv);
            const auto c_projection = MakeGmmaCThreadProjection<ValueMma>(role_tid);

            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int token0      = chunk * kChunkSize;
                const int valid       = min(seq_len - token0, kChunkSize);
                const int stage       = chunk & 1;
                const int phase       = chunk & 1;
                const int stage_phase = (chunk >> 1) & 1;

                // Acquire: WG1 waits for 32 ordinary and 64 cp.async-completion arrivals
                // plus completion of the Q/K, V, and resolvent TMA bytes. This makes the
                // matrix tiles and per-head g/beta cp.async copies visible to WG1.
                cute::wait_barrier(smem.stage_ready_mbar[stage], stage_phase);

                const int   gate_warp     = role_tid / kCudaWarpThreads;
                const int   gate_warp_tid = role_tid % kCudaWarpThreads;
                const float last_g_value  = smem.gate_stage[stage][0][valid - 1];
                if (gate_warp_tid < kFusedGdrGateRowsPerWarp) {
#pragma unroll
                    for (int pass = 0; pass < kFusedGdrGatePasses; ++pass) {
                        const int row =
                            pass * kFusedGdrGateWriterThreads + gate_warp * kFusedGdrGateRowsPerWarp + gate_warp_tid;
                        const float g_value = row < valid ? smem.gate_stage[stage][0][row] : last_g_value;
                        smem.g[row]         = g_value;
                        smem.g_exp[row]     = FastExp(g_value);
                        smem.g_rev_exp[row] = row < valid ? FastExp(last_g_value - g_value) : 0.0f;
                    }
                }
                // Rendezvous: named barrier 2 completes WG1's gate-vector stores before
                // releasing them to WGs 0 and 2.
                MmaSyncNamed<kBarrierValueU>();
                // Release: WG1 publishes g, g_exp, and g_rev_exp to WGs 0 and 2.
                cute::arrive_barrier(smem.gate_ready_bar);
                // Acquire: WG0 has published the h_shared snapshot consumed by WG1.
                cute::wait_barrier(smem.state_snapshot_bar, phase);

                auto s_k =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.k_stage[stage][0][0])),
                                      FusedGdrGmmaQkKLayout<Element>());
                auto s_h = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.h_shared[0][0])),
                                             FusedGdrGmmaStateTLayout<Element, BlockDv>());
                FusedGdrGmmaSs(tiled_mma, role_tid, s_k, s_h, tCrU, cute::SM90::GMMA::ScaleOut::Zero);
                // Release: WG1 has drained its h_shared WGMMA read for this generation.
                HReaderArrive();

                auto s_v =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.v_stage[stage][0][0])),
                                      FusedGdrSwizzledVTLayout<BlockDv>());
                auto s_w =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.v_stage[stage][0][0])),
                                      FusedGdrGmmaVdTLayout<Element, BlockDv>());
                ForEachProjectedGmmaC<ValueMma>(
                    c_projection,
                    [&](int row) { return smem.g_exp[row]; },
                    [&](auto fragment_coord, int row, int dv, float row_g_exp) {
                        float w = 0.0f;
                        if (row < valid) {
                            w = static_cast<float>(s_v(dv, row)) - row_g_exp * static_cast<float>(tCrU(fragment_coord));
                        }
                        tCrU(fragment_coord) = w;
                    });
                // FlashQLA keeps the original V loads and the W stores separated
                // because both use the same shared arena through different layouts.
                // Rendezvous: named barrier 2 completes all original V reads before WG1
                // overwrites the same v_stage storage with W.
                MmaSyncNamed<kBarrierValueU>();
                auto s_w_store =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.v_stage[stage][0][0])),
                                      FusedGdrGmmaVdRowLayout<Element, BlockDv>());
                FusedGdrStoreFragmentBf16Stsm<T, Element>(tCrU, s_w_store, thr_mma, role_tid);
                // Rendezvous: named barrier 2 completes WG1's W stores before the second
                // GMMA consumes W through the async proxy.
                MmaSyncNamed<kBarrierValueU>();

                // Acquire: WG2 has published packed AG before WG1's second GMMA.
                cute::wait_barrier(smem.ag_ready_bar, phase);
                auto s_a =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.a_stage[stage][0][0])),
                                      GmmaSquareKLayout<Element>());
                FusedGdrGmmaSs(tiled_mma, role_tid, s_a, s_w, tCrU, cute::SM90::GMMA::ScaleOut::Zero);

                ForEachProjectedGmmaC<ValueMma>(
                    c_projection,
                    [&](int row) { return row; },
                    [&](auto fragment_coord, int row, int, int) {
                        tCrU(fragment_coord) = row < valid ? static_cast<float>(tCrU(fragment_coord)) : 0.0f;
                    });
                auto s_vd_store =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.vd_shared[0][0])),
                                      FusedGdrGmmaVdRowLayout<Element, BlockDv>());
                FusedGdrStoreFragmentBf16Stsm<T, Element>(tCrU, s_vd_store, thr_mma, role_tid);
                // Release: WG1 publishes VD to WG2 for the output GMMA.
                cute::arrive_barrier(smem.vd_ready_bar);
                // Rendezvous: named barrier 2 aligns WG1 after VD publication before
                // transforming the fragment and publishing VN.
                MmaSyncNamed<kBarrierValueU>();
                ForEachProjectedGmmaC<ValueMma>(
                    c_projection,
                    [&](int row) { return smem.g_rev_exp[row]; },
                    [&](auto fragment_coord, int, int, float row_g_rev_exp) {
                        tCrU(fragment_coord) = row_g_rev_exp * static_cast<float>(tCrU(fragment_coord));
                    });
                auto s_vn_store =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.vn_shared[0][0])),
                                      FusedGdrGmmaVdRowLayout<Element, BlockDv>());
                FusedGdrStoreFragmentBf16Stsm<T, Element>(tCrU, s_vn_store, thr_mma, role_tid);
                // Release: WG1 publishes VN to WG0 for the state-update GMMA.
                cute::arrive_barrier(smem.update_ready_bar);
                // Rendezvous: WG1 waits until WG0 has consumed the gate vectors and
                // WG2 has drained vd_shared before either single-slot arena advances.
                ValueWaitForGateVdReaders();
                // Release WG1's final reads of K, V/W, gates, and AG. The stage becomes
                // reusable only after WGs 0 and 2 contribute their 128 arrivals.
                cute::arrive_barrier(smem.stage_free_bar[stage]);
            }
            return;
        }
        else if (wg_idx == 2) {
            cutlass::arch::warpgroup_reg_alloc<kFusedGdrOutputRegisters>();
            using Element          = typename FusedGdrMmaTraits<T>::Element;
            using SquareTileShape  = cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>;
            using SquareGmmaAtom   = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                                             Element,
                                                                             float,
                                                                             SquareTileShape,
                                                                             cute::SM90::GMMA::Major::K,
                                                                             cute::SM90::GMMA::Major::K>());
            using OutputTileShape  = cute::Shape<cute::Int<64>, cute::Int<BlockDv>, cute::Int<64>>;
            using OutputSsGmmaAtom = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                                               Element,
                                                                               float,
                                                                               OutputTileShape,
                                                                               cute::SM90::GMMA::Major::K,
                                                                               cute::SM90::GMMA::Major::MN>());
            using OutputRsGmmaAtom = decltype(cute::SM90::GMMA::rs_op_selector<Element,
                                                                               Element,
                                                                               float,
                                                                               OutputTileShape,
                                                                               cute::SM90::GMMA::Major::K,
                                                                               cute::SM90::GMMA::Major::MN>());
            auto square_mma        = cute::make_tiled_mma(SquareGmmaAtom{});
            auto output_mma        = cute::make_tiled_mma(OutputSsGmmaAtom{});
            auto output_rs_mma     = cute::make_tiled_mma(OutputRsGmmaAtom{});
            auto thr_square        = square_mma.get_thread_slice(role_tid);
            auto thr_output        = output_mma.get_thread_slice(role_tid);
            auto thr_output_rs     = output_rs_mma.get_thread_slice(role_tid);
            using SquareMma        = decltype(square_mma);
            using OutputMma        = decltype(output_mma);
            using OutputRsMma      = decltype(output_rs_mma);
            using SquareAtomShape  = typename SquareMma::AtomShape_MNK;
            using OutputAtomShape  = typename OutputMma::AtomShape_MNK;
            static_assert(std::is_same_v<GmmaCThreadLayout<SquareMma>, GmmaCThreadLayout<OutputMma>>);
            static_assert(std::is_same_v<GmmaCLayout<OutputMma>, GmmaCLayout<OutputRsMma>>);
            static_assert(decltype(cute::size<0>(SquareAtomShape{}))::value
                          == decltype(cute::size<0>(OutputAtomShape{}))::value);
            auto s_o_c =
                cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.o_shared[0][0][0])),
                                  cute::make_layout(cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<BlockDv>{}),
                                                    cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{})));
            auto       tCsO         = thr_output.partition_C(s_o_c);
            const auto c_projection = MakeGmmaCThreadProjection<SquareMma>(role_tid);

            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int token0      = chunk * kChunkSize;
                const int valid       = min(seq_len - token0, kChunkSize);
                const int stage       = chunk & 1;
                const int phase       = chunk & 1;
                const int stage_phase = (chunk >> 1) & 1;

                // Acquire: WG2 waits for 32 ordinary and 64 cp.async-completion arrivals
                // plus completion of the Q/K, V, and resolvent TMA bytes. This makes the
                // matrix tiles and per-head g/beta cp.async copies visible to WG2.
                cute::wait_barrier(smem.stage_ready_mbar[stage], stage_phase);

                auto s_q =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.q_stage[stage][0][0])),
                                      FusedGdrGmmaQkKLayout<Element>());
                auto s_k =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.k_stage[stage][0][0])),
                                      FusedGdrGmmaQkKLayout<Element>());
                auto s_ag =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.a_stage[stage][0][0])),
                                      GmmaSquareKLayout<Element>());
                auto tCrP = cute::partition_fragment_C(
                    square_mma, cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<kChunkSize>{}));
                FusedGdrGmmaSs<false>(square_mma, role_tid, s_q, s_k, tCrP, cute::SM90::GMMA::ScaleOut::Zero);

                // Acquire: WG1 has published g, g_exp, and g_rev_exp for P/AG formation.
                cute::wait_barrier(smem.gate_ready_bar, phase);
                auto s_a_raw = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.a_stage[stage][0][0])), SwizzledA64Layout());
                {
                    auto tCrAgPack = cute::make_fragment_like<Element>(tCrP);
                    ForEachProjectedGmmaC<SquareMma>(
                        c_projection,
                        [&](int row) { return smem.g[row]; },
                        [&](int column) {
                            const float beta_column = column < valid ? smem.gate_stage[stage][1][column] : 0.0f;
                            return cute::make_tuple(smem.g[column], beta_column);
                        },
                        [&](auto fragment_coord, int row, int column, float row_g, auto column_data) {
                            const float column_g    = cute::get<0>(column_data);
                            const float beta_column = cute::get<1>(column_data);

                            float g_rel = 0.0f;
                            if (row < valid && column < valid && column <= row) {
                                g_rel = FastExp(row_g - column_g);
                            }

                            const float a_value       = static_cast<float>(s_a_raw(row, column)) * g_rel * beta_column;
                            const float p_value       = static_cast<float>(tCrP(fragment_coord)) * kHeadScale * g_rel;
                            tCrAgPack(fragment_coord) = Element(CastFromFloat(a_value));
                            tCrP(fragment_coord)      = p_value;
                        });
                    StorePackedBf16Stsm<Element>(tCrAgPack, s_ag, thr_square, role_tid);
                }
                // Rendezvous: named barrier 4 completes WG2's packed AG STSM before
                // AG is released to WG1.
                MmaSyncNamed<kBarrierOutputAg>();
                // Release: WG2 publishes packed AG; WG1 acquires it before its second GMMA.
                cute::arrive_barrier(smem.ag_ready_bar);

                auto tCrPrs = thr_output_rs.partition_fragment_A(s_ag);
                PackSquareCAsRsA<Element, SquareMma, OutputRsMma>(tCrP, tCrPrs);

                // Acquire: WG0 has published the h_shared snapshot consumed by WG2.
                cute::wait_barrier(smem.state_snapshot_bar, phase);
                auto s_h  = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.h_shared[0][0])),
                                             FusedGdrGmmaStateTLayout<Element, BlockDv>());
                auto tCrO = thr_output.make_fragment_C(tCsO);
                FusedGdrGmmaSs(output_mma, role_tid, s_q, s_h, tCrO, cute::SM90::GMMA::ScaleOut::Zero);
                // Release: WG2 has drained its h_shared WGMMA read for this generation.
                HReaderArrive();

                ForEachProjectedGmmaC<OutputMma>(
                    c_projection,
                    [&](int row) { return smem.g_exp[row]; },
                    [&](auto fragment_coord, int row, int, float row_g_exp) {
                        if (row < valid) {
                            tCrO(fragment_coord) = static_cast<float>(tCrO(fragment_coord)) * kHeadScale * row_g_exp;
                        }
                        else {
                            tCrO(fragment_coord) = 0.0f;
                        }
                    });

                // Acquire: WG1 has published VD for WG2's output GMMA.
                cute::wait_barrier(smem.vd_ready_bar, phase);
                auto s_vd = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.vd_shared[0][0])),
                                              FusedGdrGmmaVdTLayout<Element, BlockDv>());
                GmmaRs(output_rs_mma, role_tid, tCrPrs, s_vd, tCrO, cute::SM90::GMMA::ScaleOut::One);

                // Release WG1 after the final RS-WGMMA has drained its vd_shared read.
                OutputVdReadArrive();

                const int output_stage      = chunk % kFusedGdrOutputStoreStages;
                const int output_free_phase = ((chunk / kFusedGdrOutputStoreStages) & 1) ^ 1;
                // Acquire this output slot only after the store warp has established
                // completion of the preceding TMA read. Complementary parity makes
                // the first use of both slots immediately available.
                cute::wait_barrier(smem.out_free_bar[output_stage], output_free_phase);
                auto s_out = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.o_shared[output_stage][0][0])),
                    SwizzledVRowLayout<BlockDv>());
                FusedGdrStoreFragmentBf16Stsm<T, Element>(tCrO, s_out, thr_output, role_tid);
                // Rendezvous: named barrier 5 completes WG2's output STSM publication
                // before o_shared is released to the output-store leader.
                MmaSyncNamed<kBarrierOutputLocal>();
                // Release: WG2 publishes this output slot to the store leader.
                cute::arrive_barrier(smem.out_ready_bar[output_stage]);
                // Release WG2's final reads of Q/K, resolvent, and beta. The stage
                // becomes reusable only after WGs 0 and 1 contribute their 128 arrivals.
                cute::arrive_barrier(smem.stage_free_bar[stage]);
            }
            return;
        }
    }
};

template<class T, class StateT, int BlockDv, bool ContextParallel>
__global__
    __launch_bounds__(Sm90FusedGdrFwd<T, StateT, BlockDv, ContextParallel>::kThreads,
                      Sm90FusedGdrFwd<T, StateT, BlockDv, ContextParallel>::
                          kMinBlocks) void Sm90FusedGdrFwdKernel(const CUtensorMap* __restrict__ tma_desc_workspace,
                                                                 const float* __restrict__ g_cumsum,
                                                                 const float* __restrict__ beta,
                                                                 int64_t gate_batch_stride,
                                                                 int64_t gate_stride,
                                                                 int64_t beta_batch_stride,
                                                                 int64_t beta_stride,
                                                                 int     token_num,
                                                                 const int32_t* __restrict__ q_offsets,
                                                                 const bool* __restrict__ finished,
                                                                 const int32_t* __restrict__ data_q_offsets,
                                                                 const int32_t* __restrict__ cp_source_indices,
                                                                 const int64_t* __restrict__ cp_state_ptrs,
                                                                 const int64_t* __restrict__ state_ptrs,
                                                                 int64_t state_layer_offset,
                                                                 int     data_sequence_num,
                                                                 int     hq,
                                                                 int     hv,
                                                                 int     num_head_groups,
                                                                 int     heads_per_block)
{
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    Sm90FusedGdrFwd<T, StateT, BlockDv, ContextParallel>::Run(tma_desc_workspace,
                                                              g_cumsum,
                                                              beta,
                                                              gate_batch_stride,
                                                              gate_stride,
                                                              beta_batch_stride,
                                                              beta_stride,
                                                              token_num,
                                                              q_offsets,
                                                              finished,
                                                              data_q_offsets,
                                                              cp_source_indices,
                                                              cp_state_ptrs,
                                                              state_ptrs,
                                                              state_layer_offset,
                                                              data_sequence_num,
                                                              hq,
                                                              hv,
                                                              num_head_groups,
                                                              heads_per_block,
                                                              smem_raw);
}

template<class StateT, int BlockDv, bool ContextParallel>
void SetFusedGdrFwdSharedMemoryLimit(size_t smem_bytes)
{
    static_assert(kFusedGdrValidStateT<StateT>, "fused chunk GDR StateT must be float or bfloat16");
    static const cudaError_t status =
        cudaFuncSetAttribute(Sm90FusedGdrFwdKernel<__nv_bfloat16, StateT, BlockDv, ContextParallel>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
    TM_CUDA_CHECK(status);
}

template<class StateT, int BlockDv, bool ContextParallel>
void LaunchSm90FusedGdrFwdTyped(const core::Tensor& q,
                                const core::Tensor& k,
                                const core::Tensor& v,
                                const core::Tensor& g_cumsum,
                                const core::Tensor& beta,
                                const core::Tensor& resolvent,
                                const core::Tensor& state_ptrs,
                                const core::Tensor& q_offsets,
                                const core::Tensor& finished,
                                core::Tensor&       out,
                                const Problem&      problem,
                                int64_t             state_layer_offset,
                                const core::Tensor* data_q_offsets,
                                const core::Tensor* cp_source_indices,
                                const core::Tensor* cp_state_ptrs,
                                int                 data_sequence_num,
                                void*               tma_desc_workspace,
                                cudaStream_t        stream)
{
    static_assert(kFusedGdrValidStateT<StateT>, "fused chunk GDR StateT must be float or bfloat16");
    using Kernel = Sm90FusedGdrFwd<__nv_bfloat16, StateT, BlockDv, ContextParallel>;
    static_cast<void>(q);
    static_cast<void>(k);
    static_cast<void>(v);
    static_cast<void>(resolvent);
    static_cast<void>(out);

    const int      descriptor_sequence_num = ContextParallel ? data_sequence_num : problem.sequence_num;
    const auto*    q_offsets_ptr           = q_offsets.data<int32_t>();
    const auto*    finished_ptr            = finished.data<bool>();
    const int32_t* data_q_offsets_ptr      = nullptr;
    const int32_t* cp_source_indices_ptr   = nullptr;
    const int64_t* cp_state_ptrs_ptr       = nullptr;
    if constexpr (ContextParallel) {
        data_q_offsets_ptr    = data_q_offsets->data<int32_t>();
        cp_source_indices_ptr = cp_source_indices->data<int32_t>();
        cp_state_ptrs_ptr     = cp_state_ptrs->data<int64_t>();
    }

    constexpr int block_dv     = BlockDv;
    const int     dv_tiles     = CeilDiv(kHeadDim, block_dv);
    const dim3    grid         = dim3(problem.sequence_num, problem.hv, dv_tiles);
    const dim3    block        = dim3(Kernel::kThreads);
    const size_t  smem_bytes   = Kernel::SharedBytes();
    auto*         tma_desc_ptr = reinterpret_cast<CUtensorMap*>(tma_desc_workspace);

    SetFusedGdrFwdSharedMemoryLimit<StateT, block_dv, ContextParallel>(smem_bytes);
    Sm90FusedGdrFwdKernel<__nv_bfloat16, StateT, block_dv, ContextParallel>
        <<<grid, block, smem_bytes, stream>>>(tma_desc_ptr,
                                              g_cumsum.data<float>(),
                                              beta.data<float>(),
                                              problem.gate_batch_stride,
                                              problem.gate_stride,
                                              problem.beta_batch_stride,
                                              problem.beta_stride,
                                              problem.token_num,
                                              q_offsets_ptr,
                                              finished_ptr,
                                              data_q_offsets_ptr,
                                              cp_source_indices_ptr,
                                              cp_state_ptrs_ptr,
                                              reinterpret_cast<const int64_t*>(state_ptrs.raw_data()),
                                              state_layer_offset,
                                              descriptor_sequence_num,
                                              problem.hq,
                                              problem.hv,
                                              problem.num_head_groups,
                                              problem.heads_per_block);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<class StateT, int BlockDv>
void LaunchSm90FusedGdrFwdRegistered(const core::Tensor& q,
                                     const core::Tensor& k,
                                     const core::Tensor& v,
                                     const core::Tensor& g_cumsum,
                                     const core::Tensor& beta,
                                     const core::Tensor& resolvent,
                                     const core::Tensor& state_ptrs,
                                     const core::Tensor& q_offsets,
                                     const core::Tensor& finished,
                                     core::Tensor&       out,
                                     const Problem&      problem,
                                     int64_t             state_layer_offset,
                                     const core::Tensor* data_q_offsets,
                                     const core::Tensor* cp_source_indices,
                                     const core::Tensor* cp_state_ptrs,
                                     int                 data_sequence_num,
                                     void*               tma_desc_workspace,
                                     cudaStream_t        stream)
{
    static_assert(kFusedGdrValidStateT<StateT>, "fused chunk GDR StateT must be float or bfloat16");
    if (cp_source_indices != nullptr) {
        LaunchSm90FusedGdrFwdTyped<StateT, BlockDv, true>(q,
                                                          k,
                                                          v,
                                                          g_cumsum,
                                                          beta,
                                                          resolvent,
                                                          state_ptrs,
                                                          q_offsets,
                                                          finished,
                                                          out,
                                                          problem,
                                                          state_layer_offset,
                                                          data_q_offsets,
                                                          cp_source_indices,
                                                          cp_state_ptrs,
                                                          data_sequence_num,
                                                          tma_desc_workspace,
                                                          stream);
        return;
    }
    LaunchSm90FusedGdrFwdTyped<StateT, BlockDv, false>(q,
                                                       k,
                                                       v,
                                                       g_cumsum,
                                                       beta,
                                                       resolvent,
                                                       state_ptrs,
                                                       q_offsets,
                                                       finished,
                                                       out,
                                                       problem,
                                                       state_layer_offset,
                                                       data_q_offsets,
                                                       cp_source_indices,
                                                       cp_state_ptrs,
                                                       data_sequence_num,
                                                       tma_desc_workspace,
                                                       stream);
}

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
