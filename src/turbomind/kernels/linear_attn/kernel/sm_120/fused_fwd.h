// Inspired by
// https://github.com/QwenLM/FlashQLA/blob/60f81453143e724bcaf3fc7921e71e7328f6ebcd/flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py

#pragma once

#include "src/turbomind/kernels/linear_attn/kernel/sm_120/common.h"

namespace turbomind::linear_attn::delta_rule {
namespace {

// SM120 grouped-bf16 chunked GDR CTA. State-ptr chunked only; StateT is the external state dtype.
template<class T, class StateT, int BlockDv, bool ContextParallel>
struct Sm120FusedGdrFwd {
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);

    using MmaElement = cute::bfloat16_t;
    using MmaAtom    = cute::SM80_16x8x16_F32BF16BF16F32_TN;

    static constexpr float    kHeadScale             = 0.08838834764831845f;
    static constexpr int      kRoleThreads           = 128;
    static constexpr int      kConsumerThreads       = 3 * kRoleThreads;
    static constexpr int      kThreads               = kConsumerThreads + kRoleThreads;
    static constexpr int      kWarpThreads           = 32;
    static constexpr int      kGateRowsPerWarp       = 8;
    static constexpr int      kGateWriterThreads     = (kRoleThreads / kWarpThreads) * kGateRowsPerWarp;
    static constexpr int      kGatePasses            = kChunk32Size / kGateWriterThreads;
    static constexpr int      kStateRegisters        = 144;
    static constexpr int      kValueRegisters        = 144;
    static constexpr int      kOutputRegisters       = 160;
    static constexpr int      kMinBlocks             = 1;
    static constexpr int      kProducerRegisters     = 32;
    static constexpr int      kFusedGdrDataDescCount = 8;
    static constexpr int      kFusedGdrGateTmaHeads  = 4;
    static constexpr uint64_t kTmaNoCacheHint        = 0;
    static constexpr size_t   kMaxDynamicSharedBytes = 102400 - 1024;

    enum TmaDescIndex : int
    {
        kFusedGdrQDesc = 0,
        kFusedGdrQHiDesc,
        kFusedGdrKDesc,
        kFusedGdrKHiDesc,
        kFusedGdrVDesc,
        kFusedGdrGDesc,
        kFusedGdrResolventDesc,
        kFusedGdrOutDesc,
    };

    template<class TensorMapPtr>
    struct FusedGdrTmaDescriptorSlices {
        TensorMapPtr data{};
        TensorMapPtr state{};
    };

    template<class TensorMapPtr>
    static CUTE_HOST_DEVICE FusedGdrTmaDescriptorSlices<TensorMapPtr> MakeFusedGdrTmaDescriptorSlices(TensorMapPtr base,
                                                                                                      int sequence_num)
    {
        return {base, base + sequence_num * kFusedGdrDataDescCount};
    }

    template<class TensorMapPtr>
    struct ContextParallelFusedGdrTmaDescriptorSlices {
        TensorMapPtr data{};
        TensorMapPtr cp_state{};
    };

    template<class TensorMapPtr>
    static CUTE_HOST_DEVICE ContextParallelFusedGdrTmaDescriptorSlices<TensorMapPtr>
                            MakeContextParallelFusedGdrTmaDescriptorSlices(TensorMapPtr base, int sequence_num)
    {
        return {base, base + sequence_num * kFusedGdrDataDescCount};
    }

    static __device__ __forceinline__ int GateTmaCoord(int value_head)
    {
        return (value_head / kFusedGdrGateTmaHeads) * kFusedGdrGateTmaHeads;
    }

    static __device__ __forceinline__ int CeilDiv(int value, int divisor)
    {
        return (value + divisor - 1) / divisor;
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

    static __device__ __forceinline__ StateT* GroupedStateBase(const int64_t* state_ptrs,
                                                               int            sequence,
                                                               int            value_head,
                                                               int            num_head_groups,
                                                               int            heads_per_block,
                                                               int64_t        state_layer_offset)
    {
        const int     head_group = value_head / heads_per_block;
        const int     local_head = value_head % heads_per_block;
        const int64_t address    = state_ptrs[sequence * num_head_groups + head_group];
        return reinterpret_cast<StateT*>(static_cast<uintptr_t>(address)) + state_layer_offset
               + static_cast<int64_t>(local_head) * kHeadDim * kHeadDim;
    }

    static constexpr int kBarrierConsumer           = 0;
    static constexpr int kBarrierStateUpdate        = 1;
    static constexpr int kBarrierValueU             = 2;
    static constexpr int kBarrierAgReady            = 3;
    static constexpr int kBarrierOutputP            = 4;
    static constexpr int kBarrierOutputLocal        = 5;
    static constexpr int kBarrierProducerStatePack  = 6;
    static constexpr int kBarrierProducerStateStore = 7;
    static constexpr int kBarrierPackedVd           = kBarrierProducerStatePack;
    static constexpr int kBarrierStateToValue       = kBarrierConsumer;
    static constexpr int kBarrierStateToOutput      = kBarrierProducerStateStore;

    static_assert(kThreads == 512);
    static_assert(kGateWriterThreads == kChunk32Size);
    static_assert(kGatePasses == 1);
    static_assert(kBarrierProducerStateStore + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount
                  < cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers);

    template<int BarrierId>
    static __device__ __forceinline__ void MmaSyncNamed()
    {
        cutlass::arch::NamedBarrier::sync(kRoleThreads, BarrierId);
    }

    static __device__ __forceinline__ void StatePackArrive()
    {
        cutlass::arch::NamedBarrier::arrive(2 * kRoleThreads, kBarrierStateToValue);
        cutlass::arch::NamedBarrier::arrive(2 * kRoleThreads, kBarrierStateToOutput);
    }

    static __device__ __forceinline__ void StatePackValueSync()
    {
        cutlass::arch::NamedBarrier::sync(2 * kRoleThreads, kBarrierStateToValue);
    }

    static __device__ __forceinline__ void StatePackOutputSync()
    {
        cutlass::arch::NamedBarrier::sync(2 * kRoleThreads, kBarrierStateToOutput);
    }

    static __device__ __forceinline__ void PackedVdSync()
    {
        cutlass::arch::NamedBarrier::sync(2 * kRoleThreads, kBarrierPackedVd);
    }

    static __device__ __forceinline__ void AgReadyArrive()
    {
        cutlass::arch::NamedBarrier::arrive(2 * kRoleThreads, kBarrierAgReady);
    }

    static __device__ __forceinline__ void AgReadySync()
    {
        cutlass::arch::NamedBarrier::sync(2 * kRoleThreads, kBarrierAgReady);
    }

    static CUTE_HOST_DEVICE constexpr auto SplitQkLayout()
    {
        return cute::composition(
            cute::Swizzle<3, 3, 3>{},
            cute::Layout<cute::Shape<cute::_32, cute::Shape<cute::_64, cute::_2>>,
                         cute::Stride<cute::_64, cute::Stride<cute::_1, cute::Int<kChunk32Size * 64>>>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto SplitQkTransposedLayout()
    {
        return cute::composition(
            cute::Swizzle<3, 3, 3>{},
            cute::Layout<cute::Shape<cute::Shape<cute::_64, cute::_2>, cute::_32>,
                         cute::Stride<cute::Stride<cute::_1, cute::Int<kChunk32Size * 64>>, cute::_64>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto SquareLayout()
    {
        return cute::composition(cute::Swizzle<2, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::_32, cute::_32>, cute::Stride<cute::_32, cute::_1>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto P128BRowLayout()
    {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::_32, cute::_32>, cute::Stride<cute::_64, cute::_1>>{});
    }

    static constexpr int kPOffset        = 0;
    static constexpr int kVdOffset       = 2 * kChunk32Size * kChunk32Size;
    static constexpr int kStageSlotElems = kChunk32Size * kHeadDim;
    static_assert(cute::cosize_v<decltype(SplitQkLayout())> == kChunk32Size * kHeadDim);
    static_assert(cute::cosize_v<decltype(SplitQkTransposedLayout())> == kChunk32Size * kHeadDim);
    static_assert(cute::cosize_v<decltype(SquareLayout())> == kChunk32Size * kChunk32Size);
    static_assert(cute::cosize_v<decltype(P128BRowLayout())> <= kVdOffset);
    static_assert(kVdOffset + kChunk32Size * BlockDv <= kStageSlotElems);

    static CUTE_HOST_DEVICE constexpr auto VRowLayout()
    {
        if constexpr (BlockDv == kContextParallelGdrBlockDv) {
            return SquareLayout();
        }
        else {
            return cute::composition(
                cute::Swizzle<3, 3, 3>{},
                cute::Layout<cute::Shape<cute::_32, cute::_64>, cute::Stride<cute::_64, cute::_1>>{});
        }
    }

    static CUTE_HOST_DEVICE constexpr auto VTLayout()
    {
        if constexpr (BlockDv == kContextParallelGdrBlockDv) {
            return cute::composition(
                cute::Swizzle<2, 3, 3>{},
                cute::Layout<cute::Shape<cute::_32, cute::_32>, cute::Stride<cute::_1, cute::_32>>{});
        }
        else {
            return cute::composition(
                cute::Swizzle<3, 3, 3>{},
                cute::Layout<cute::Shape<cute::_64, cute::_32>, cute::Stride<cute::_1, cute::_64>>{});
        }
    }

    static CUTE_HOST_DEVICE constexpr auto StateTLayout()
    {
        if constexpr (BlockDv == kContextParallelGdrBlockDv) {
            return cute::composition(cute::Swizzle<2, 3, 3>{},
                                     cute::Layout<cute::Shape<cute::Int<BlockDv>, cute::Int<kHeadDim>>,
                                                  cute::Stride<cute::_1, cute::Int<BlockDv>>>{});
        }
        else {
            return cute::composition(cute::Swizzle<3, 3, 3>{},
                                     cute::Layout<cute::Shape<cute::Int<BlockDv>, cute::Int<kHeadDim>>,
                                                  cute::Stride<cute::_1, cute::Int<BlockDv>>>{});
        }
    }

    static CUTE_HOST_DEVICE constexpr auto StateCLayout()
    {
        if constexpr (BlockDv == kContextParallelGdrBlockDv) {
            return cute::composition(cute::Swizzle<2, 3, 3>{},
                                     cute::Layout<cute::Shape<cute::Int<kHeadDim>, cute::Int<BlockDv>>,
                                                  cute::Stride<cute::Int<BlockDv>, cute::_1>>{});
        }
        else {
            return cute::composition(cute::Swizzle<3, 3, 3>{},
                                     cute::Layout<cute::Shape<cute::Int<kHeadDim>, cute::Int<BlockDv>>,
                                                  cute::Stride<cute::Int<BlockDv>, cute::_1>>{});
        }
    }

    static_assert(cute::cosize_v<decltype(StateTLayout())> == BlockDv * kHeadDim);
    static_assert(cute::cosize_v<decltype(StateCLayout())> == BlockDv * kHeadDim);

    static __device__ __forceinline__ MmaElement CastFromFloat(float value)
    {
        return MmaElement(__float2bfloat16(value));
    }

    static __device__ __forceinline__ float FastExp(float value)
    {
        return exp2f(value * 1.4426950408889634f);
    }

    static __device__ __forceinline__ void StoreBf16Pair(MmaElement* ptr, float2 values)
    {
        uint32_t pack;
        reinterpret_cast<__nv_bfloat162*>(&pack)[0] = __float22bfloat162_rn(values);
        *reinterpret_cast<uint32_t*>(ptr)           = pack;
    }

    template<class... MmaArgs, class TA, class ALayout, class TB, class BLayout, class StateFragment>
    static __device__ __forceinline__ void StateUpdateFragmentFromScaledVd(uint32_t                          thread_idx,
                                                                           cute::TiledMMA<MmaArgs...> const& tiled_mma,
                                                                           cute::Tensor<TA, ALayout> const&  sA,
                                                                           cute::Tensor<TB, BLayout> const&  sB,
                                                                           StateFragment&                    tCrState)
    {
        using InputTypeA = typename TA::value_type;
        using InputTypeB = typename TB::value_type;
        static_assert(std::is_same_v<InputTypeB, MmaElement>);

        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        auto tCrA    = thr_mma.partition_fragment_A(sA);
        auto tCrAi   = cute::make_fragment_like<InputTypeA>(tCrA);
        auto tCrB    = thr_mma.partition_fragment_B(sB);
        auto tCrBi   = cute::make_fragment_like<InputTypeB>(tCrB);

        auto smem_tiled_copy_A =
            cute::make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, InputTypeA>{}, thr_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
        auto tCsA            = smem_thr_copy_A.partition_S(sA);
        auto tCrAi_copy_view = smem_thr_copy_A.retile_D(tCrAi);

        auto smem_tiled_copy_B =
            cute::make_tiled_copy_B(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, InputTypeB>{}, thr_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
        auto tCsB            = smem_thr_copy_B.partition_S(sB);
        auto tCrBi_copy_view = smem_thr_copy_B.retile_D(tCrBi);

        cute::copy(smem_tiled_copy_A,
                   tCsA(cute::_, cute::_, cute::Int<0>{}),
                   tCrAi_copy_view(cute::_, cute::_, cute::Int<0>{}));
        cute::copy(smem_tiled_copy_B,
                   tCsB(cute::_, cute::_, cute::Int<0>{}),
                   tCrBi_copy_view(cute::_, cute::_, cute::Int<0>{}));

        constexpr int K_BLOCK_MAX = cute::size<2>(decltype(tCrA){});
#pragma unroll
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            if (k_block < K_BLOCK_MAX - 1) {
                const int k_next = k_block + 1;
                cute::copy(
                    smem_tiled_copy_A, tCsA(cute::_, cute::_, k_next), tCrAi_copy_view(cute::_, cute::_, k_next));
                cute::copy(
                    smem_tiled_copy_B, tCsB(cute::_, cute::_, k_next), tCrBi_copy_view(cute::_, cute::_, k_next));
            }
            cute::transform(tCrAi(cute::_, cute::_, k_block), tCrA(cute::_, cute::_, k_block), cute::identity{});
            cute::transform(tCrBi(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), cute::identity{});
            cute::gemm(thr_mma, tCrA(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), tCrState);
        }
    }

    template<class StateFragment>
    static __device__ __forceinline__ void DecayStateFragment(StateFragment& tCrState, float decay)
    {
#pragma unroll
        for (int i = 0; i < cute::size(tCrState); ++i) {
            tCrState(i) *= decay;
        }
    }

    template<class GlobalStateT, class StateFragment, class GlobalTensor, class ThrMma>
    static __device__ __forceinline__ void
    LoadStateFragmentGlobal(StateFragment& tCrState, GlobalTensor const& g_state, ThrMma const& thr_mma, int role_tid)
    {
        auto gmem_tiled_copy_C =
            cute::make_tiled_copy_C(cute::Copy_Atom<cute::AutoVectorizingCopy, GlobalStateT>{}, thr_mma);
        auto gmem_thr_copy_C = gmem_tiled_copy_C.get_thread_slice(role_tid);
        auto tCgState        = gmem_thr_copy_C.partition_S(g_state);
        if constexpr (std::is_same_v<GlobalStateT, float>) {
            auto tCrStateView = gmem_thr_copy_C.retile_D(tCrState);
            cute::copy(gmem_tiled_copy_C, tCgState, tCrStateView);
        }
        else {
            static_assert(std::is_same_v<GlobalStateT, __nv_bfloat16>);
            auto tCrPacked     = cute::make_fragment_like<GlobalStateT>(tCrState);
            auto tCrPackedView = gmem_thr_copy_C.retile_D(tCrPacked);
            cute::copy(gmem_tiled_copy_C, tCgState, tCrPackedView);
            cute::copy(tCrPacked, tCrState);
        }
    }

    template<class GlobalStateT, class StateFragment, class GlobalTensor, class ThrMma>
    static __device__ __forceinline__ void StoreStateFragmentGlobal(StateFragment const& tCrState,
                                                                    GlobalTensor const&  g_state,
                                                                    ThrMma const&        thr_mma,
                                                                    int                  role_tid)
    {
        auto gmem_tiled_copy_C =
            cute::make_tiled_copy_C(cute::Copy_Atom<cute::AutoVectorizingCopy, GlobalStateT>{}, thr_mma);
        auto gmem_thr_copy_C = gmem_tiled_copy_C.get_thread_slice(role_tid);
        auto tCgState        = gmem_thr_copy_C.partition_D(g_state);
        if constexpr (std::is_same_v<GlobalStateT, float>) {
            auto tCrStateView = gmem_thr_copy_C.retile_S(tCrState);
            cute::copy(gmem_tiled_copy_C, tCrStateView, tCgState);
        }
        else {
            static_assert(std::is_same_v<GlobalStateT, __nv_bfloat16>);
            auto tCrPacked     = cute::make_fragment_like<GlobalStateT>(tCrState);
            auto tCrPackedView = gmem_thr_copy_C.retile_S(tCrPacked);
            cute::copy(tCrState, tCrPacked);
            cute::copy(gmem_tiled_copy_C, tCrPackedView, tCgState);
        }
    }

    template<class Fragment, class ThrMma>
    static __device__ __forceinline__ void
    StoreValueFragmentBf16Stsm(Fragment const& fragment, MmaElement* value_pack, ThrMma const& thr_mma, int role_tid)
    {
        auto tCrPack = cute::make_fragment_like<MmaElement>(fragment);
#pragma unroll
        for (int i = 0; i < cute::size(fragment); ++i) {
            tCrPack(i) = CastFromFloat(static_cast<float>(fragment(i)));
        }

        auto s_value_pack = cute::as_position_independent_swizzle_tensor(
            cute::make_tensor(cute::make_smem_ptr(value_pack), VRowLayout()));
        auto smem_tiled_copy_C =
            cute::make_tiled_copy_C(cute::Copy_Atom<cute::SM90_U32x4_STSM_N, MmaElement>{}, thr_mma);
        auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(role_tid);
        auto tCsPack         = smem_thr_copy_C.partition_D(s_value_pack);
        auto tCrPackView     = smem_thr_copy_C.retile_S(tCrPack);
        cute::copy(smem_tiled_copy_C, tCrPackView, tCsPack);
    }

    template<class StateFragment, class ThrMma>
    static __device__ __forceinline__ void StoreStateFragmentBf16Stsm(StateFragment const& tCrState,
                                                                      MmaElement*          state_pack,
                                                                      ThrMma const&        thr_mma,
                                                                      int                  role_tid)
    {
        auto s_state_pack = cute::as_position_independent_swizzle_tensor(
            cute::make_tensor(cute::make_smem_ptr(state_pack), StateCLayout()));
        auto tCrPack = cute::make_fragment_like<MmaElement>(tCrState);
#pragma unroll
        for (int i = 0; i < cute::size(tCrState); ++i) {
            tCrPack(i) = CastFromFloat(static_cast<float>(tCrState(i)));
        }

        auto smem_tiled_copy_C =
            cute::make_tiled_copy_C(cute::Copy_Atom<cute::SM90_U32x4_STSM_N, MmaElement>{}, thr_mma);
        auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(role_tid);
        auto tCsPack         = smem_thr_copy_C.partition_D(s_state_pack);
        auto tCrPackView     = smem_thr_copy_C.retile_S(tCrPack);
        cute::copy(smem_tiled_copy_C, tCrPackView, tCsPack);
    }

    struct SharedStorage {
        // Chunk32 arena. p_stage holds the 32x32 resolvent plus packed W, while q_stage
        // is reused after Q@state for the packed-P/Vd handoff and output TMA store.
        alignas(1024) T q_stage[2][kChunk32Size][kHeadDim];
        alignas(1024) T k_stage[2][kChunk32Size][kHeadDim];
        alignas(128) float gate_stage[2][2][kChunk32Size][4];
        float g[2][kChunk32Size];
        float g_exp[2][kChunk32Size];
        float g_rev_exp[2][kChunk32Size];
        // Keep per-slot barriers as explicit fields so every mbarrier address remains 16-byte aligned.
        alignas(16) cute::uint64_t data_ready_mbar0;
        alignas(16) cute::uint64_t data_ready_mbar1;
        alignas(16) cute::uint64_t data_free_bar0;
        alignas(16) cute::uint64_t data_free_bar1;
        alignas(16) cute::uint64_t q_store_done_bar0;
        alignas(16) cute::uint64_t q_store_done_bar1;
        alignas(16) cute::uint64_t out_ready_bar0;
        alignas(16) cute::uint64_t out_ready_bar1;
        alignas(16) cute::uint64_t update_ready_bar;
        alignas(1024) float vd[kChunk32Size][BlockDv];
        alignas(1024) T state_stage[kHeadDim][BlockDv];
        alignas(1024) float p_stage[2][kChunk32Size][BlockDv];
    };

    static constexpr const char* kUnsupportedMessage =
        "chunk32 fused GDR forward supports only the SM120 bf16 chunked target shape "
        "(int32 q_offsets, bool finished mask, head_dim=128, chunk_size=32, Hv % Hq == 0)";
    static constexpr size_t kSharedBytes = sizeof(SharedStorage);

    static __device__ __forceinline__ void Run(const CUtensorMap* __restrict__ tma_desc_workspace,
                                               const T* __restrict__ q_global,
                                               const T* __restrict__ k_global,
                                               const float* __restrict__ beta,
                                               const int32_t* __restrict__ q_offsets,
                                               const bool* __restrict__ finished,
                                               const int32_t* __restrict__ data_q_offsets,
                                               const int32_t* __restrict__ cp_source_indices,
                                               const int64_t* __restrict__ cp_state_ptrs,
                                               const int64_t* __restrict__ state_ptrs,
                                               int64_t        state_layer_offset,
                                               int            data_sequence_num,
                                               int            token_num,
                                               int            hq,
                                               int            hv,
                                               int            num_head_groups,
                                               int            heads_per_block,
                                               int64_t        beta_stride,
                                               int64_t        beta_batch_stride,
                                               int64_t        q_batch_stride,
                                               int64_t        q_token_stride,
                                               int64_t        q_head_stride,
                                               int64_t        k_batch_stride,
                                               int64_t        k_token_stride,
                                               int64_t        k_head_stride,
                                               unsigned char* smem_raw)
    {
        static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
        static_assert(std::is_same_v<StateT, float> || std::is_same_v<StateT, __nv_bfloat16>,
                      "fused chunk GDR StateT must be float or bfloat16");
        static_assert(kSharedBytes <= kMaxDynamicSharedBytes);
        auto&         smem         = *reinterpret_cast<SharedStorage*>(smem_raw);
        const int     tid          = static_cast<int>(threadIdx.x);
        const int     wg_idx       = cutlass::canonical_warp_group_idx();
        const int     role_tid     = tid % kRoleThreads;
        constexpr int dv_tiles     = kHeadDim / BlockDv;
        const int     linear_block = static_cast<int>(blockIdx.x);
        const int     dv_tile      = linear_block % dv_tiles;
        const int     value_head   = linear_block / dv_tiles % hv;
        const int     batch_id     = linear_block / (dv_tiles * hv);
        const int     dv0          = dv_tile * BlockDv;
        const int     qk_head      = value_head / (hv / hq);
        const int     segment_id   = batch_id;
        const int     sequence_id  = ContextParallel ? cp_source_indices[segment_id] : batch_id;
        if (ContextParallel && (sequence_id < 0 || sequence_id >= data_sequence_num)) {
            return;
        }
        constexpr int StateRegisters  = kStateRegisters;
        constexpr int ValueRegisters  = kValueRegisters;
        constexpr int OutputRegisters = kOutputRegisters;
        const int     seq_start       = q_offsets[batch_id];
        const int     seq_end         = q_offsets[batch_id + 1];
        const int     seq_len         = seq_end - seq_start;
        if (seq_len <= 0) {
            return;
        }
        const int sequence_begin = ContextParallel ? data_q_offsets[sequence_id] : seq_start;
        if (ContextParallel) {
            const int data_seq_end = data_q_offsets[sequence_id + 1];
            if (seq_start < sequence_begin || seq_end > data_seq_end) {
                return;
            }
        }
        const int  token_base           = ContextParallel ? seq_start - sequence_begin : 0;
        const int  physical_batch       = sequence_begin / token_num;
        const int  local_sequence_begin = sequence_begin - physical_batch * token_num;
        const int  qk_tma_head_coord    = qk_head;
        const int  gate_tma_coord       = GateTmaCoord(value_head);
        const int  chunks               = CeilDiv(seq_len, kChunk32Size);
        const auto direct_slices        = MakeFusedGdrTmaDescriptorSlices(tma_desc_workspace, data_sequence_num);
        const auto context_parallel_fused_gdr_slices =
            MakeContextParallelFusedGdrTmaDescriptorSlices(tma_desc_workspace, data_sequence_num);
        const auto* data_desc = ContextParallel ?
                                    context_parallel_fused_gdr_slices.data + sequence_id * kFusedGdrDataDescCount :
                                    direct_slices.data + batch_id * kFusedGdrDataDescCount;
        AcquireAndPrefetchDataTmaDescriptors(data_desc, tid);
        const CUtensorMap* q_desc         = &data_desc[kFusedGdrQDesc];
        const CUtensorMap* q_hi_desc      = &data_desc[kFusedGdrQHiDesc];
        const CUtensorMap* k_desc         = &data_desc[kFusedGdrKDesc];
        const CUtensorMap* k_hi_desc      = &data_desc[kFusedGdrKHiDesc];
        const CUtensorMap* v_desc         = &data_desc[kFusedGdrVDesc];
        const CUtensorMap* g_desc         = &data_desc[kFusedGdrGDesc];
        const CUtensorMap* resolvent_desc = &data_desc[kFusedGdrResolventDesc];
        const CUtensorMap* out_desc       = &data_desc[kFusedGdrOutDesc];

        if (tid == 0) {
            cute::initialize_barrier(smem.data_ready_mbar0, 4);
            cute::initialize_barrier(smem.data_ready_mbar1, 4);
            cute::initialize_barrier(smem.data_free_bar0, 3 * kRoleThreads);
            cute::initialize_barrier(smem.data_free_bar1, 3 * kRoleThreads);
            cute::initialize_barrier(smem.q_store_done_bar0, 1);
            cute::initialize_barrier(smem.q_store_done_bar1, 1);
            cute::initialize_barrier(smem.out_ready_bar0, kRoleThreads);
            cute::initialize_barrier(smem.out_ready_bar1, kRoleThreads);
            cute::initialize_barrier(smem.update_ready_bar, kRoleThreads);
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        if (wg_idx == 3) {
            cutlass::arch::warpgroup_reg_dealloc<kProducerRegisters>();
            const bool store_leader = role_tid == 0;
            const bool q_leader     = role_tid == kWarpThreads;
            const bool k_leader     = role_tid == 2 * kWarpThreads;
            const bool early_leader = role_tid == 3 * kWarpThreads;

            constexpr int kQkHalfTmaBytesPerRow = 64 * static_cast<int>(sizeof(T));
            constexpr int kGateTmaBytesPerRow   = 4 * static_cast<int>(sizeof(float));
            constexpr int kEarlyTmaBytesPerRow =
                BlockDv * static_cast<int>(sizeof(T)) + kChunk32Size * static_cast<int>(sizeof(T));
            constexpr int kTmaBoxRows = kChunk32Size;

            // Q/K/V/A/g share data_ready/data_free for each physical slot. The q producer also
            // waits for q_store_done because the output store reuses q_stage.
            if (role_tid >= 3 * kWarpThreads) {
                for (int load_chunk = 0; load_chunk < chunks; ++load_chunk) {
                    const int data_buf     = load_chunk & 1;
                    const int buffer_phase = (load_chunk >> 1) & 1;
                    if (early_leader && load_chunk >= 2) {
                        auto& data_free_bar = data_buf == 0 ? smem.data_free_bar0 : smem.data_free_bar1;
                        cute::wait_barrier(data_free_bar, buffer_phase ^ 1);
                    }
                    __syncwarp();
                    auto&       data_ready_mbar = data_buf == 0 ? smem.data_ready_mbar0 : smem.data_ready_mbar1;
                    const int   token0          = token_base + load_chunk * kChunk32Size;
                    MmaElement* w_pack =
                        reinterpret_cast<MmaElement*>(&smem.p_stage[data_buf][0][0]) + kChunk32Size * kChunk32Size;

                    if (early_leader) {
                        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                            &data_ready_mbar, kTmaBoxRows * (kEarlyTmaBytesPerRow + kGateTmaBytesPerRow));
                        cute::SM90_TMA_LOAD_3D::copy(g_desc,
                                                     &data_ready_mbar,
                                                     kTmaNoCacheHint,
                                                     &smem.gate_stage[data_buf][0][0][0],
                                                     gate_tma_coord,
                                                     token0,
                                                     0);
                        cute::SM90_TMA_LOAD_4D::copy(
                            v_desc, &data_ready_mbar, kTmaNoCacheHint, w_pack, dv0, value_head, token0, 0);
                        cute::SM90_TMA_LOAD_4D::copy(resolvent_desc,
                                                     &data_ready_mbar,
                                                     kTmaNoCacheHint,
                                                     &smem.p_stage[data_buf][0][0],
                                                     0,
                                                     value_head,
                                                     token0,
                                                     0);
                    }

                    const int  beta_row   = role_tid - 3 * kWarpThreads;
                    const bool beta_valid = beta_row < seq_len - load_chunk * kChunk32Size;
                    float      beta_value = 0.0f;
                    if (beta_valid) {
                        const int64_t beta_offset =
                            static_cast<int64_t>(physical_batch) * beta_batch_stride
                            + static_cast<int64_t>(local_sequence_begin + token0 + beta_row) * beta_stride + value_head;
                        beta_value = beta[beta_offset];
                    }
                    smem.gate_stage[data_buf][1][beta_row][value_head & 3] = beta_value;
                    __syncwarp();
                    if (early_leader) {
                        cute::arrive_barrier(data_ready_mbar);
                    }
                }
            }

            if (q_leader) {
                for (int load_chunk = 0; load_chunk < chunks; ++load_chunk) {
                    const int data_buf     = load_chunk & 1;
                    const int buffer_phase = (load_chunk >> 1) & 1;
                    if (load_chunk >= 2) {
                        auto& q_store_done_bar = data_buf == 0 ? smem.q_store_done_bar0 : smem.q_store_done_bar1;
                        cute::wait_barrier(q_store_done_bar, buffer_phase ^ 1);
                        auto& data_free_bar = data_buf == 0 ? smem.data_free_bar0 : smem.data_free_bar1;
                        cute::wait_barrier(data_free_bar, buffer_phase ^ 1);
                    }
                    auto&     data_ready_mbar = data_buf == 0 ? smem.data_ready_mbar0 : smem.data_ready_mbar1;
                    const int token0          = token_base + load_chunk * kChunk32Size;
                    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                        &data_ready_mbar, 2 * kTmaBoxRows * kQkHalfTmaBytesPerRow);
                    cute::SM90_TMA_LOAD_5D::copy(q_desc,
                                                 &data_ready_mbar,
                                                 kTmaNoCacheHint,
                                                 &smem.q_stage[data_buf][0][0],
                                                 0,
                                                 0,
                                                 qk_tma_head_coord,
                                                 token0,
                                                 0);
                    cute::SM90_TMA_LOAD_5D::copy(q_hi_desc,
                                                 &data_ready_mbar,
                                                 kTmaNoCacheHint,
                                                 &smem.q_stage[data_buf][0][0] + kChunk32Size * 64,
                                                 0,
                                                 0,
                                                 qk_tma_head_coord,
                                                 token0,
                                                 0);
                }
            }

            if (k_leader) {
                for (int load_chunk = 0; load_chunk < chunks; ++load_chunk) {
                    const int data_buf     = load_chunk & 1;
                    const int buffer_phase = (load_chunk >> 1) & 1;
                    if (load_chunk >= 2) {
                        auto& data_free_bar = data_buf == 0 ? smem.data_free_bar0 : smem.data_free_bar1;
                        cute::wait_barrier(data_free_bar, buffer_phase ^ 1);
                    }
                    auto&     data_ready_mbar = data_buf == 0 ? smem.data_ready_mbar0 : smem.data_ready_mbar1;
                    const int token0          = token_base + load_chunk * kChunk32Size;
                    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                        &data_ready_mbar, 2 * kTmaBoxRows * kQkHalfTmaBytesPerRow);
                    cute::SM90_TMA_LOAD_5D::copy(k_desc,
                                                 &data_ready_mbar,
                                                 kTmaNoCacheHint,
                                                 &smem.k_stage[data_buf][0][0],
                                                 0,
                                                 0,
                                                 qk_tma_head_coord,
                                                 token0,
                                                 0);
                    cute::SM90_TMA_LOAD_5D::copy(k_hi_desc,
                                                 &data_ready_mbar,
                                                 kTmaNoCacheHint,
                                                 &smem.k_stage[data_buf][0][0] + kChunk32Size * 64,
                                                 0,
                                                 0,
                                                 qk_tma_head_coord,
                                                 token0,
                                                 0);
                }
            }

            if (store_leader) {
                for (int chunk = 0; chunk < chunks; ++chunk) {
                    const int data_buf      = chunk & 1;
                    const int buffer_phase  = (chunk >> 1) & 1;
                    const int token0        = token_base + chunk * kChunk32Size;
                    auto&     out_ready_bar = data_buf == 0 ? smem.out_ready_bar0 : smem.out_ready_bar1;
                    cute::wait_barrier(out_ready_bar, buffer_phase);

                    cute::tma_store_fence();
                    MmaElement* out_stage = reinterpret_cast<MmaElement*>(&smem.q_stage[data_buf][0][0]);
                    cute::SM90_TMA_STORE_4D::copy(out_desc, out_stage, dv0, value_head, token0, 0);
                    cute::tma_store_arrive();
                    cute::tma_store_wait<0>();

                    auto& q_store_done_bar = data_buf == 0 ? smem.q_store_done_bar0 : smem.q_store_done_bar1;
                    cute::arrive_barrier(q_store_done_bar);
                }
            }

            return;
        }

        if (wg_idx == 0) {
            cutlass::arch::warpgroup_reg_alloc<StateRegisters>();
            using Element = MmaElement;
            using Mma     = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
                                       cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                       cute::Tile<cute::Underscore, cute::Int<32>, cute::Underscore>>;

            Mma  mma;
            auto thr_mma           = mma.get_thread_slice(role_tid);
            auto state_tile_layout = cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<BlockDv>{}),
                                                       cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{}));
            auto g_state_for_fragment =
                cute::make_tensor(cute::make_gmem_ptr(static_cast<float*>(nullptr)), state_tile_layout);
            auto tCgStateForFragment = thr_mma.partition_C(g_state_for_fragment);
            auto tCrState            = thr_mma.make_fragment_C(tCgStateForFragment);
            auto c_state             = cute::make_identity_tensor(cute::shape(g_state_for_fragment));
            auto tCcState            = thr_mma.partition_C(c_state);
            if constexpr (ContextParallel) {
                auto* cp_state_base = reinterpret_cast<float*>(static_cast<uintptr_t>(cp_state_ptrs[segment_id]));
                auto* state_base    = cp_state_base + static_cast<int64_t>(value_head) * kHeadDim * kHeadDim + dv0;
                auto  g_state       = cute::make_tensor(cute::make_gmem_ptr(state_base), state_tile_layout);
                LoadStateFragmentGlobal<float>(tCrState, g_state, thr_mma, role_tid);
            }
            else {
                auto* state_base =
                    GroupedStateBase(
                        state_ptrs, batch_id, value_head, num_head_groups, heads_per_block, state_layer_offset)
                    + dv0;
                auto g_state = cute::make_tensor(cute::make_gmem_ptr(state_base), state_tile_layout);
                LoadStateFragmentGlobal<StateT>(tCrState, g_state, thr_mma, role_tid);
            }
            MmaSyncNamed<kBarrierStateUpdate>();

            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int token0       = chunk * kChunk32Size;
                const int remaining    = seq_len - token0;
                const int valid        = remaining < kChunk32Size ? remaining : kChunk32Size;
                const int last_row     = valid - 1;
                const int data_buf     = chunk & 1;
                const int buffer_phase = (chunk >> 1) & 1;
                const int chunk_phase  = chunk & 1;

                auto& data_ready_mbar = data_buf == 0 ? smem.data_ready_mbar0 : smem.data_ready_mbar1;
                cute::wait_barrier(data_ready_mbar, buffer_phase);

                const int gate_lane     = value_head & 3;
                const int gate_warp     = role_tid / kWarpThreads;
                const int gate_warp_tid = role_tid % kWarpThreads;
                auto      s_a_smem      = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.p_stage[data_buf][0][0])), SquareLayout());
                const float last_g_value = smem.gate_stage[data_buf][0][last_row][gate_lane];
                if (gate_warp_tid < kGateRowsPerWarp) {
#pragma unroll
                    for (int pass = 0; pass < kGatePasses; ++pass) {
                        const int   row = pass * kGateWriterThreads + gate_warp * kGateRowsPerWarp + gate_warp_tid;
                        const bool  row_valid         = row < valid;
                        const float stored_g          = smem.gate_stage[data_buf][0][row][gate_lane];
                        const float g_value           = row_valid ? stored_g : last_g_value;
                        const float g_exp             = row_valid ? FastExp(g_value) : 0.0f;
                        smem.g_exp[data_buf][row]     = g_exp;
                        smem.g[data_buf][row]         = g_value;
                        smem.g_rev_exp[data_buf][row] = row_valid ? FastExp(last_g_value - g_value) : 0.0f;
                    }
                }
                const float state_decay = FastExp(last_g_value);
                Element*    state_pack  = reinterpret_cast<Element*>(&smem.state_stage[0][0]);
                StoreStateFragmentBf16Stsm(tCrState, state_pack, thr_mma, role_tid);
                StatePackArrive();

                DecayStateFragment(tCrState, state_decay);

                cute::wait_barrier(smem.update_ready_bar, chunk_phase);

                auto s_k_t =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.k_stage[data_buf][0][0])),
                                      SplitQkTransposedLayout());
                auto s_vn_t =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.vd[0][0])), VTLayout());

                StateUpdateFragmentFromScaledVd(role_tid, mma, s_k_t, s_vn_t, tCrState);
                MmaSyncNamed<kBarrierStateUpdate>();
                auto& data_free_bar = data_buf == 0 ? smem.data_free_bar0 : smem.data_free_bar1;
                cute::arrive_barrier(data_free_bar);
                const bool sequence_terminal = !ContextParallel || seq_end == data_q_offsets[sequence_id + 1];
                const bool store_final_state = chunk == chunks - 1 && sequence_terminal && !finished[batch_id];
                if (store_final_state) {
                    auto* state_base =
                        GroupedStateBase(
                            state_ptrs, sequence_id, value_head, num_head_groups, heads_per_block, state_layer_offset)
                        + dv0;
                    auto g_state = cute::make_tensor(cute::make_gmem_ptr(state_base), state_tile_layout);
                    StoreStateFragmentGlobal<StateT>(tCrState, g_state, thr_mma, role_tid);
                    MmaSyncNamed<kBarrierStateUpdate>();
                }
            }
            return;
        }
        else if (wg_idx == 1) {
            cutlass::arch::warpgroup_reg_alloc<ValueRegisters>();
            using Element = MmaElement;

            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int data_buf     = chunk & 1;
                const int buffer_phase = (chunk >> 1) & 1;

                auto s_k_smem = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.k_stage[data_buf][0][0])), SplitQkLayout());
                auto s_a_smem = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.p_stage[data_buf][0][0])), SquareLayout());
                Element* w_pack =
                    reinterpret_cast<Element*>(&smem.p_stage[data_buf][0][0]) + kChunk32Size * kChunk32Size;
                auto& data_ready_mbar = data_buf == 0 ? smem.data_ready_mbar0 : smem.data_ready_mbar1;
                cute::wait_barrier(data_ready_mbar, buffer_phase);

                StatePackValueSync();

                using Mma = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
                                           cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                           cute::Tile<cute::Underscore, cute::Int<32>, cute::Underscore>>;

                Mma      mma;
                auto     thr_mma     = mma.get_thread_slice(role_tid);
                Element* packed_base = reinterpret_cast<Element*>(&smem.q_stage[data_buf][0][0]);
                Element* packed_vd   = packed_base + kVdOffset;
                auto     s_w_row     = cute::make_tensor(cute::make_smem_ptr(w_pack), VRowLayout());

                auto s_state = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.state_stage[0][0])), StateTLayout());
                auto s_k_state = cute::make_tensor(
                    cute::make_smem_ptr(&smem.vd[0][0]),
                    cute::make_layout(cute::make_shape(cute::Int<kChunk32Size>{}, cute::Int<BlockDv>{}),
                                      cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{})));
                auto c_k_state = cute::make_identity_tensor(cute::shape(s_k_state));

                auto     tCsC    = thr_mma.partition_C(s_k_state);
                auto     tCcC    = thr_mma.partition_C(c_k_state);
                auto     tCrC    = thr_mma.make_fragment_C(tCsC);
                auto     tCrW    = cute::make_fragment_like<Element>(tCrC);
                Element* vn_bf16 = reinterpret_cast<Element*>(&smem.vd[0][0]);
                cute::clear(tCrC);
                cute::cooperative_gemm(role_tid,
                                       mma,
                                       s_k_smem,
                                       s_state,
                                       tCrC,
                                       cute::identity{},
                                       cute::identity{},
                                       cute::SM75_U32x2_LDSM_N{},
                                       cute::SM75_U16x8_LDSM_T{});
#pragma unroll
                for (int i = 0; i < cute::size(tCrC); ++i) {
                    auto      coord = tCcC(i);
                    const int row   = cute::get<0>(coord);
                    const int dv    = cute::get<1>(coord);
                    tCrW(i)         = s_w_row(row, dv);
                }
                MmaSyncNamed<kBarrierValueU>();
                // Sequence-clipped TMA loads and the zero beta/gate tail keep every
                // downstream partial-chunk fragment algebraically zero for finite inputs.
#pragma unroll
                for (int i = 0; i < cute::size(tCrC); ++i) {
                    const int   row     = cute::get<0>(tCcC(i));
                    const float v_value = static_cast<float>(tCrW(i));
                    const float delta   = v_value - smem.g_exp[data_buf][row] * tCrC(i);
                    tCrW(i)             = Element(CastFromFloat(delta));
                }
                StoreValueFragmentBf16Stsm(tCrW, w_pack, thr_mma, role_tid);
                MmaSyncNamed<kBarrierValueU>();
                // Transformed A is independent of the state snapshot. Acquire it
                // without waiting for WG2's Q@state reads to finish.
                AgReadySync();

                auto s_w = cute::make_tensor(cute::make_smem_ptr(w_pack), VTLayout());
                cute::clear(tCrC);
                cute::cooperative_gemm(role_tid,
                                       mma,
                                       s_a_smem,
                                       s_w,
                                       tCrC,
                                       cute::identity{},
                                       cute::identity{},
                                       cute::SM75_U32x2_LDSM_N{},
                                       cute::SM75_U16x8_LDSM_T{});
                auto& data_free_bar = data_buf == 0 ? smem.data_free_bar0 : smem.data_free_bar1;
                cute::arrive_barrier(data_free_bar);

                // WG2 has stopped reading Q before WG1 reuses the disjoint packed-Vd
                // region of q_stage. WG2 packs P concurrently with this STSM.
                PackedVdSync();
                StoreValueFragmentBf16Stsm(tCrC, packed_vd, thr_mma, role_tid);
                // Publish packed Vd before WG2 starts the local output GEMM.
                PackedVdSync();

#pragma unroll
                for (int i = 0; i < cute::size(tCrC); ++i) {
                    const int   row = cute::get<0>(tCcC(i));
                    const float vn  = smem.g_rev_exp[data_buf][row] * static_cast<float>(tCrC(i));
                    tCrW(i)         = Element(CastFromFloat(vn));
                }
                StoreValueFragmentBf16Stsm(tCrW, vn_bf16, thr_mma, role_tid);
                cute::arrive_barrier(smem.update_ready_bar);
            }
            return;
        }
        else if (wg_idx == 2) {
            using Element = MmaElement;

            cutlass::arch::warpgroup_reg_alloc<OutputRegisters>();
            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int data_buf     = chunk & 1;
                const int buffer_phase = (chunk >> 1) & 1;

                auto s_q_smem = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.q_stage[data_buf][0][0])), SplitQkLayout());
                auto s_k_smem = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.k_stage[data_buf][0][0])), SplitQkLayout());
                auto s_a_smem = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.p_stage[data_buf][0][0])), SquareLayout());
                auto& data_ready_mbar = data_buf == 0 ? smem.data_ready_mbar0 : smem.data_ready_mbar1;
                cute::wait_barrier(data_ready_mbar, buffer_phase);

                StatePackOutputSync();

                const int gate_lane = value_head & 3;
                using Mma           = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
                                           cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                           cute::Tile<cute::Underscore, cute::Int<32>, cute::Underscore>>;

                Mma      mma;
                auto     thr_mma     = mma.get_thread_slice(role_tid);
                Element* packed_base = reinterpret_cast<Element*>(&smem.q_stage[data_buf][0][0]);
                Element* packed_p    = packed_base + kPOffset;
                Element* packed_vd   = packed_base + kVdOffset;

                auto s_p_float = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<float*>(&smem.q_stage[data_buf][0][0])),
                    cute::make_layout(cute::make_shape(cute::Int<kChunk32Size>{}, cute::Int<kChunk32Size>{}),
                                      cute::make_stride(cute::Int<kChunk32Size>{}, cute::Int<1>{})));
                auto s_packed_p = cute::make_tensor(cute::make_smem_ptr(packed_p), P128BRowLayout());
                auto c_p        = cute::make_identity_tensor(cute::shape(s_p_float));
                auto tCsP       = thr_mma.partition_C(s_p_float);
                auto tCcP       = thr_mma.partition_C(c_p);
                auto tCrGRel    = thr_mma.make_fragment_C(tCsP);

#pragma unroll
                for (int i = 0; i < cute::size(tCrGRel); ++i) {
                    auto        coord    = tCcP(i);
                    const int   row      = cute::get<0>(coord);
                    const int   col      = cute::get<1>(coord);
                    float       g_rel    = 0.0f;
                    const bool  lower    = col <= row;
                    const float beta_col = smem.gate_stage[data_buf][1][col][gate_lane];
                    g_rel                = lower ? FastExp(smem.g[data_buf][row] - smem.g[data_buf][col]) : 0.0f;
                    const float a_value  = static_cast<float>(s_a_smem(row, col));
                    const float ag_value = a_value * g_rel * beta_col;
                    s_a_smem(row, col)   = Element(CastFromFloat(ag_value));
                    tCrGRel(i)           = g_rel;
                }
                AgReadyArrive();

                auto s_q_state_c = cute::make_tensor(
                    cute::make_smem_ptr(&smem.p_stage[data_buf][0][0]),
                    cute::make_layout(cute::make_shape(cute::Int<kChunk32Size>{}, cute::Int<BlockDv>{}),
                                      cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{})));
                auto tCsQState = thr_mma.partition_C(s_q_state_c);
                auto c_q_state = cute::make_identity_tensor(cute::shape(s_q_state_c));
                auto tCcQState = thr_mma.partition_C(c_q_state);
                auto tCrQState = thr_mma.make_fragment_C(tCsQState);
                auto s_state   = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.state_stage[0][0])), StateTLayout());

                cute::clear(tCrQState);
                cute::cooperative_gemm(role_tid,
                                       mma,
                                       s_q_smem,
                                       s_state,
                                       tCrQState,
                                       cute::identity{},
                                       cute::identity{},
                                       cute::SM75_U32x2_LDSM_N{},
                                       cute::SM75_U16x8_LDSM_T{});
#pragma unroll
                for (int i = 0; i < cute::size(tCrQState); ++i) {
                    const int row = cute::get<0>(tCcQState(i));
                    tCrQState(i)  = tCrQState(i) * kHeadScale * smem.g_exp[data_buf][row];
                }

                auto tCrC = thr_mma.make_fragment_C(tCsP);
                cute::clear(tCrC);
                cute::cooperative_gemm(role_tid,
                                       mma,
                                       s_q_smem,
                                       s_k_smem,
                                       tCrC,
                                       cute::identity{},
                                       cute::identity{},
                                       cute::SM75_U32x2_LDSM_N{},
                                       cute::SM75_U32x2_LDSM_N{});
                MmaSyncNamed<kBarrierOutputP>();
                // Release the packed-Vd region after all Q reads have completed.
                PackedVdSync();
#pragma unroll
                for (int i = 0; i < cute::size(tCrC); i += 2) {
                    auto        coord    = tCcP(i);
                    const int   row      = cute::get<0>(coord);
                    const int   col      = cute::get<1>(coord);
                    const int   next_col = col + 1;
                    const float p0       = col <= row ? kHeadScale * tCrGRel(i) * tCrC(i) : 0.0f;
                    const float p1       = next_col <= row ? kHeadScale * tCrGRel(i + 1) * tCrC(i + 1) : 0.0f;
                    // The C-fragment layout emits adjacent-column pairs, and even columns
                    // stay contiguous under Swizzle<3,3,3>.
                    StoreBf16Pair(&s_packed_p(row, col), make_float2(p0, p1));
                }
                // Acquire WG1's packed Vd after this warp group has finished packing P.
                PackedVdSync();

                auto  s_packed_p_local = cute::make_tensor(cute::make_smem_ptr(packed_p), P128BRowLayout());
                auto  s_packed_vd      = cute::make_tensor(cute::make_smem_ptr(packed_vd), VTLayout());
                auto* out_stage        = reinterpret_cast<Element*>(&smem.q_stage[data_buf][0][0]);
                auto  s_out            = cute::make_tensor(cute::make_smem_ptr(out_stage), VRowLayout());

                auto tCsOut   = thr_mma.partition_C(s_out);
                auto tCrLocal = thr_mma.make_fragment_C(tCsOut);
                cute::clear(tCrLocal);
                cute::cooperative_gemm(role_tid,
                                       mma,
                                       s_packed_p_local,
                                       s_packed_vd,
                                       tCrLocal,
                                       cute::identity{},
                                       cute::identity{},
                                       cute::SM75_U32x2_LDSM_N{},
                                       cute::SM75_U16x8_LDSM_T{});
                MmaSyncNamed<kBarrierOutputLocal>();

#pragma unroll
                for (int i = 0; i < cute::size(tCrLocal); ++i) {
                    const float out_value = tCrQState(i) + tCrLocal(i);
                    tCsOut(i)             = Element(CastFromFloat(out_value));
                }
                auto& data_free_bar = data_buf == 0 ? smem.data_free_bar0 : smem.data_free_bar1;
                cute::arrive_barrier(data_free_bar);
                auto& out_ready_bar = data_buf == 0 ? smem.out_ready_bar0 : smem.out_ready_bar1;
                cute::arrive_barrier(out_ready_bar);
            }
            return;
        }
    }
};

template<class T, class StateT, int BlockDv, bool ContextParallel>
__global__
    __launch_bounds__(Sm120FusedGdrFwd<T, StateT, BlockDv, ContextParallel>::kThreads,
                      Sm120FusedGdrFwd<T, StateT, BlockDv, ContextParallel>::
                          kMinBlocks) void Sm120FusedGdrFwdKernel(const CUtensorMap* __restrict__ tma_desc_workspace,
                                                                  const T* __restrict__ q_global,
                                                                  const T* __restrict__ k_global,
                                                                  const float* __restrict__ beta,
                                                                  const int32_t* __restrict__ q_offsets,
                                                                  const bool* __restrict__ finished,
                                                                  const int32_t* __restrict__ data_q_offsets,
                                                                  const int32_t* __restrict__ cp_source_indices,
                                                                  const int64_t* __restrict__ cp_state_ptrs,
                                                                  const int64_t* __restrict__ state_ptrs,
                                                                  int64_t state_layer_offset,
                                                                  int     data_sequence_num,
                                                                  int     token_num,
                                                                  int     hq,
                                                                  int     hv,
                                                                  int     num_head_groups,
                                                                  int     heads_per_block,
                                                                  int64_t beta_stride,
                                                                  int64_t beta_batch_stride,
                                                                  int64_t q_batch_stride,
                                                                  int64_t q_token_stride,
                                                                  int64_t q_head_stride,
                                                                  int64_t k_batch_stride,
                                                                  int64_t k_token_stride,
                                                                  int64_t k_head_stride)
{
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    Sm120FusedGdrFwd<T, StateT, BlockDv, ContextParallel>::Run(tma_desc_workspace,
                                                               q_global,
                                                               k_global,
                                                               beta,
                                                               q_offsets,
                                                               finished,
                                                               data_q_offsets,
                                                               cp_source_indices,
                                                               cp_state_ptrs,
                                                               state_ptrs,
                                                               state_layer_offset,
                                                               data_sequence_num,
                                                               token_num,
                                                               hq,
                                                               hv,
                                                               num_head_groups,
                                                               heads_per_block,
                                                               beta_stride,
                                                               beta_batch_stride,
                                                               q_batch_stride,
                                                               q_token_stride,
                                                               q_head_stride,
                                                               k_batch_stride,
                                                               k_token_stride,
                                                               k_head_stride,
                                                               smem_raw);
}

template<class StateT, int BlockDv, bool ContextParallel>
void SetFusedGdrFwdSharedMemoryLimit()
{
    static_assert(std::is_same_v<StateT, float> || std::is_same_v<StateT, __nv_bfloat16>,
                  "fused chunk GDR StateT must be float or bfloat16");
    using Kernel = Sm120FusedGdrFwd<__nv_bfloat16, StateT, BlockDv, ContextParallel>;
    static const cudaError_t status =
        cudaFuncSetAttribute(Sm120FusedGdrFwdKernel<__nv_bfloat16, StateT, BlockDv, ContextParallel>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(Kernel::kSharedBytes));
    TM_CUDA_CHECK(status);
}

template<class StateT, int BlockDv, bool ContextParallel>
void LaunchSm120FusedGdrFwd(const core::Tensor& q,
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
    static_assert(std::is_same_v<StateT, float> || std::is_same_v<StateT, __nv_bfloat16>,
                  "fused chunk GDR StateT must be float or bfloat16");
    using Kernel = Sm120FusedGdrFwd<__nv_bfloat16, StateT, BlockDv, ContextParallel>;
    static_cast<void>(q);
    static_cast<void>(k);
    static_cast<void>(v);
    static_cast<void>(g_cumsum);
    static_cast<void>(resolvent);
    static_cast<void>(out);

    const int   descriptor_sequence_num = ContextParallel ? data_sequence_num : problem.sequence_num;
    const auto* q_offsets_ptr           = q_offsets.data<int32_t>();
    const auto* finished_ptr            = finished.data<bool>();
    const auto* data_q_offsets_ptr      = ContextParallel ? data_q_offsets->data<int32_t>() : nullptr;
    const auto* cp_source_indices_ptr   = ContextParallel ? cp_source_indices->data<int32_t>() : nullptr;
    const auto* cp_state_ptrs_ptr       = ContextParallel ? cp_state_ptrs->data<int64_t>() : nullptr;

    constexpr int block_dv     = BlockDv;
    const int     dv_tiles     = (kHeadDim + block_dv - 1) / block_dv;
    const dim3    grid         = dim3(problem.sequence_num * problem.hv * dv_tiles);
    const dim3    block        = dim3(Kernel::kThreads);
    auto*         tma_desc_ptr = reinterpret_cast<CUtensorMap*>(tma_desc_workspace);

    SetFusedGdrFwdSharedMemoryLimit<StateT, block_dv, ContextParallel>();
    Sm120FusedGdrFwdKernel<__nv_bfloat16, StateT, block_dv, ContextParallel>
        <<<grid, block, Kernel::kSharedBytes, stream>>>(tma_desc_ptr,
                                                        q.data<__nv_bfloat16>(),
                                                        k.data<__nv_bfloat16>(),
                                                        beta.data<float>(),
                                                        q_offsets_ptr,
                                                        finished_ptr,
                                                        data_q_offsets_ptr,
                                                        cp_source_indices_ptr,
                                                        cp_state_ptrs_ptr,
                                                        reinterpret_cast<const int64_t*>(state_ptrs.raw_data()),
                                                        state_layer_offset,
                                                        descriptor_sequence_num,
                                                        problem.token_num,
                                                        problem.hq,
                                                        problem.hv,
                                                        problem.num_head_groups,
                                                        problem.heads_per_block,
                                                        problem.beta_stride,
                                                        problem.beta_batch_stride,
                                                        q.stride(0),
                                                        q.stride(1),
                                                        q.stride(2),
                                                        k.stride(0),
                                                        k.stride(1),
                                                        k.stride(2));
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
