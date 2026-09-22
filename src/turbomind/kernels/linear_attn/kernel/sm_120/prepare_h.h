// Inspired by
// https://github.com/QwenLM/FlashQLA/blob/60f81453143e724bcaf3fc7921e71e7328f6ebcd/flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_h.py

#pragma once

#include "src/turbomind/kernels/linear_attn/kernel/sm_120/common.h"

namespace turbomind::linear_attn::delta_rule {
namespace {

template<class T, int BlockDv>
struct Sm120FusedGdrH {
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    static_assert(BlockDv == kFusedGdrHBlockDv);

    using MmaElement = cute::bfloat16_t;
    using MmaAtom    = cute::SM80_16x8x16_F32BF16BF16F32_TN;

    static constexpr int      kRoleThreads              = 128;
    static constexpr int      kConsumerThreads          = 3 * kRoleThreads;
    static constexpr int      kThreads                  = kConsumerThreads + kRoleThreads;
    static constexpr int      kWarpThreads              = 32;
    static constexpr int      kGateRowsPerWarp          = 8;
    static constexpr int      kGateWriterThreads        = (kRoleThreads / kWarpThreads) * kGateRowsPerWarp;
    static constexpr int      kGatePasses               = kChunk32Size / kGateWriterThreads;
    static constexpr int      kMinBlocks                = 1;
    static constexpr int      kProducerRegisters        = 24;
    static constexpr int      kStateRegisters           = 144;
    static constexpr int      kValueRegisters           = 144;
    static constexpr int      kFusedGdrHDataDescCount   = 4;
    static constexpr int      kFusedGdrHTensorDescCount = 2;
    static constexpr int      kFusedGdrGateTmaHeads     = 4;
    static constexpr uint64_t kTmaNoCacheHint           = 0;
    static constexpr size_t   kMaxDynamicSharedBytes    = 102400 - 1024;

    enum TmaDescIndex : int
    {
        kFusedGdrHKDesc = 0,
        kFusedGdrHVDesc,
        kFusedGdrHGDesc,
        kFusedGdrHResolventDesc,
        kFusedGdrHSegmentStateDesc,
        kFusedGdrHSegmentMDesc,
    };

    template<class TensorMapPtr>
    struct FusedGdrHTmaDescriptorSlices {
        TensorMapPtr data{};
        TensorMapPtr segment_state{};
        TensorMapPtr segment_m{};
    };

    template<class TensorMapPtr>
    static CUTE_HOST_DEVICE FusedGdrHTmaDescriptorSlices<TensorMapPtr>
                            MakeFusedGdrHTmaDescriptorSlices(TensorMapPtr base, int sequence_num)
    {
        auto* segment_state = base + sequence_num * kFusedGdrHDataDescCount;
        return {base, segment_state, segment_state + 1};
    }

    static __device__ __forceinline__ int GateTmaCoord(int value_head)
    {
        return (value_head / kFusedGdrGateTmaHeads) * kFusedGdrGateTmaHeads;
    }

    static constexpr int kBarrierValueU         = 2;
    static constexpr int kBarrierHVdReady       = 5;
    static constexpr int kBarrierHStateReadDone = 6;
    static constexpr int kBarrierMVdReady       = 7;

    static_assert(kThreads == 512);
    static_assert(kGateWriterThreads == kChunk32Size);
    static_assert(kGatePasses == 1);
    static_assert(kBarrierMVdReady + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount
                  < cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers);

    template<int BarrierId>
    static __device__ __forceinline__ void MmaSyncNamed()
    {
        cutlass::arch::NamedBarrier::sync(kRoleThreads, BarrierId);
    }

    static CUTE_HOST_DEVICE constexpr auto QkLayout()
    {
        return cute::composition(
            cute::Swizzle<3, 3, 3>{},
            cute::Layout<cute::Shape<cute::_32, cute::_128>, cute::Stride<cute::_128, cute::_1>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto QkTransposedLayout()
    {
        return cute::composition(
            cute::Swizzle<3, 3, 3>{},
            cute::Layout<cute::Shape<cute::_128, cute::_32>, cute::Stride<cute::_1, cute::_128>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto SquareLayout()
    {
        return cute::composition(cute::Swizzle<2, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::_32, cute::_32>, cute::Stride<cute::_32, cute::_1>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto VRowLayout()
    {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::_32, cute::_64>, cute::Stride<cute::_64, cute::_1>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto VTLayout()
    {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::_64, cute::_32>, cute::Stride<cute::_1, cute::_64>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto StateTLayout()
    {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::Int<BlockDv>, cute::Int<kHeadDim>>,
                                              cute::Stride<cute::_1, cute::Int<BlockDv>>>{});
    }

    static CUTE_HOST_DEVICE constexpr auto StateCLayout()
    {
        return cute::composition(cute::Swizzle<3, 3, 3>{},
                                 cute::Layout<cute::Shape<cute::Int<kHeadDim>, cute::Int<BlockDv>>,
                                              cute::Stride<cute::Int<BlockDv>, cute::_1>>{});
    }

    static_assert(cute::cosize_v<decltype(QkLayout())> == kChunk32Size * kHeadDim);
    static_assert(cute::cosize_v<decltype(QkTransposedLayout())> == kChunk32Size * kHeadDim);
    static_assert(cute::cosize_v<decltype(SquareLayout())> == kChunk32Size * kChunk32Size);
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

    static constexpr const char* kUnsupportedMessage = "chunk32 fused GDR H requires fixed CP segment tensors";

    struct alignas(1024) EarlyStage {
        alignas(1024) T a[kChunk32Size][kChunk32Size];
        alignas(1024) T v[kChunk32Size][BlockDv];
        alignas(1024) float gate_stage[2][kChunk32Size][4];
        alignas(1024) float g[kChunk32Size];
        alignas(1024) float g_exp[kChunk32Size];
    };

    struct alignas(1024) SharedStorage {
        alignas(1024) T k_stage[2][kChunk32Size][kHeadDim];
        alignas(1024) EarlyStage early[2];
        alignas(1024) T vd[kHeadDim][BlockDv];
        alignas(16) cute::uint64_t state_ready_mbar;
        alignas(16) cute::uint64_t early_ready_mbar[2];
        alignas(16) cute::uint64_t early_free_bar[2];
        alignas(16) cute::uint64_t k_ready_mbar[2];
        alignas(16) cute::uint64_t k_free_bar[2];
        alignas(16) cute::uint64_t h_final_ready_bar;
        alignas(16) cute::uint64_t m_final_ready_bar;
        alignas(16) cute::uint64_t h_pack_ready_bar;
        alignas(16) cute::uint64_t m_pack_ready_bar;
        alignas(16) cute::uint64_t scratch_free_bar;
        alignas(16) cute::uint64_t a_read_done_bar;
    };

    static constexpr size_t kSharedBytes = sizeof(SharedStorage);

    struct WarmupMetadata {
        int chunks;
        int fallback;
    };

    static __device__ __forceinline__ WarmupMetadata
    ComputeWarmupMetadata(const float* __restrict__ g_cumsum,
                          const int32_t* __restrict__ q_offsets,
                          const int32_t* __restrict__ cp_source_indices,
                          const int32_t* __restrict__ cp_q_offsets,
                          int     segment_id,
                          int     value_head,
                          int     token_num,
                          int     sequence_num,
                          int64_t gate_stride,
                          int64_t gate_batch_stride,
                          bool    allow_warmup)
    {
        WarmupMetadata out{0, 0};
        const int      sequence_id = cp_source_indices[segment_id];
        if (sequence_id < 0 || sequence_id >= sequence_num) {
            return out;
        }
        const int segment_begin = cp_q_offsets[segment_id];
        const int segment_end   = cp_q_offsets[segment_id + 1];
        if (segment_end == q_offsets[sequence_id + 1]) {
            return out;
        }

        constexpr float kWarmupThreshold = -10.0f;
        const int       segment_chunks   = (segment_end - segment_begin) / kChunk32Size;
        float           gate_sum         = 0.0f;
        out.chunks                       = segment_chunks;
        out.fallback                     = 1;
        if (!allow_warmup) {
            return out;
        }
        for (int chunk = 0; chunk < segment_chunks; ++chunk) {
            const int     flat_token     = segment_end - chunk * kChunk32Size - 1;
            const int     physical_batch = flat_token / token_num;
            const int     local_token    = flat_token - physical_batch * token_num;
            const int64_t gate_offset    = static_cast<int64_t>(physical_batch) * gate_batch_stride
                                        + static_cast<int64_t>(local_token) * gate_stride + value_head;
            gate_sum += g_cumsum[gate_offset];
            if (gate_sum < kWarmupThreshold) {
                out.chunks   = chunk + 1;
                out.fallback = 0;
                break;
            }
        }
        return out;
    }

    static __device__ __forceinline__ void Run(const CUtensorMap* __restrict__ tma_desc_workspace,
                                               float* __restrict__ segment_state,
                                               float* __restrict__ segment_m,
                                               const float* __restrict__ g_cumsum,
                                               const float* __restrict__ beta,
                                               const int32_t* __restrict__ q_offsets,
                                               const int32_t* __restrict__ cp_source_indices,
                                               const int32_t* __restrict__ cp_q_offsets,
                                               const bool* __restrict__ cp_finished,
                                               bool* __restrict__ cp_fallback,
                                               int            sequence_num,
                                               int            token_num,
                                               int            hq,
                                               int            hv,
                                               int64_t        gate_stride,
                                               int64_t        gate_batch_stride,
                                               int64_t        beta_stride,
                                               int64_t        beta_batch_stride,
                                               bool           allow_warmup,
                                               unsigned char* smem_raw)
    {
        static_assert(BlockDv == kFusedGdrHBlockDv);
        static_assert(kSharedBytes <= kMaxDynamicSharedBytes);
        auto&      smem            = *reinterpret_cast<SharedStorage*>(smem_raw);
        const int  tid             = static_cast<int>(threadIdx.x);
        const int  wg_idx          = cutlass::canonical_warp_group_idx();
        const int  role_tid        = tid % kRoleThreads;
        const bool producer_leader = wg_idx == 3 && role_tid == 0;
        const int  segment_id      = static_cast<int>(blockIdx.x);
        const int  sequence_id     = cp_source_indices[segment_id];
        if (sequence_id < 0 || sequence_id >= sequence_num) {
            return;
        }
        constexpr int kDvTilesPerHead = kHeadDim / BlockDv;
        const int     head_tile       = static_cast<int>(blockIdx.y);
        const int     value_head      = head_tile / kDvTilesPerHead;
        const int     dv_tile         = head_tile - value_head * kDvTilesPerHead;
        const int     dv0             = dv_tile * BlockDv;
        auto&         warmup          = *reinterpret_cast<WarmupMetadata*>(smem_raw);
        if (tid == 0) {
            warmup                                                          = ComputeWarmupMetadata(g_cumsum,
                                           q_offsets,
                                           cp_source_indices,
                                           cp_q_offsets,
                                           segment_id,
                                           value_head,
                                           token_num,
                                           sequence_num,
                                           gate_stride,
                                           gate_batch_stride,
                                           allow_warmup);
            cp_fallback[static_cast<int64_t>(segment_id) * hv + value_head] = warmup.fallback != 0;
        }
        __syncthreads();
        const int  warmup_chunks    = warmup.chunks;
        const bool needs_transition = warmup.fallback != 0;
        __syncthreads();
        if (warmup_chunks <= 0) {
            return;
        }
        const int sequence_begin  = q_offsets[sequence_id];
        const int segment_begin   = cp_q_offsets[segment_id];
        const int segment_end     = cp_q_offsets[segment_id + 1];
        const int requested_begin = segment_end - warmup_chunks * kChunk32Size;
        const int warmup_begin    = requested_begin > segment_begin ? requested_begin : segment_begin;
        const int seq_len         = segment_end - warmup_begin;
        if (seq_len <= 0) {
            return;
        }
        const int   token_base             = warmup_begin - sequence_begin;
        const int   physical_batch         = sequence_begin / token_num;
        const int   local_sequence_begin   = sequence_begin - physical_batch * token_num;
        const int   qk_head                = value_head / (hv / hq);
        const int   gate_tma_coord         = GateTmaCoord(value_head);
        const int   qk_tma_head_coord      = qk_head;
        const int   chunks                 = warmup_chunks;
        const auto  slices                 = MakeFusedGdrHTmaDescriptorSlices(tma_desc_workspace, sequence_num);
        const auto* data_desc              = slices.data + sequence_id * kFusedGdrHDataDescCount;
        const auto* k_tma_desc             = &data_desc[kFusedGdrHKDesc];
        const auto* v_tma_desc             = &data_desc[kFusedGdrHVDesc];
        const auto* g_tma_desc             = &data_desc[kFusedGdrHGDesc];
        const auto* resolvent_tma_desc     = &data_desc[kFusedGdrHResolventDesc];
        const auto* segment_state_tma_desc = slices.segment_state;
        const auto* segment_m_tma_desc     = slices.segment_m;

        if (tid == 0) {
            for (int idx = 0; idx < kFusedGdrHDataDescCount; ++idx) {
                cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(&data_desc[idx]));
            }
            cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(segment_state_tma_desc));
            cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(segment_m_tma_desc));
            for (int idx = 0; idx < kFusedGdrHDataDescCount; ++idx) {
                cute::prefetch_tma_descriptor(&data_desc[idx]);
            }
            cute::prefetch_tma_descriptor(segment_state_tma_desc);
            cute::prefetch_tma_descriptor(segment_m_tma_desc);
            cute::initialize_barrier(smem.state_ready_mbar, kGateWriterThreads);
            cute::initialize_barrier(smem.early_ready_mbar[0], 1);
            cute::initialize_barrier(smem.early_ready_mbar[1], 1);
            cute::initialize_barrier(smem.early_free_bar[0], kConsumerThreads);
            cute::initialize_barrier(smem.early_free_bar[1], kConsumerThreads);
            cute::initialize_barrier(smem.k_ready_mbar[0], 1);
            cute::initialize_barrier(smem.k_ready_mbar[1], 1);
            cute::initialize_barrier(smem.k_free_bar[0], kConsumerThreads);
            cute::initialize_barrier(smem.k_free_bar[1], kConsumerThreads);
            cute::initialize_barrier(smem.h_pack_ready_bar, kRoleThreads);
            cute::initialize_barrier(smem.m_pack_ready_bar, kRoleThreads);
            cute::initialize_barrier(smem.scratch_free_bar, kRoleThreads);
            cute::initialize_barrier(smem.a_read_done_bar, kRoleThreads);
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        if (wg_idx == 3) {
            cutlass::arch::warpgroup_reg_dealloc<kProducerRegisters>();

            if (role_tid < kWarpThreads && chunks > 0) {
                constexpr int kQkTmaBytesPerRow = kHeadDim * static_cast<int>(sizeof(T));
                constexpr int kVBytes           = BlockDv * static_cast<int>(sizeof(T));
                constexpr int kABytes           = kChunk32Size * static_cast<int>(sizeof(T));
                constexpr int kGateBytes        = 4 * static_cast<int>(sizeof(float));
                constexpr int kEarlyBytesPerRow = kVBytes + kABytes + kGateBytes;
                const int     token0            = token_base;
                auto&         early0            = smem.early[0];
                const int     initial_valid     = min(seq_len, kChunk32Size);
                float         beta_value        = 0.0f;
                if (role_tid < initial_valid) {
                    const int64_t beta_offset =
                        static_cast<int64_t>(physical_batch) * beta_batch_stride
                        + static_cast<int64_t>(local_sequence_begin + token0 + role_tid) * beta_stride + value_head;
                    beta_value = beta[beta_offset];
                }
                early0.gate_stage[1][role_tid][value_head & 3] = beta_value;
                __syncwarp();

                if (producer_leader) {
                    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&smem.early_ready_mbar[0],
                                                                                   kChunk32Size * kEarlyBytesPerRow);
                    cute::SM90_TMA_LOAD_4D::copy(v_tma_desc,
                                                 &smem.early_ready_mbar[0],
                                                 kTmaNoCacheHint,
                                                 &early0.v[0][0],
                                                 dv0,
                                                 value_head,
                                                 token0,
                                                 0);
                    if constexpr (BlockDv > kFusedGdrBlockDv) {
                        cute::SM90_TMA_LOAD_4D::copy(v_tma_desc,
                                                     &smem.early_ready_mbar[0],
                                                     kTmaNoCacheHint,
                                                     &early0.v[0][0] + kChunk32Size * kFusedGdrBlockDv,
                                                     dv0 + kFusedGdrBlockDv,
                                                     value_head,
                                                     token0,
                                                     0);
                    }
                    cute::SM90_TMA_LOAD_4D::copy(resolvent_tma_desc,
                                                 &smem.early_ready_mbar[0],
                                                 kTmaNoCacheHint,
                                                 &early0.a[0][0],
                                                 0,
                                                 value_head,
                                                 token0,
                                                 0);
                    cute::SM90_TMA_LOAD_3D::copy(g_tma_desc,
                                                 &smem.early_ready_mbar[0],
                                                 kTmaNoCacheHint,
                                                 &early0.gate_stage[0][0][0],
                                                 gate_tma_coord,
                                                 token0,
                                                 0);

                    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&smem.k_ready_mbar[0],
                                                                                   kChunk32Size * kQkTmaBytesPerRow);
                    cute::SM90_TMA_LOAD_5D::copy(k_tma_desc,
                                                 &smem.k_ready_mbar[0],
                                                 kTmaNoCacheHint,
                                                 &smem.k_stage[0][0][0],
                                                 0,
                                                 0,
                                                 qk_tma_head_coord,
                                                 token0,
                                                 0);
                }
            }

            int early_free_phase0 = 0;
            int early_free_phase1 = 0;
            int k_free_phase0     = 0;
            int k_free_phase1     = 0;
            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int next_chunk = chunk + 1;

                if (role_tid < kWarpThreads) {
                    constexpr int kQkTmaBytesPerRow = kHeadDim * static_cast<int>(sizeof(T));
                    constexpr int kVBytes           = BlockDv * static_cast<int>(sizeof(T));
                    constexpr int kABytes           = kChunk32Size * static_cast<int>(sizeof(T));
                    constexpr int kGateBytes        = 4 * static_cast<int>(sizeof(float));
                    constexpr int kEarlyBytesPerRow = kVBytes + kABytes + kGateBytes;

                    if (next_chunk < chunks) {
                        const int next_buf    = next_chunk & 1;
                        const int next_token0 = token_base + next_chunk * kChunk32Size;
                        auto&     next_early  = smem.early[next_buf];
                        if (producer_leader && next_chunk >= 2) {
                            if (next_buf == 0) {
                                cute::wait_barrier(smem.early_free_bar[0], early_free_phase0);
                                early_free_phase0 ^= 1;
                            }
                            else {
                                cute::wait_barrier(smem.early_free_bar[1], early_free_phase1);
                                early_free_phase1 ^= 1;
                            }
                        }
                        __syncwarp();
                        const int next_valid = min(seq_len - next_chunk * kChunk32Size, kChunk32Size);
                        float     beta_value = 0.0f;
                        if (role_tid < next_valid) {
                            const int64_t beta_offset =
                                static_cast<int64_t>(physical_batch) * beta_batch_stride
                                + static_cast<int64_t>(local_sequence_begin + next_token0 + role_tid) * beta_stride
                                + value_head;
                            beta_value = beta[beta_offset];
                        }
                        next_early.gate_stage[1][role_tid][value_head & 3] = beta_value;
                        __syncwarp();

                        if (producer_leader) {
                            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                                &smem.early_ready_mbar[next_buf], kChunk32Size * kEarlyBytesPerRow);
                            cute::SM90_TMA_LOAD_4D::copy(v_tma_desc,
                                                         &smem.early_ready_mbar[next_buf],
                                                         kTmaNoCacheHint,
                                                         &next_early.v[0][0],
                                                         dv0,
                                                         value_head,
                                                         next_token0,
                                                         0);
                            if constexpr (BlockDv > kFusedGdrBlockDv) {
                                cute::SM90_TMA_LOAD_4D::copy(v_tma_desc,
                                                             &smem.early_ready_mbar[next_buf],
                                                             kTmaNoCacheHint,
                                                             &next_early.v[0][0] + kChunk32Size * kFusedGdrBlockDv,
                                                             dv0 + kFusedGdrBlockDv,
                                                             value_head,
                                                             next_token0,
                                                             0);
                            }
                            cute::SM90_TMA_LOAD_4D::copy(resolvent_tma_desc,
                                                         &smem.early_ready_mbar[next_buf],
                                                         kTmaNoCacheHint,
                                                         &next_early.a[0][0],
                                                         0,
                                                         value_head,
                                                         next_token0,
                                                         0);
                            cute::SM90_TMA_LOAD_3D::copy(g_tma_desc,
                                                         &smem.early_ready_mbar[next_buf],
                                                         kTmaNoCacheHint,
                                                         &next_early.gate_stage[0][0][0],
                                                         gate_tma_coord,
                                                         next_token0,
                                                         0);

                            if (next_chunk >= 2) {
                                if (next_buf == 0) {
                                    cute::wait_barrier(smem.k_free_bar[0], k_free_phase0);
                                    k_free_phase0 ^= 1;
                                }
                                else {
                                    cute::wait_barrier(smem.k_free_bar[1], k_free_phase1);
                                    k_free_phase1 ^= 1;
                                }
                            }
                            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                                &smem.k_ready_mbar[next_buf], kChunk32Size * kQkTmaBytesPerRow);
                            cute::SM90_TMA_LOAD_5D::copy(k_tma_desc,
                                                         &smem.k_ready_mbar[next_buf],
                                                         kTmaNoCacheHint,
                                                         &smem.k_stage[next_buf][0][0],
                                                         0,
                                                         0,
                                                         qk_tma_head_coord,
                                                         next_token0,
                                                         0);
                        }
                    }
                }
            }
            return;
        }
        else if (wg_idx == 0) {
            cutlass::arch::warpgroup_reg_alloc<kStateRegisters>();
            using Element = MmaElement;

            using Mma = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
                                       cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                       cute::Tile<cute::Underscore, cute::Int<32>, cute::Underscore>>;

            Mma  mma;
            auto thr_mma = mma.get_thread_slice(role_tid);
            auto s_state_stage =
                cute::make_tensor(cute::make_smem_ptr(&smem.vd[0][0]),
                                  cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<BlockDv>{}),
                                                    cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{})));
            auto c_state  = cute::make_identity_tensor(cute::shape(s_state_stage));
            auto tCsState = thr_mma.partition_C(s_state_stage);
            auto tCcState = thr_mma.partition_C(c_state);
            auto tCrState = thr_mma.make_fragment_C(tCsState);
            cute::clear(tCrState);

            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int segment_token0 = chunk * kChunk32Size;
                const int remaining      = seq_len - segment_token0;
                const int valid          = remaining < kChunk32Size ? remaining : kChunk32Size;
                const int last_row       = valid - 1;
                const int data_phase     = chunk & 1;
                const int stage_phase    = (chunk >> 1) & 1;
                auto&     early          = smem.early[data_phase];

                cute::wait_barrier(smem.early_ready_mbar[data_phase], stage_phase);

                const int gate_lane     = value_head & 3;
                const int gate_warp     = role_tid / kWarpThreads;
                const int gate_warp_tid = role_tid % kWarpThreads;
                if (gate_warp_tid < kGateRowsPerWarp) {
#pragma unroll
                    for (int pass = 0; pass < kGatePasses; ++pass) {
                        const int   row     = pass * kGateWriterThreads + gate_warp * kGateRowsPerWarp + gate_warp_tid;
                        const float g_value = early.gate_stage[0][row][gate_lane];
                        const float g_exp   = FastExp(g_value);
                        early.g_exp[row]    = g_exp;
                        early.g[row]        = 1.0f / g_exp;
                    }
                }
                const float state_decay = FastExp(early.gate_stage[0][last_row][gate_lane]);
                if (gate_warp_tid < kGateRowsPerWarp) {
                    cute::arrive_barrier(smem.state_ready_mbar);
                }

                if (chunk > 0) {
                    cute::wait_barrier(smem.scratch_free_bar, (chunk - 1) & 1);
                }
                Element* state_pack = reinterpret_cast<Element*>(&smem.vd[0][0]);
                StoreStateFragmentBf16Stsm(tCrState, state_pack, thr_mma, role_tid);
                cute::arrive_barrier(smem.h_pack_ready_bar);

                DecayStateFragment(tCrState, state_decay);

                const int k_phase = (chunk >> 1) & 1;
                cute::wait_barrier(smem.k_ready_mbar[data_phase], k_phase);
                cutlass::arch::NamedBarrier::sync(2 * kRoleThreads, kBarrierHVdReady);
                Element* k_stage  = reinterpret_cast<Element*>(&smem.k_stage[data_phase][0][0]);
                Element* vd_stage = reinterpret_cast<Element*>(&early.v[0][0]);
                auto     s_k_t    = cute::make_tensor(cute::make_smem_ptr(k_stage), QkTransposedLayout());
                auto     s_vd_t   = cute::make_tensor(cute::make_smem_ptr(vd_stage), VTLayout());

                StateUpdateFragmentFromScaledVd(role_tid, mma, s_k_t, s_vd_t, tCrState);
                cute::arrive_barrier(smem.early_free_bar[data_phase]);
                cute::arrive_barrier(smem.k_free_bar[data_phase]);
            }

            if (chunks > 0) {
                const int final_data_phase = (chunks - 1) & 1;
                const int final_k_phase    = ((chunks - 1) >> 1) & 1;
                cute::wait_barrier(smem.k_free_bar[final_data_phase], final_k_phase);
            }
            const int64_t state_offset =
                (static_cast<int64_t>(segment_id + 1) * hv + value_head) * kHeadDim * kHeadDim + dv0;
            auto g_final_state =
                cute::make_tensor(cute::make_gmem_ptr(segment_state + state_offset),
                                  cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<BlockDv>{}),
                                                    cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{})));
            StoreStateFragmentGlobal<float>(tCrState, g_final_state, thr_mma, role_tid);
            return;
        }
        else if (wg_idx == 1) {
            cutlass::arch::warpgroup_reg_alloc<kStateRegisters>();
            using Element = MmaElement;

            if (!needs_transition) {
                for (int chunk = 0; chunk < chunks; ++chunk) {
                    const int data_phase  = chunk & 1;
                    const int stage_phase = (chunk >> 1) & 1;
                    cute::wait_barrier(smem.early_ready_mbar[data_phase], stage_phase);
                    cute::wait_barrier(smem.k_ready_mbar[data_phase], stage_phase);
                    cute::arrive_barrier(smem.early_free_bar[data_phase]);
                    cute::arrive_barrier(smem.k_free_bar[data_phase]);
                }
                return;
            }

            using Mma = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
                                       cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                       cute::Tile<cute::Underscore, cute::Int<32>, cute::Underscore>>;

            Mma  mma;
            auto thr_mma = mma.get_thread_slice(role_tid);
            auto s_state_stage =
                cute::make_tensor(cute::make_smem_ptr(&smem.vd[0][0]),
                                  cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<BlockDv>{}),
                                                    cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{})));
            auto c_state  = cute::make_identity_tensor(cute::shape(s_state_stage));
            auto tCsState = thr_mma.partition_C(s_state_stage);
            auto tCcState = thr_mma.partition_C(c_state);
            auto tCrState = thr_mma.make_fragment_C(tCsState);
#pragma unroll
            for (int i = 0; i < cute::size(tCrState); ++i) {
                auto      coord = tCcState(i);
                const int dk    = cute::get<0>(coord);
                const int dv    = cute::get<1>(coord);
                tCrState(i)     = dk == dv0 + dv ? 1.0f : 0.0f;
            }

            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int segment_token0 = chunk * kChunk32Size;
                const int remaining      = seq_len - segment_token0;
                const int valid          = remaining < kChunk32Size ? remaining : kChunk32Size;
                const int last_row       = valid - 1;
                const int data_phase     = chunk & 1;
                const int stage_phase    = (chunk >> 1) & 1;
                auto&     early          = smem.early[data_phase];

                cute::wait_barrier(smem.early_ready_mbar[data_phase], stage_phase);
                cute::wait_barrier(smem.state_ready_mbar, chunk & 1);
                cutlass::arch::NamedBarrier::sync(2 * kRoleThreads, kBarrierHStateReadDone);

                const float state_decay = early.g_exp[last_row];
                Element*    state_pack  = reinterpret_cast<Element*>(&smem.vd[0][0]);
                StoreStateFragmentBf16Stsm(tCrState, state_pack, thr_mma, role_tid);
                cute::arrive_barrier(smem.m_pack_ready_bar);

                DecayStateFragment(tCrState, state_decay);

                cute::wait_barrier(smem.k_ready_mbar[data_phase], stage_phase);
                cutlass::arch::NamedBarrier::sync(2 * kRoleThreads, kBarrierMVdReady);
                Element* k_stage  = reinterpret_cast<Element*>(&smem.k_stage[data_phase][0][0]);
                Element* vd_stage = reinterpret_cast<Element*>(&early.a[0][0]);
                auto     s_k_t    = cute::make_tensor(cute::make_smem_ptr(k_stage), QkTransposedLayout());
                auto     s_vd_t   = cute::make_tensor(cute::make_smem_ptr(vd_stage), VTLayout());

                StateUpdateFragmentFromScaledVd(role_tid, mma, s_k_t, s_vd_t, tCrState);
                cute::arrive_barrier(smem.early_free_bar[data_phase]);
                cute::arrive_barrier(smem.k_free_bar[data_phase]);
            }

            if (chunks > 0) {
                const int final_data_phase = (chunks - 1) & 1;
                const int final_k_phase    = ((chunks - 1) >> 1) & 1;
                cute::wait_barrier(smem.k_free_bar[final_data_phase], final_k_phase);
            }
            const int64_t state_offset =
                (static_cast<int64_t>(segment_id) * hv + value_head) * kHeadDim * kHeadDim + dv0;
            auto g_final_state =
                cute::make_tensor(cute::make_gmem_ptr(segment_m + state_offset),
                                  cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<BlockDv>{}),
                                                    cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{})));
            StoreStateFragmentGlobal<float>(tCrState, g_final_state, thr_mma, role_tid);
            return;
        }
        else if (wg_idx == 2) {
            cutlass::arch::warpgroup_reg_alloc<kValueRegisters>();
            using Element = MmaElement;

            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int segment_token0 = chunk * kChunk32Size;
                const int remaining      = seq_len - segment_token0;
                const int valid          = remaining < kChunk32Size ? remaining : kChunk32Size;
                const int last_row       = valid - 1;
                const int data_phase     = chunk & 1;
                const int stage_phase    = (chunk >> 1) & 1;
                auto&     early          = smem.early[data_phase];
                Element*  k_stage        = reinterpret_cast<Element*>(&smem.k_stage[data_phase][0][0]);
                auto      s_k_smem       = cute::make_tensor(cute::make_smem_ptr(k_stage), QkLayout());
                auto      s_a_smem =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&early.a[0][0])), SquareLayout());
                Element* w_pack = reinterpret_cast<Element*>(&early.v[0][0]);

                cute::wait_barrier(smem.early_ready_mbar[data_phase], stage_phase);
                cute::wait_barrier(smem.state_ready_mbar, chunk & 1);

                const int gate_lane = value_head & 3;
                using Mma           = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
                                           cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                           cute::Tile<cute::Underscore, cute::Int<32>, cute::Underscore>>;

                cute::wait_barrier(smem.k_ready_mbar[data_phase], stage_phase);

                const int passes = needs_transition ? 2 : 1;
                for (int pass = 0; pass < passes; ++pass) {
                    Mma      mma;
                    auto     thr_mma     = mma.get_thread_slice(role_tid);
                    Element* state_stage = reinterpret_cast<Element*>(&smem.vd[0][0]);
                    Element* w_stage     = pass == 0 ? w_pack : state_stage;
                    Element* vd_stage    = pass == 0 ? reinterpret_cast<Element*>(&early.v[0][0]) :
                                                       reinterpret_cast<Element*>(&early.a[0][0]);
                    auto     s_w_row     = cute::make_tensor(cute::make_smem_ptr(w_stage), VRowLayout());
                    auto     s_state     = cute::make_tensor(cute::make_smem_ptr(state_stage), StateTLayout());
                    auto     s_k_state   = cute::make_tensor(
                        cute::make_smem_ptr(&smem.vd[0][0]),
                        cute::make_layout(cute::make_shape(cute::Int<kChunk32Size>{}, cute::Int<BlockDv>{}),
                                          cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{})));
                    auto c_k_state = cute::make_identity_tensor(cute::shape(s_k_state));

                    auto tCsC = thr_mma.partition_C(s_k_state);
                    auto tCcC = thr_mma.partition_C(c_k_state);
                    auto tCrC = thr_mma.make_fragment_C(tCsC);
                    auto tCrW = cute::make_fragment_like<Element>(tCrC);
                    auto smem_tiled_copy_C =
                        cute::make_tiled_copy_C(cute::Copy_Atom<cute::DefaultCopy, Element>{}, thr_mma);
                    auto        smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(role_tid);
                    auto        tCsWStore       = smem_thr_copy_C.partition_D(s_w_row);
                    auto        tCrWStoreView   = smem_thr_copy_C.retile_S(tCrW);
                    auto        s_w             = cute::make_tensor(cute::make_smem_ptr(w_stage), VTLayout());
                    auto        s_vd            = cute::make_tensor(cute::make_smem_ptr(vd_stage), VRowLayout());
                    auto        tCsVdStore      = smem_thr_copy_C.partition_D(s_vd);
                    const float last_g_exp      = early.g_exp[last_row];

                    if (pass == 0) {
                        cute::wait_barrier(smem.h_pack_ready_bar, chunk & 1);
                    }
                    else {
                        cute::wait_barrier(smem.m_pack_ready_bar, chunk & 1);
                    }

                    cute::clear(tCrC);
                    if (pass == 0) {
                        using InputTypeA = Element;
                        using InputTypeB = Element;

                        auto tCrA  = thr_mma.partition_fragment_A(s_k_smem);
                        auto tCrAi = cute::make_fragment_like<InputTypeA>(tCrA);
                        auto tCrB  = thr_mma.partition_fragment_B(s_state);
                        auto tCrBi = cute::make_fragment_like<InputTypeB>(tCrB);

                        auto smem_tiled_copy_B =
                            cute::make_tiled_copy_B(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, InputTypeB>{}, thr_mma);
                        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(role_tid);
                        auto tCsB            = smem_thr_copy_B.partition_S(s_state);
                        auto tCrBi_copy_view = smem_thr_copy_B.retile_D(tCrBi);

                        constexpr int K_BLOCK_MAX = cute::size<2>(decltype(tCrA){});
#pragma unroll
                        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
                            cute::copy(smem_tiled_copy_B,
                                       tCsB(cute::_, cute::_, k_block),
                                       tCrBi_copy_view(cute::_, cute::_, k_block));
                        }
                        if (needs_transition) {
                            cutlass::arch::NamedBarrier::arrive(2 * kRoleThreads, kBarrierHStateReadDone);
                        }
                        else {
                            cute::arrive_barrier(smem.scratch_free_bar);
                        }

                        auto smem_tiled_copy_A =
                            cute::make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, InputTypeA>{}, thr_mma);
                        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(role_tid);
                        auto tCsA            = smem_thr_copy_A.partition_S(s_k_smem);
                        auto tCrAi_copy_view = smem_thr_copy_A.retile_D(tCrAi);

                        cute::copy(smem_tiled_copy_A,
                                   tCsA(cute::_, cute::_, cute::Int<0>{}),
                                   tCrAi_copy_view(cute::_, cute::_, cute::Int<0>{}));

#pragma unroll
                        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
                            if (k_block < K_BLOCK_MAX - 1) {
                                const int k_next = k_block + 1;
                                cute::copy(smem_tiled_copy_A,
                                           tCsA(cute::_, cute::_, k_next),
                                           tCrAi_copy_view(cute::_, cute::_, k_next));
                            }

                            cute::transform(
                                tCrAi(cute::_, cute::_, k_block), tCrA(cute::_, cute::_, k_block), cute::identity{});
                            cute::transform(
                                tCrBi(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), cute::identity{});
                            cute::gemm(thr_mma, tCrA(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block), tCrC);
                        }

                        auto tCsWLoad     = smem_thr_copy_C.partition_S(s_w_row);
                        auto tCrWLoadView = smem_thr_copy_C.retile_D(tCrW);
                        cute::copy(smem_tiled_copy_C, tCsWLoad, tCrWLoadView);
                    }
                    else {
                        cute::cooperative_gemm(role_tid,
                                               mma,
                                               s_k_smem,
                                               s_state,
                                               tCrC,
                                               cute::identity{},
                                               cute::identity{},
                                               cute::SM75_U32x4_LDSM_N{},
                                               cute::SM75_U16x8_LDSM_T{});
                        MmaSyncNamed<kBarrierValueU>();
                    }
#pragma unroll
                    for (int i = 0; i < cute::size(tCrC); ++i) {
                        auto        coord       = tCcC(i);
                        const int   row         = cute::get<0>(coord);
                        const float state_v     = early.g_exp[row] * tCrC(i);
                        const float input_v     = pass == 0 ? static_cast<float>(tCrW(i)) : 0.0f;
                        const float input_scale = early.g[row] * early.gate_stage[1][row][gate_lane];
                        tCrW(i)                 = Element(CastFromFloat((input_v - state_v) * input_scale));
                    }
                    cute::copy(smem_tiled_copy_C, tCrWStoreView, tCsWStore);
                    MmaSyncNamed<kBarrierValueU>();

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
                    if (pass != 0) {
                        cute::arrive_barrier(smem.a_read_done_bar);
                    }
#pragma unroll
                    for (int i = 0; i < cute::size(tCrC); ++i) {
                        const float gate = last_g_exp;
                        tCrW(i)          = Element(CastFromFloat(gate * tCrC(i)));
                    }
                    if (pass == 0) {
                        MmaSyncNamed<kBarrierValueU>();
                        cute::copy(smem_tiled_copy_C, tCrWStoreView, tCsVdStore);
                        cutlass::arch::NamedBarrier::arrive(2 * kRoleThreads, kBarrierHVdReady);
                    }
                    else {
                        cute::arrive_barrier(smem.scratch_free_bar);
                        cute::wait_barrier(smem.a_read_done_bar, chunk & 1);
                        cute::copy(smem_tiled_copy_C, tCrWStoreView, tCsVdStore);
                        cutlass::arch::NamedBarrier::arrive(2 * kRoleThreads, kBarrierMVdReady);
                    }
                }
                cute::arrive_barrier(smem.early_free_bar[data_phase]);
                cute::arrive_barrier(smem.k_free_bar[data_phase]);
            }
            return;
        }
        static_cast<void>(cp_finished);
    }
};

template<class T, int BlockDv>
__global__ __launch_bounds__(
    Sm120FusedGdrH<T, BlockDv>::kThreads,
    Sm120FusedGdrH<T,
                   BlockDv>::kMinBlocks) void Sm120FusedGdrHKernel(const CUtensorMap* __restrict__ tma_desc_workspace,
                                                                   float* __restrict__ segment_state,
                                                                   float* __restrict__ segment_m,
                                                                   const float* __restrict__ g_cumsum,
                                                                   const float* __restrict__ beta,
                                                                   const int32_t* __restrict__ q_offsets,
                                                                   const int32_t* __restrict__ cp_source_indices,
                                                                   const int32_t* __restrict__ cp_q_offsets,
                                                                   const bool* __restrict__ cp_finished,
                                                                   bool* __restrict__ cp_fallback,
                                                                   int     sequence_num,
                                                                   int     token_num,
                                                                   int     hq,
                                                                   int     hv,
                                                                   int64_t gate_stride,
                                                                   int64_t gate_batch_stride,
                                                                   int64_t beta_stride,
                                                                   int64_t beta_batch_stride,
                                                                   bool    allow_warmup)
{
#if __CUDA_ARCH__
    if constexpr (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1300) {
        extern __shared__ __align__(1024) unsigned char smem_raw[];
        Sm120FusedGdrH<T, BlockDv>::Run(tma_desc_workspace,
                                        segment_state,
                                        segment_m,
                                        g_cumsum,
                                        beta,
                                        q_offsets,
                                        cp_source_indices,
                                        cp_q_offsets,
                                        cp_finished,
                                        cp_fallback,
                                        sequence_num,
                                        token_num,
                                        hq,
                                        hv,
                                        gate_stride,
                                        gate_batch_stride,
                                        beta_stride,
                                        beta_batch_stride,
                                        allow_warmup,
                                        smem_raw);
    }
#endif
}

template<int BlockDv>
void SetFusedGdrHSharedMemoryLimit()
{
    static_assert(BlockDv == kFusedGdrHBlockDv);
    using Kernel = Sm120FusedGdrH<__nv_bfloat16, BlockDv>;
    TM_CUDA_CHECK(cudaFuncSetAttribute(Sm120FusedGdrHKernel<__nv_bfloat16, BlockDv>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(Kernel::kSharedBytes)));
}

template<int BlockDv>
void LaunchSm120FusedGdrHTyped(const core::Tensor&        k,
                               const core::Tensor&        v,
                               const core::Tensor&        g_cumsum,
                               const core::Tensor&        beta,
                               const core::Tensor&        resolvent,
                               core::Tensor&              segment_state,
                               core::Tensor&              segment_m,
                               const Problem&             problem,
                               const ContextParallelPlan& cp,
                               const core::Tensor&        q_offsets,
                               const core::Tensor&        cp_source_indices,
                               const core::Tensor&        cp_q_offsets,
                               const core::Tensor&        cp_finished,
                               core::Tensor&              cp_fallback,
                               void*                      tma_desc_workspace,
                               cudaStream_t               stream)
{
    static_assert(BlockDv == kFusedGdrHBlockDv);
    using Kernel = Sm120FusedGdrH<__nv_bfloat16, BlockDv>;
    static_cast<void>(k);
    static_cast<void>(v);
    static_cast<void>(resolvent);
    static_cast<void>(segment_state);
    static_cast<void>(segment_m);

    if (problem.arch != 1200 || problem.input_dtype != kBfloat16 || problem.hv % problem.hq != 0
        || problem.head_dim != kHeadDim || problem.chunk_size != kChunk32Size) {
        throw std::invalid_argument(Kernel::kUnsupportedMessage);
    }
    if (tma_desc_workspace == nullptr) {
        throw std::invalid_argument("chunk32 Fused GDR H requires a tensor-map descriptor workspace");
    }
    if (q_offsets.dtype() != kInt32 || q_offsets.size() < problem.sequence_num + 1) {
        throw std::invalid_argument("chunk32 Fused GDR H requires int32 q_offsets");
    }
    if (cp_source_indices.dtype() != kInt32 || cp_source_indices.size() < cp.total_segments) {
        throw std::invalid_argument("chunk32 Fused GDR H requires int32 CP source-index workspace");
    }
    if (cp_q_offsets.dtype() != kInt32 || cp_q_offsets.size() < cp.total_segments + 1) {
        throw std::invalid_argument("chunk32 Fused GDR H requires int32 CP q_offsets workspace");
    }
    if (cp_finished.dtype() != kBool || cp_finished.size() < cp.total_segments) {
        throw std::invalid_argument("chunk32 Fused GDR H requires bool CP finished workspace");
    }

    constexpr int block_dv = BlockDv;
    const int     dv_tiles = (kHeadDim + block_dv - 1) / block_dv;
    const dim3    grid(cp.total_segments, problem.hv * dv_tiles, 1);
    const dim3    block(Kernel::kThreads);

    SetFusedGdrHSharedMemoryLimit<block_dv>();
    Sm120FusedGdrHKernel<__nv_bfloat16, block_dv>
        <<<grid, block, Kernel::kSharedBytes, stream>>>(reinterpret_cast<CUtensorMap*>(tma_desc_workspace),
                                                        segment_state.data<float>(),
                                                        segment_m.data<float>(),
                                                        g_cumsum.data<float>(),
                                                        beta.data<float>(),
                                                        q_offsets.data<int32_t>(),
                                                        cp_source_indices.data<int32_t>(),
                                                        cp_q_offsets.data<int32_t>(),
                                                        cp_finished.data<bool>(),
                                                        cp_fallback.data<bool>(),
                                                        problem.sequence_num,
                                                        problem.token_num,
                                                        problem.hq,
                                                        problem.hv,
                                                        problem.gate_stride,
                                                        problem.gate_batch_stride,
                                                        problem.beta_stride,
                                                        problem.beta_batch_stride,
                                                        cp.cp_level == ContextParallelLevel::kAll);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
