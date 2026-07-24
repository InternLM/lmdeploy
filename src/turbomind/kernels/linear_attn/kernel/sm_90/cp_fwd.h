// Inspired by
// https://github.com/QwenLM/FlashQLA/blob/60f81453143e724bcaf3fc7921e71e7328f6ebcd/flash_qla/ops/gated_delta_rule/chunk/hopper/cp_fwd.py

#pragma once

#include "src/turbomind/kernels/linear_attn/kernel/sm_90/common.h"

namespace turbomind::linear_attn::delta_rule {
namespace {

template<class StateT, int BlockDv>
struct Sm90CorrectInitialStates {
    enum TmaDescIndex : int
    {
        kCorrectInitialStatesCpStateDesc = 0,
        kCorrectInitialStatesSegmentStateDesc,
        kCorrectInitialStatesSegmentMDesc,
    };

    static constexpr int kCorrectInitialStatesMRowsPerTma = 64;

    static constexpr int kCorrectInitialStatesKTile                = 128;
    static constexpr int kCorrectInitialStatesStoreStages          = 2;
    static constexpr int kCorrectInitialStatesProducerTid0         = 128;
    static constexpr int kCorrectInitialStatesConsumerThreads      = 128;
    static constexpr int kCorrectInitialStatesProducerRegisters    = 24;
    static constexpr int kCorrectInitialStatesConsumerRegisters    = 240;
    static constexpr int kCorrectInitialStatesConsumerNamedBarrier = 1;
    static constexpr int kCorrectInitialStatesBarrierCount         = 8;
    static constexpr int kCorrectInitialStatesHReadyBarrier0       = 0;
    static constexpr int kCorrectInitialStatesMReadyBarrier0       = 2;
    static constexpr int kCorrectInitialStatesHFreeBarrier0        = 4;
    static constexpr int kCorrectInitialStatesMFreeBarrier0        = 6;

    static_assert(kCorrectInitialStatesKTile == kHeadDim);
    static_assert(kCorrectInitialStatesStoreStages > 1);

    // Correct-initial-states actor/barrier contract:
    // - Producer WG1 collectively deallocates registers; its sole leader,
    //   thread 128, observes parity, arms H/M TMA loads, and waits for stage
    //   ownership. Consumer WG0 holds the 128 state fragments; thread 0 leads
    //   TMA stores, and the TMA load/store engines complete async transfers.
    // - h_ready_mbar[2] and m_ready_mbar[2] each have count 1 plus expected
    //   bytes. Thread 128 arms and issues each load; TMA completion releases
    //   H or M to WG0. Their phase is (iter >> 1) & 1 on stage reuse.
    // - h_free_bar[2] and m_free_bar[2] each have count 128. Every WG0 lane
    //   releases H after loading its fragment and M after optional fallback
    //   GMMA consumption. Thread 128 acquires ownership with free parity
    //   ((iter >> 1) & 1) ^ 1; generation 0 needs no explicit priming arrivals.
    // - ConsumerSync is named barrier 1 with count 128. Its five call-site
    //   contracts publish recycled h_store ownership, completed h_store writes,
    //   h_prev overwrite ownership, completed h_prev writes, and the final
    //   TMA-store drain, respectively.
    // Consumer thread 0 owns the store tail: fence prior SMEM writes into the
    // async proxy, issue and commit each store, throttle with wait<1>, and drain
    // every committed store with wait<0> before all consumers exit.
    struct SharedStorage {
        alignas(1024) __nv_bfloat16 m_stage[2][kHeadDim][kCorrectInitialStatesKTile];
        alignas(1024) __nv_bfloat16 h_stage[2][kHeadDim][BlockDv];
        alignas(1024) __nv_bfloat16 h_prev[kHeadDim][BlockDv];
        alignas(1024) float h_store[kCorrectInitialStatesStoreStages][kHeadDim][BlockDv];
    };

    static_assert(sizeof(SharedStorage) <= kFusedGdrMaxDynamicSharedBytes);
    static_assert(offsetof(SharedStorage, m_stage) % 1024 == 0);
    static_assert(offsetof(SharedStorage, h_stage) % 1024 == 0);
    static_assert(offsetof(SharedStorage, h_prev) % 1024 == 0);
    static_assert(offsetof(SharedStorage, h_store) % 1024 == 0);

    static __device__ __forceinline__ void ConsumerSync()
    {
        cutlass::arch::NamedBarrier::sync(kCorrectInitialStatesConsumerThreads,
                                          kCorrectInitialStatesConsumerNamedBarrier);
    }

    static CUTE_HOST_DEVICE constexpr auto GlobalStateTileLayout()
    {
        static_assert(BlockDv == 32);
        return cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<BlockDv>{}),
                                 cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{}));
    }

    template<class Element>
    static CUTE_HOST_DEVICE constexpr auto SharedStateTileLayout()
    {
        static_assert(BlockDv == 32);
        auto base = cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<BlockDv>{}),
                                      cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{}));
        if constexpr (std::is_same_v<Element, __nv_bfloat16> || std::is_same_v<Element, cute::bfloat16_t>) {
            return cute::composition(cute::Swizzle<2, 3, 3>{}, base);
        }
        else {
            static_assert(std::is_same_v<Element, float>);
            return cute::composition(cute::Swizzle<3, 2, 3>{}, base);
        }
    }

    static_assert(SharedStateTileLayout<__nv_bfloat16>()(cute::Int<2>{}, cute::Int<0>{}) == 72);
    static_assert(SharedStateTileLayout<float>()(cute::Int<1>{}, cute::Int<0>{}) == 36);

    template<class StateFragment, class SmemTensor, class ThrMma>
    static __device__ __forceinline__ void
    LoadBf16FragmentShared(StateFragment& tCrState, SmemTensor const& s_state, ThrMma const& thr_mma, int role_tid)
    {
        using Element = typename SmemTensor::value_type;
        static_assert(std::is_same_v<Element, cute::bfloat16_t>);

        // h_stage is a 128x32 bf16 TMA tile, so its legal TMA swizzle is SW64.
        // Use x1 LDSM: x2/x4 would span 16/32 row starts and repeat bank groups
        // under SW64, while x1 only consumes one 8-row matrix per instruction.
        auto s_pack            = cute::as_position_independent_swizzle_tensor(s_state);
        auto smem_tiled_copy_C = cute::make_tiled_copy_C(cute::Copy_Atom<cute::SM75_U32x1_LDSM_N, Element>{}, thr_mma);
        auto smem_thr_copy_C   = smem_tiled_copy_C.get_thread_slice(role_tid);
        auto tCsState          = smem_thr_copy_C.partition_S(s_pack);
        auto tCrPacked         = cute::make_fragment_like<Element>(tCrState);
        auto tCrPackedView     = smem_thr_copy_C.retile_D(tCrPacked);
        cute::copy(smem_tiled_copy_C, tCsState, tCrPackedView);
        cute::copy(tCrPacked, tCrState);
    }

    template<class StateFragment, class SmemTensor, class ThrMma>
    static __device__ __forceinline__ void
    StoreFloatFragmentShared(StateFragment const& tCrState, SmemTensor& s_state, ThrMma const& thr_mma, int role_tid)
    {
        auto smem_tiled_copy_C = cute::make_tiled_copy_C(cute::Copy_Atom<cute::AutoVectorizingCopy, float>{}, thr_mma);
        auto smem_thr_copy_C   = smem_tiled_copy_C.get_thread_slice(role_tid);
        auto tCsState          = smem_thr_copy_C.partition_D(s_state);
        auto tCrStateView      = smem_thr_copy_C.retile_S(tCrState);
        cute::copy(smem_tiled_copy_C, tCrStateView, tCsState);
    }

    static constexpr int kThreads   = 256;
    static constexpr int kMinBlocks = 1;

    static constexpr size_t SharedBytes()
    {
        return sizeof(SharedStorage);
    }

    static __device__ __forceinline__ void Run(const CUtensorMap* __restrict__ tma_desc_workspace,
                                               const int64_t* __restrict__ state_ptrs,
                                               const int32_t* __restrict__ cp_sequence_starts,
                                               const bool* __restrict__ finished,
                                               const bool* __restrict__ cp_fallback,
                                               const __nv_bfloat16* __restrict__ segment_state,
                                               const __nv_bfloat16* __restrict__ segment_m,
                                               float* __restrict__ cp_state_base,
                                               int64_t        state_layer_offset,
                                               int            num_head_groups,
                                               int            heads_per_block,
                                               int            sequence_num,
                                               unsigned char* smem_raw)
    {
        static_assert(kFusedGdrValidStateT<StateT>, "fused chunk GDR StateT must be float or bfloat16");
        static_assert(BlockDv == 32);
        auto&      smem = *reinterpret_cast<SharedStorage*>(smem_raw);
        __shared__ __align__(16) cute::uint64_t mbarrier_mem[kCorrectInitialStatesBarrierCount];
        auto*                                   h_ready_mbar = mbarrier_mem + kCorrectInitialStatesHReadyBarrier0;
        auto*                                   m_ready_mbar = mbarrier_mem + kCorrectInitialStatesMReadyBarrier0;
        auto*                                   h_free_bar   = mbarrier_mem + kCorrectInitialStatesHFreeBarrier0;
        auto*                                   m_free_bar   = mbarrier_mem + kCorrectInitialStatesMFreeBarrier0;

        const int     tid             = static_cast<int>(threadIdx.x);
        const int     sequence_id     = static_cast<int>(blockIdx.z);
        const int     head_tile       = static_cast<int>(blockIdx.x);
        constexpr int kDvTilesPerHead = kHeadDim / BlockDv;
        const int     value_head      = head_tile / kDvTilesPerHead;
        const int     dv_tile         = head_tile - value_head * kDvTilesPerHead;
        const int     dv0             = dv_tile * BlockDv;
        const int     hv              = static_cast<int>(gridDim.x) / kDvTilesPerHead;
        if (sequence_id >= sequence_num) {
            return;
        }
        const int first_segment_id = cp_sequence_starts[sequence_id];
        const int last_segment_id  = cp_sequence_starts[sequence_id + 1];
        if (first_segment_id == last_segment_id) {
            return;
        }

        const auto* cp_state_tma_desc      = tma_desc_workspace + kCorrectInitialStatesCpStateDesc;
        const auto* segment_state_tma_desc = tma_desc_workspace + kCorrectInitialStatesSegmentStateDesc;
        const auto* segment_m_tma_desc     = tma_desc_workspace + kCorrectInitialStatesSegmentMDesc;
        if (tid == 0) {
            cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(cp_state_tma_desc));
            cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(segment_state_tma_desc));
            cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(segment_m_tma_desc));
            cute::prefetch_tma_descriptor(cp_state_tma_desc);
            cute::prefetch_tma_descriptor(segment_state_tma_desc);
            cute::prefetch_tma_descriptor(segment_m_tma_desc);
#pragma unroll
            for (int stage = 0; stage < 2; ++stage) {
                cute::initialize_barrier(h_ready_mbar[stage], 1);
                cute::initialize_barrier(m_ready_mbar[stage], 1);
                cute::initialize_barrier(h_free_bar[stage], kCorrectInitialStatesConsumerThreads);
                cute::initialize_barrier(m_free_bar[stage], kCorrectInitialStatesConsumerThreads);
            }
            // Release barrier initialization to the CTA; the following CTA sync is
            // the acquire that publishes all eight initialized barrier objects.
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        constexpr int kPrefixHBytes = kHeadDim * BlockDv * static_cast<int>(sizeof(__nv_bfloat16));
        constexpr int kPrefixMBytes = kHeadDim * kCorrectInitialStatesKTile * static_cast<int>(sizeof(__nv_bfloat16));
        static_assert(kCorrectInitialStatesMRowsPerTma == 64);

        if (tid >= kCorrectInitialStatesProducerTid0) {
            // Register deallocation is WG1-collective, while the h_free/m_free
            // ownership-release parity waits must remain leader-only. Only thread
            // 128 may observe them, so lagging producer lanes cannot see the same
            // phase after two barrier generations (the ABA hazard).
            cutlass::arch::warpgroup_reg_dealloc<kCorrectInitialStatesProducerRegisters>();
            if (tid != kCorrectInitialStatesProducerTid0) {
                return;
            }
            for (int segment_id = first_segment_id; segment_id + 1 < last_segment_id; ++segment_id) {
                const int iter       = segment_id - first_segment_id;
                const int stage      = iter & 1;
                const int phase      = (iter >> 1) & 1;
                const int free_phase = phase ^ 1;
                // Acquire: thread 128 receives this H stage after all 128 consumers
                // release it; first use succeeds through complementary free parity.
                cute::wait_barrier(h_free_bar[stage], free_phase);
                // Release: arm a kPrefixHBytes transaction. TMA completion releases the
                // BF16 state tile through h_ready_mbar to consumer WG0.
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&h_ready_mbar[stage], kPrefixHBytes);
                cute::SM90_TMA_LOAD_4D::copy(segment_state_tma_desc,
                                             &h_ready_mbar[stage],
                                             kTmaNoCacheHint,
                                             &smem.h_stage[stage][0][0],
                                             dv0,
                                             0,
                                             value_head,
                                             segment_id);

                // Acquire: thread 128 receives this M stage after all 128 consumers
                // release it; first use also uses complementary free parity.
                cute::wait_barrier(m_free_bar[stage], free_phase);
                // Release: arm a kPrefixMBytes transaction. Completion of both 64-row TMA
                // boxes releases the full BF16 M tile through m_ready_mbar.
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&m_ready_mbar[stage], kPrefixMBytes);
                auto* m_stage = &smem.m_stage[stage][0][0];
#pragma unroll
                for (int col = 0; col < kHeadDim; col += kCorrectInitialStatesMRowsPerTma) {
                    cute::SM90_TMA_LOAD_4D::copy(segment_m_tma_desc,
                                                 &m_ready_mbar[stage],
                                                 kTmaNoCacheHint,
                                                 m_stage + col * kHeadDim,
                                                 col,
                                                 0,
                                                 value_head,
                                                 segment_id);
                }
            }
            return;
        }

        cutlass::arch::warpgroup_reg_alloc<kCorrectInitialStatesConsumerRegisters>();

        using Element           = typename FusedGdrMmaTraits<__nv_bfloat16>::Element;
        using FallbackTileShape = cute::Shape<cute::Int<64>, cute::Int<BlockDv>, cute::Int<64>>;
        using FallbackGmmaAtom  = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                                           Element,
                                                                           float,
                                                                           FallbackTileShape,
                                                                           cute::SM90::GMMA::Major::MN,
                                                                           cute::SM90::GMMA::Major::MN>());
        auto  fallback_mma      = cute::make_tiled_mma(FallbackGmmaAtom{});
        auto  fallback_thr_mma  = fallback_mma.get_thread_slice(tid);
        float h_fragment[BlockDv];
        auto  tCrH = cute::make_tensor(
            cute::make_rmem_ptr(h_fragment),
            cute::partition_shape_C(fallback_mma, cute::Shape<cute::Int<kHeadDim>, cute::Int<BlockDv>>{}));
        auto* initial_state_base =
            GroupedStateBase<StateT>(
                state_ptrs, sequence_id, value_head, num_head_groups, heads_per_block, state_layer_offset)
            + dv0;
        auto g_initial_state = cute::make_tensor(cute::make_gmem_ptr(initial_state_base), GlobalStateTileLayout());
        FusedGdrLoadStateFragmentGlobal<StateT>(tCrH, g_initial_state, fallback_thr_mma, tid);

        const auto store_cp_state = [&](int segment_id, int store_iter) {
            const int store_stage = store_iter % kCorrectInitialStatesStoreStages;
            if (store_iter >= kCorrectInitialStatesStoreStages) {
                if (tid == 0) {
                    // Acquire: throttle until at most one committed TMA store is
                    // pending, which releases this recycled h_store stage.
                    cute::tma_store_wait<kCorrectInitialStatesStoreStages - 1>();
                }
                // Rendezvous: the leader's wait<1> releases recycled h_store
                // ownership to all peer writers before they overwrite the stage.
                ConsumerSync();
            }

            auto s_store = cute::make_tensor(cute::make_smem_ptr(&smem.h_store[store_stage][0][0]),
                                             SharedStateTileLayout<float>());
            StoreFloatFragmentShared(tCrH, s_store, fallback_thr_mma, tid);
            // Rendezvous: all peer writes release the complete h_store tile to the
            // TMA-store leader before it exposes the tile to the async proxy.
            ConsumerSync();

            if (tid == 0) {
                // Release prior h_store SMEM writes from the generic proxy to the
                // async proxy before the TMA engine reads them.
                cute::tma_store_fence();
                cute::SM90_TMA_STORE_4D::copy(
                    cp_state_tma_desc, &smem.h_store[store_stage][0][0], dv0, 0, value_head, segment_id);
                // Release the issued TMA store into a committed async store group;
                // wait<1> throttles these groups and the final wait<0> drains them.
                cute::tma_store_arrive();
            }
        };

        for (int segment_id = first_segment_id; segment_id < last_segment_id; ++segment_id) {
            const int store_iter = segment_id - first_segment_id;
            store_cp_state(segment_id, store_iter);

            if (segment_id + 1 == last_segment_id) {
                break;
            }

            const int  iter     = segment_id - first_segment_id;
            const int  stage    = iter & 1;
            const int  phase    = (iter >> 1) & 1;
            const bool fallback = cp_fallback[segment_id * hv + value_head];

            if (fallback) {
                // Rendezvous: any prior fallback GMMA releases h_prev to all peer
                // writers before they overwrite the aliased snapshot.
                ConsumerSync();
                auto s_h_prev_store =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.h_prev[0][0])),
                                      FusedGdrGmmaStateRowLayout<Element, BlockDv>());
                FusedGdrStoreFragmentBf16Stsm<__nv_bfloat16, Element>(tCrH, s_h_prev_store, fallback_thr_mma, tid);
            }

            // Acquire this H stage from the TMA producer; kPrefixHBytes of BF16
            // state are visible before consumer WG0 loads its fragments.
            cute::wait_barrier(h_ready_mbar[stage], phase);
            // Rendezvous: h_prev writers release the complete snapshot to the
            // fallback GMMA; non-fallback iterations remain generation-aligned.
            ConsumerSync();
            auto s_h_stage =
                cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.h_stage[stage][0][0])),
                                  FusedGdrGmmaStateRowLayout<Element, BlockDv>());
            LoadBf16FragmentShared(tCrH, s_h_stage, fallback_thr_mma, tid);
            // Release H-stage ownership from all 128 consumers to thread 128 for reuse.
            cute::arrive_barrier(h_free_bar[stage]);

            // Acquire this M stage from the TMA producer; kPrefixMBytes of BF16 M
            // are visible before the optional fallback GMMA consumes the tile.
            cute::wait_barrier(m_ready_mbar[stage], phase);
            if (fallback) {
                auto s_m =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.m_stage[stage][0][0])),
                                      FusedGdrGmmaStateTLayout<Element, kWideGdrBlockDv>());
                auto s_h_prev = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.h_prev[0][0])),
                                                  FusedGdrGmmaStateTLayout<Element, BlockDv>());
                FusedGdrGmmaSs(fallback_mma, tid, s_m, s_h_prev, tCrH, cute::SM90::GMMA::ScaleOut::One);
            }
            // Release M-stage ownership from all 128 consumers after optional GMMA
            // consumption to thread 128 for reuse.
            cute::arrive_barrier(m_free_bar[stage]);
        }

        if (tid == 0) {
            // Acquire: drain every committed TMA store before the CTA can exit.
            cute::tma_store_wait<0>();
        }
        // Rendezvous: the leader's final wait<0> releases all store completion to
        // the consumer WG before any lane exits the kernel.
        ConsumerSync();

        static_cast<void>(cp_state_base);
        static_cast<void>(segment_state);
        static_cast<void>(segment_m);
        static_cast<void>(finished);
    }
};

template<class StateT, int BlockDv>
__global__ __launch_bounds__(
    Sm90CorrectInitialStates<StateT, BlockDv>::kThreads,
    Sm90CorrectInitialStates<StateT, BlockDv>::
        kMinBlocks) void Sm90CorrectInitialStatesKernel(const CUtensorMap* __restrict__ tma_desc_workspace,
                                                        const int64_t* __restrict__ state_ptrs,
                                                        const int32_t* __restrict__ cp_sequence_starts,
                                                        const bool* __restrict__ finished,
                                                        const bool* __restrict__ cp_fallback,
                                                        const __nv_bfloat16* __restrict__ segment_state,
                                                        const __nv_bfloat16* __restrict__ segment_m,
                                                        float* __restrict__ cp_state_base,
                                                        int64_t state_layer_offset,
                                                        int     num_head_groups,
                                                        int     heads_per_block,
                                                        int     sequence_num)
{
#if __CUDA_ARCH__
    if constexpr (__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000) {
        extern __shared__ __align__(1024) unsigned char smem_raw[];
        Sm90CorrectInitialStates<StateT, BlockDv>::Run(tma_desc_workspace,
                                                       state_ptrs,
                                                       cp_sequence_starts,
                                                       finished,
                                                       cp_fallback,
                                                       segment_state,
                                                       segment_m,
                                                       cp_state_base,
                                                       state_layer_offset,
                                                       num_head_groups,
                                                       heads_per_block,
                                                       sequence_num,
                                                       smem_raw);
    }
#endif
}

template<class StateT, int BlockDv>
void SetCorrectInitialStatesSharedMemoryLimit(size_t smem_bytes)
{
    static_assert(kFusedGdrValidStateT<StateT>, "fused chunk GDR StateT must be float or bfloat16");
    TM_CUDA_CHECK(cudaFuncSetAttribute(Sm90CorrectInitialStatesKernel<StateT, BlockDv>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(smem_bytes)));
}

template<class StateT>
void LaunchSm90CorrectInitialStatesTyped(core::Tensor&              cp_state,
                                         const core::Tensor&        state_ptrs,
                                         const core::Tensor&        finished,
                                         const core::Tensor&        cp_sequence_starts,
                                         const core::Tensor&        segment_state,
                                         const core::Tensor&        segment_m,
                                         const core::Tensor&        cp_fallback,
                                         const Problem&             problem,
                                         const ContextParallelPlan& cp,
                                         int64_t                    state_layer_offset,
                                         void*                      tma_desc_workspace,
                                         cudaStream_t               stream)
{
    static_assert(kFusedGdrValidStateT<StateT>, "fused chunk GDR StateT must be float or bfloat16");
    constexpr int block_dv = 32;
    const int     dv_tiles = CeilDiv(kHeadDim, block_dv);
    const dim3    grid(problem.hv * dv_tiles, 1, problem.sequence_num);
    using Kernel = Sm90CorrectInitialStates<StateT, block_dv>;
    const dim3   block(Kernel::kThreads);
    const size_t smem_bytes = Kernel::SharedBytes();
    static_cast<void>(cp);
    SetCorrectInitialStatesSharedMemoryLimit<StateT, block_dv>(smem_bytes);

    Sm90CorrectInitialStatesKernel<StateT, block_dv>
        <<<grid, block, smem_bytes, stream>>>(reinterpret_cast<CUtensorMap*>(tma_desc_workspace),
                                              reinterpret_cast<const int64_t*>(state_ptrs.raw_data()),
                                              cp_sequence_starts.data<int32_t>(),
                                              finished.data<bool>(),
                                              cp_fallback.data<bool>(),
                                              segment_state.data<__nv_bfloat16>(),
                                              segment_m.data<__nv_bfloat16>(),
                                              cp_state.data<float>(),
                                              state_layer_offset,
                                              problem.num_head_groups,
                                              problem.heads_per_block,
                                              problem.sequence_num);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
