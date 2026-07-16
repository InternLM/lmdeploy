// Inspired by
// https://github.com/QwenLM/FlashQLA/blob/60f81453143e724bcaf3fc7921e71e7328f6ebcd/flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_h.py

#pragma once

#include "src/turbomind/kernels/linear_attn/kernel/sm_90/common.h"

#include <cute/arch/copy_sm80.hpp>

namespace turbomind::linear_attn::delta_rule {
namespace {

template<class T>
struct Sm90FusedGdrH {
    static constexpr int kFusedGdrHProducerRegisters = 32;
    static constexpr int kFusedGdrHStateRegisters    = 160;
    static constexpr int kFusedGdrHXRegisters        = 160;
    static constexpr int kFusedGdrHYRegisters        = 160;

    static_assert(kFusedGdrHStateRegisters + kFusedGdrHXRegisters + kFusedGdrHYRegisters + kFusedGdrHProducerRegisters
                  == 512);

    static constexpr const char* kUnsupportedMessage = "Fused GDR H requires fixed CP segment tensors";

    // Fused GDR H actor/barrier contract:
    // - Producer WG3 uses warp 0 for TMA K, warp 1 for TMA V/resolvent, warp 2
    //   for cp.async gate data, and warp 3 as the observer. Consumer WGs 0, 1,
    //   and 2 own state, X, and Y, respectively.
    // - stage_ready_mbar[2] has count 96 plus expected bytes. TMA producer
    //   warps 0/1 contribute 64 plain arrivals after their leaders register
    //   K and V/resolvent bytes; gate warp 2 contributes 32 no-increment
    //   cp.async completions. Those 96 contributions, together with completion
    //   of the expected K/V/resolvent TMA bytes, release the staged payload to
    //   all consumer WGs. Its phase is (chunk >> 1) & 1 on each stage reuse.
    // - stage_free_bar[2] has count 384. WGs 0/1/2 each release 128 arrivals
    //   after their final stage use. Generation 0 is acquired with the
    //   complementary free parity (((chunk >> 1) & 1) ^ 1); there is no
    //   explicit priming arrival.
    // - iteration_entry_bar has count 416 and phase (chunk & 1). All three
    //   consumer WGs and the 32-thread observer warp arrive and wait at each
    //   chunk generation.
    // - h_gate_ready_bar has count 256 and phase (chunk & 1). WG0 publishes H
    //   and WG2 publishes derived gate data; both arrive and wait for each
    //   other, while the observer warp waits only to phase-lock arena reuse.
    // - xy_ready_bar has count 384 and phase (chunk & 1). WG1 publishes X and
    //   WG2 publishes Y with arrive-only handoffs; WG0 contributes completed
    //   decay, then waits for both payloads before updating state.
    // - state_update_done_bar has count 128 and phase (chunk & 1) when CalcM is
    //   enabled. WG0 arrives only after its state update; WGs 1/2 wait only
    //   before reusing aliased arenas for the optional M recurrence.
    // There is no async-store tail here: every consumer acquires each ready
    // generation and releases its final stage use before leaving the loop.
    struct SharedStorage {
        union {
            struct {
                alignas(1024) T h_shared[kWideGdrBlockDv][kHeadDim];
                alignas(1024) T k_stage[2][kChunkSize][kHeadDim];
                alignas(1024) T q_stage[2][kChunkSize][kHeadDim];
            };
            alignas(1024) float state_stage[kHeadDim][kWideGdrBlockDv];
        };
        alignas(1024) T v_stage[2][kWideGdrBlockDv][kChunkSize];
        alignas(1024) T a_stage[2][kChunkSize][kChunkSize];
        alignas(1024) T o_shared[kWideGdrBlockDv][kChunkSize];
        alignas(1024) T vd_shared[kWideGdrBlockDv][kChunkSize];
        alignas(1024) T vn_shared[kWideGdrBlockDv][kChunkSize];
        alignas(1024) T p_shared[kChunkSize][kChunkSize];
        alignas(1024) float gate_stage[2][2][kChunkSize][4];
        float g_rev_exp[kChunkSize];
        alignas(16) cute::uint64_t stage_ready_mbar[2];
        alignas(16) cute::uint64_t stage_free_bar[2];
        alignas(16) cute::uint64_t iteration_entry_bar;
        alignas(16) cute::uint64_t h_gate_ready_bar;
        alignas(16) cute::uint64_t xy_ready_bar;
        alignas(16) cute::uint64_t state_update_done_bar;
    };

    static constexpr size_t SharedBytes()
    {
        return sizeof(SharedStorage);
    }

    static_assert(SharedBytes() <= kFusedGdrMaxDynamicSharedBytes);

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
                          int64_t gate_batch_stride)
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
        const int       segment_chunks   = (segment_end - segment_begin) / kChunkSize;
        float           gate_sum         = 0.0f;
        out.chunks                       = segment_chunks;
        out.fallback                     = 1;
        for (int chunk = 0; chunk < segment_chunks; ++chunk) {
            const int     flat_token     = segment_end - chunk * kChunkSize - 1;
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

    static __device__ __forceinline__ void
    LoadKTile(const CUtensorMap* desc, cute::uint64_t* mbar, T* smem_tile, int qk_head, int token0)
    {
        // The SM90 WGMMA K-major consumer layout is half-head major in shared memory:
        // [DK/2][row][dk_mod]. Load the two contiguous global half-head boxes into
        // those two physical slabs.
        constexpr int kQkHalfElements = kChunkSize * (kHeadDim / 2);
        cute::SM90_TMA_LOAD_5D::copy(desc, mbar, kTmaNoCacheHint, smem_tile, 0, 0, qk_head, token0, 0);
        cute::SM90_TMA_LOAD_5D::copy(
            desc, mbar, kTmaNoCacheHint, smem_tile + kQkHalfElements, 0, 1, qk_head, token0, 0);
    }

    static __device__ __forceinline__ void LoadGateAsync(float (&gate_stage)[2][2][kChunkSize][4],
                                                         const float* __restrict__ g_cumsum,
                                                         const float* __restrict__ beta,
                                                         int     stage,
                                                         int     role_tid,
                                                         int     valid,
                                                         int     last_row,
                                                         int     gate_lane,
                                                         int64_t gate_stride,
                                                         int64_t gate_batch_stride,
                                                         int64_t beta_stride,
                                                         int64_t beta_batch_stride,
                                                         int     token_num,
                                                         int     flat_token0,
                                                         int     value_head)
    {
#pragma unroll
        for (int row = role_tid - 64; row < kChunkSize; row += kCudaWarpThreads) {
            const int     source_row  = row < valid ? row : last_row;
            const int     flat_token  = flat_token0 + source_row;
            const int     batch_id    = flat_token / token_num;
            const int     local_token = flat_token - batch_id * token_num;
            const int64_t gate_offset = static_cast<int64_t>(batch_id) * gate_batch_stride
                                        + static_cast<int64_t>(local_token) * gate_stride + value_head;
            const int64_t beta_offset = static_cast<int64_t>(batch_id) * beta_batch_stride
                                        + static_cast<int64_t>(local_token) * beta_stride + value_head;
            cute::SM80_CP_ASYNC_CACHEALWAYS<float>::copy(g_cumsum[gate_offset], gate_stage[stage][0][row][gate_lane]);
            cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<float>::copy(
                beta[beta_offset], gate_stage[stage][1][row][gate_lane], row < valid);
        }
    }

    template<bool CalcM>
    static __device__ __forceinline__ void Body(SharedStorage& smem,
                                                const CUtensorMap* __restrict__ tma_desc_workspace,
                                                const float* __restrict__ g_cumsum,
                                                const float* __restrict__ beta,
                                                __nv_bfloat16* __restrict__ segment_state,
                                                __nv_bfloat16* __restrict__ segment_m,
                                                const int32_t* __restrict__ q_offsets,
                                                const int32_t* __restrict__ cp_source_indices,
                                                const int32_t* __restrict__ cp_q_offsets,
                                                const bool* __restrict__ cp_finished,
                                                int     warmup_chunks,
                                                int     token_num,
                                                int     sequence_num,
                                                int     hq,
                                                int     hv,
                                                int64_t gate_stride,
                                                int64_t gate_batch_stride,
                                                int64_t beta_stride,
                                                int64_t beta_batch_stride)
    {
        static_assert(kFusedGdrHBlockDv == kWideGdrBlockDv);
        using Element = typename FusedGdrMmaTraits<T>::Element;

        const int tid         = static_cast<int>(threadIdx.x);
        const int wg_idx      = cutlass::canonical_warp_group_idx();
        const int role_tid    = tid % kFusedGdrRoleThreads;
        const int segment_id  = static_cast<int>(blockIdx.x);
        const int value_head  = static_cast<int>(blockIdx.y);
        const int sequence_id = cp_source_indices[segment_id];
        if (sequence_id < 0 || sequence_id >= sequence_num) {
            return;
        }
        const int sequence_begin = q_offsets[sequence_id];
        const int segment_begin  = cp_q_offsets[segment_id];
        const int segment_end    = cp_q_offsets[segment_id + 1];
        if (warmup_chunks <= 0) {
            return;
        }
        const int requested_begin = segment_end - warmup_chunks * kChunkSize;
        const int warmup_begin    = requested_begin > segment_begin ? requested_begin : segment_begin;
        const int seq_len         = segment_end - warmup_begin;
        if (seq_len <= 0) {
            return;
        }
        const int   token_base         = warmup_begin - sequence_begin;
        const int   chunks             = warmup_chunks;
        const int   qk_head            = value_head / (hv / hq);
        const auto  slices             = MakeFusedGdrHTmaDescriptorSlices(tma_desc_workspace, sequence_num);
        const auto* data_desc          = slices.data + sequence_id * kFusedGdrHDataDescCount;
        const auto* k_tma_desc         = &data_desc[kFusedGdrHKDesc];
        const auto* v_tma_desc         = &data_desc[kFusedGdrHVDesc];
        const auto* resolvent_tma_desc = &data_desc[kFusedGdrHResolventDesc];

        if (tid == 0) {
            cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(k_tma_desc));
            cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(v_tma_desc));
            cute::tma_descriptor_fence_acquire(reinterpret_cast<const cute::TmaDescriptor*>(resolvent_tma_desc));
            cute::prefetch_tma_descriptor(k_tma_desc);
            cute::prefetch_tma_descriptor(v_tma_desc);
            cute::prefetch_tma_descriptor(resolvent_tma_desc);
            cute::initialize_barrier(smem.iteration_entry_bar, kFusedGdrConsumerStoreThreads);
            cute::initialize_barrier(smem.h_gate_ready_bar, 2 * kFusedGdrRoleThreads);
            cute::initialize_barrier(smem.xy_ready_bar, kFusedGdrConsumerThreads);
            cute::initialize_barrier(smem.state_update_done_bar, kFusedGdrRoleThreads);
#pragma unroll
            for (int stage = 0; stage < 2; ++stage) {
                cute::initialize_barrier(smem.stage_ready_mbar[stage], 3 * kCudaWarpThreads);
                cute::initialize_barrier(smem.stage_free_bar[stage], kFusedGdrConsumerThreads);
            }
            // Release barrier initialization to the CTA; the following CTA sync is
            // the acquire that makes every initialized barrier visible to all WGs.
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        constexpr int kQkTmaBytes     = kChunkSize * kHeadDim * static_cast<int>(sizeof(T));
        constexpr int kValueTmaBytes  = kChunkSize * kWideGdrBlockDv * static_cast<int>(sizeof(T));
        constexpr int kSquareTmaBytes = kChunkSize * kChunkSize * static_cast<int>(sizeof(T));

        if (wg_idx == 3) {
            cutlass::arch::warpgroup_reg_dealloc<kFusedGdrHProducerRegisters>();
            if (role_tid < 96) {
                for (int chunk = 0; chunk < chunks; ++chunk) {
                    const int stage       = chunk & 1;
                    const int free_phase  = ((chunk >> 1) + 1) & 1;
                    const int token0      = token_base + chunk * kChunkSize;
                    const int flat_token0 = warmup_begin + chunk * kChunkSize;
                    const int remaining   = seq_len - chunk * kChunkSize;
                    const int valid       = remaining < kChunkSize ? remaining : kChunkSize;
                    const int last_row    = valid - 1;
                    const int gate_lane   = value_head & 3;
                    // Acquire: producer warps 0/1/2 wait for 384 consumer releases
                    // before reusing the stage; generation 0 uses complementary parity.
                    cute::wait_barrier(smem.stage_free_bar[stage], free_phase);
                    if (role_tid == 0) {
                        // Release: warp-0 leader registers kQkTmaBytes. K-tile TMA
                        // completion releases those bytes to stage_ready_mbar.
                        cutlass::arch::ClusterTransactionBarrier::expect_transaction(&smem.stage_ready_mbar[stage],
                                                                                     kQkTmaBytes);
                        LoadKTile(
                            k_tma_desc, &smem.stage_ready_mbar[stage], &smem.k_stage[stage][0][0], qk_head, token0);
                    }
                    else if (role_tid == 32) {
                        // Release: warp-1 leader registers kValueTmaBytes plus
                        // kSquareTmaBytes. V/resolvent TMA completion releases them.
                        cutlass::arch::ClusterTransactionBarrier::expect_transaction(&smem.stage_ready_mbar[stage],
                                                                                     kValueTmaBytes + kSquareTmaBytes);
                        cute::SM90_TMA_LOAD_4D::copy(v_tma_desc,
                                                     &smem.stage_ready_mbar[stage],
                                                     kTmaNoCacheHint,
                                                     &smem.v_stage[stage][0][0],
                                                     0,
                                                     value_head,
                                                     token0,
                                                     0);
                        cute::SM90_TMA_LOAD_4D::copy(v_tma_desc,
                                                     &smem.stage_ready_mbar[stage],
                                                     kTmaNoCacheHint,
                                                     &smem.v_stage[stage][kWideGdrBlockDv / 2][0],
                                                     kWideGdrBlockDv / 2,
                                                     value_head,
                                                     token0,
                                                     0);
                        cute::SM90_TMA_LOAD_4D::copy(resolvent_tma_desc,
                                                     &smem.stage_ready_mbar[stage],
                                                     kTmaNoCacheHint,
                                                     &smem.a_stage[stage][0][0],
                                                     0,
                                                     value_head,
                                                     token0,
                                                     0);
                    }
                    else if (role_tid >= 64) {
                        LoadGateAsync(smem.gate_stage,
                                      g_cumsum,
                                      beta,
                                      stage,
                                      role_tid,
                                      valid,
                                      last_row,
                                      gate_lane,
                                      gate_stride,
                                      gate_batch_stride,
                                      beta_stride,
                                      beta_batch_stride,
                                      token_num,
                                      flat_token0,
                                      value_head);
                        // Release: each gate producer attaches its cp.async completion
                        // without incrementing the 96-arrival count, which already
                        // includes all 32 gate lanes.
                        cutlass::arch::cpasync_barrier_arrive_noinc(&smem.stage_ready_mbar[stage]);
                    }
                    if (role_tid < 64) {
                        // Release: TMA warps 0/1 contribute 64 plain arrivals; the
                        // registered transaction bytes complete asynchronously.
                        cute::arrive_barrier(smem.stage_ready_mbar[stage]);
                    }
                }
            }
            else {
                for (int chunk = 0; chunk < chunks; ++chunk) {
                    const int phase = chunk & 1;
                    // Rendezvous: the observer warp arrives and waits with all three
                    // consumer WGs at the entry to this chunk generation.
                    cute::arrive_barrier(smem.iteration_entry_bar);
                    cute::wait_barrier(smem.iteration_entry_bar, phase);
                    // Acquire: the observer waits only for WG0 H and WG2 gate
                    // publication, phase-locking aliased-arena reuse without reading it.
                    cute::wait_barrier(smem.h_gate_ready_bar, phase);
                }
            }
            return;
        }

        if (wg_idx == 0) {
            cutlass::arch::warpgroup_reg_alloc<kFusedGdrHStateRegisters>();
            using StateTileShape = cute::Shape<cute::Int<64>, cute::Int<kWideGdrBlockDv>, cute::Int<64>>;
            using StateGmmaAtom  = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                                            Element,
                                                                            float,
                                                                            StateTileShape,
                                                                            cute::SM90::GMMA::Major::MN,
                                                                            cute::SM90::GMMA::Major::MN>());
            auto state_mma       = cute::make_tiled_mma(StateGmmaAtom{});
            auto thr_mma         = state_mma.get_thread_slice(role_tid);
            auto s_state_stage   = cute::make_tensor(
                cute::make_smem_ptr(&smem.state_stage[0][0]),
                cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<kWideGdrBlockDv>{}),
                                  cute::make_stride(cute::Int<kWideGdrBlockDv>{}, cute::Int<1>{})));
            auto c_state  = cute::make_identity_tensor(cute::shape(s_state_stage));
            auto tCsState = thr_mma.partition_C(s_state_stage);
            auto tCcState = thr_mma.partition_C(c_state);
            auto tCrState = thr_mma.make_fragment_C(tCsState);
            cute::clear(tCrState);

            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int segment_token0 = chunk * kChunkSize;
                const int remaining      = seq_len - segment_token0;
                const int valid          = remaining < kChunkSize ? remaining : kChunkSize;
                const int last_row       = valid - 1;
                const int stage          = chunk & 1;
                const int phase          = chunk & 1;
                const int stage_phase    = (chunk >> 1) & 1;
                const int gate_lane      = value_head & 3;

                // Acquire: WG0 waits for all 96 producer contributions, including
                // gate cp.async completions, plus expected K/V/resolvent TMA bytes.
                cute::wait_barrier(smem.stage_ready_mbar[stage], stage_phase);
                // Rendezvous: WG0 arrives and waits with consumer WGs 1/2 and the
                // observer warp at this chunk-generation entry.
                cute::arrive_barrier(smem.iteration_entry_bar);
                cute::wait_barrier(smem.iteration_entry_bar, phase);

                auto s_h = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.h_shared[0][0])),
                                             FusedGdrGmmaStateRowLayout<Element, kWideGdrBlockDv>());
                FusedGdrStoreFragmentBf16Stsm<T, Element>(tCrState, s_h, thr_mma, role_tid);
                // Release: WG0's 128 lanes publish H to WG2 and contribute arrivals
                // observed by the wait-only observer, which only phase-locks reuse.
                cute::arrive_barrier(smem.h_gate_ready_bar);
                // Acquire: WG0 waits for WG2's 128 derived-gate arrivals before decay.
                cute::wait_barrier(smem.h_gate_ready_bar, phase);

                const float last_g_value = smem.gate_stage[stage][0][last_row][gate_lane];
                FusedGdrDecayStateFragment(tCrState, FastExp(last_g_value));
                // Release: WG0 contributes its completed decay to the X/Y generation.
                cute::arrive_barrier(smem.xy_ready_bar);

                // Acquire: WG0 waits for WG1's X and WG2's Y publications before
                // consuming both shared-memory operands in the state update.
                cute::wait_barrier(smem.xy_ready_bar, phase);
                auto s_x_t =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.q_stage[stage][0][0])),
                                      FusedGdrGmmaQkTransposeALayout<Element>());
                Element* y_stage = stage == 0 ? reinterpret_cast<Element*>(&smem.vn_shared[0][0]) :
                                                reinterpret_cast<Element*>(&smem.vd_shared[0][0]);
                auto     s_y_t =
                    cute::make_tensor(cute::make_smem_ptr(y_stage), FusedGdrGmmaVdTLayout<Element, kWideGdrBlockDv>());
                FusedGdrStateUpdateFragmentGmmaBf16Vd<kWideGdrBlockDv, T>(role_tid, s_x_t, s_y_t, tCrState);
                if constexpr (CalcM) {
                    // Release: WG0 alone contributes the 128 arrivals that hand
                    // state-update/scratch ownership to the wait-only WGs 1 and 2.
                    cute::arrive_barrier(smem.state_update_done_bar);
                }
                // Release: WG0's 128 lanes release their final stage use to WG3.
                cute::arrive_barrier(smem.stage_free_bar[stage]);
            }
            __nv_bfloat16* segment_state_head =
                segment_state + (static_cast<int64_t>(segment_id) * hv + value_head) * kHeadDim * kHeadDim;
            auto g_state =
                cute::make_tensor(cute::make_gmem_ptr(segment_state_head),
                                  cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<kHeadDim>{}),
                                                    cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{})));
            FusedGdrStoreStateFragmentGlobal<__nv_bfloat16>(tCrState, g_state, thr_mma, role_tid);
            return;
        }

        if (wg_idx == 1) {
            cutlass::arch::warpgroup_reg_alloc<kFusedGdrHXRegisters>();

            using XTileShape = cute::Shape<cute::Int<64>, cute::Int<kHeadDim>, cute::Int<64>>;
            using XGmmaAtom  = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                                        Element,
                                                                        float,
                                                                        XTileShape,
                                                                        cute::SM90::GMMA::Major::MN,
                                                                        cute::SM90::GMMA::Major::MN>());
            using MTileShape = cute::Shape<cute::Int<64>, cute::Int<kFusedGdrBlockDv>, cute::Int<64>>;
            using MGmmaAtom  = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                                        Element,
                                                                        float,
                                                                        MTileShape,
                                                                        cute::SM90::GMMA::Major::MN,
                                                                        cute::SM90::GMMA::Major::MN>());
            using ZGmmaAtom  = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                                        Element,
                                                                        float,
                                                                        MTileShape,
                                                                        cute::SM90::GMMA::Major::K,
                                                                        cute::SM90::GMMA::Major::MN>());
            auto x_mma       = cute::make_tiled_mma(XGmmaAtom{});
            auto m_mma       = cute::make_tiled_mma(MGmmaAtom{});
            auto z_mma       = cute::make_tiled_mma(ZGmmaAtom{});
            auto thr_x       = x_mma.get_thread_slice(role_tid);
            auto thr_m       = m_mma.get_thread_slice(role_tid);
            auto thr_z       = z_mma.get_thread_slice(role_tid);

            auto s_x_c =
                cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.q_stage[0][0][0])),
                                  cute::make_layout(cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<kHeadDim>{}),
                                                    cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{})));
            auto c_x  = cute::make_identity_tensor(cute::shape(s_x_c));
            auto tCsX = thr_x.partition_C(s_x_c);
            auto tCcX = thr_x.partition_C(c_x);
            auto tCrX = thr_x.make_fragment_C(tCsX);

            auto s_m_c = cute::make_tensor(
                cute::make_smem_ptr(&smem.state_stage[0][0]),
                cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<kFusedGdrBlockDv>{}),
                                  cute::make_stride(cute::Int<kFusedGdrBlockDv>{}, cute::Int<1>{})));
            auto c_m  = cute::make_identity_tensor(cute::shape(s_m_c));
            auto tCsM = thr_m.partition_C(s_m_c);
            auto tCcM = thr_m.partition_C(c_m);
            auto tCrM = thr_m.make_fragment_C(tCsM);
            if constexpr (CalcM) {
#pragma unroll
                for (int i = 0; i < cute::size(tCrM); ++i) {
                    auto      coord = tCcM(i);
                    const int dk    = cute::get<0>(coord);
                    const int dv    = cute::get<1>(coord);
                    tCrM(i)         = dk == kFusedGdrBlockDv + dv ? 1.0f : 0.0f;
                }
            }
            float g_prod = 0.0f;

            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int segment_token0 = chunk * kChunkSize;
                const int remaining      = seq_len - segment_token0;
                const int valid          = remaining < kChunkSize ? remaining : kChunkSize;
                const int last_row       = valid - 1;
                const int stage          = chunk & 1;
                const int phase          = chunk & 1;
                const int stage_phase    = (chunk >> 1) & 1;
                const int gate_lane      = value_head & 3;

                // Acquire: WG1 waits for the complete K/V/resolvent/gate stage
                // before its X and optional M-recurrence reads.
                cute::wait_barrier(smem.stage_ready_mbar[stage], stage_phase);
                // Rendezvous: WG1 arrives and waits with WGs 0/2 and the observer.
                cute::arrive_barrier(smem.iteration_entry_bar);
                cute::wait_barrier(smem.iteration_entry_bar, phase);

                auto s_a_t =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.a_stage[stage][0][0])),
                                      FusedGdrGmmaSquareLayout<Element>());
                auto s_k =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.k_stage[stage][0][0])),
                                      FusedGdrGmmaQkKLayout<Element>());
                auto s_k_x =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.k_stage[stage][0][0])),
                                      FusedGdrGmmaQkTransposeALayout<Element>());
                FusedGdrGmmaSs<false>(x_mma, role_tid, s_a_t, s_k_x, tCrX, cute::SM90::GMMA::ScaleOut::Zero);

                auto s_x_store =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.q_stage[stage][0][0])),
                                      FusedGdrGmmaQkKLayout<Element>());
#pragma unroll
                for (int i = 0; i < cute::size(tCrX); ++i) {
                    auto        coord      = tCcX(i);
                    const int   row        = cute::get<0>(coord);
                    const float beta_value = row < valid ? smem.gate_stage[stage][1][row][gate_lane] : 0.0f;
                    tCrX(i)                = -beta_value * static_cast<float>(tCrX(i));
                }
                FusedGdrStoreFragmentBf16Stsm<T, Element>(tCrX, s_x_store, thr_x, role_tid);
                // Release: WG1 publishes X to WG0 with 128 arrive-only handoffs;
                // it does not wait on the X/Y generation.
                cute::arrive_barrier(smem.xy_ready_bar);

                if constexpr (CalcM) {
                    g_prod += smem.gate_stage[stage][0][last_row][gate_lane];
                    Element* m_pack   = reinterpret_cast<Element*>(&smem.o_shared[0][0]);
                    Element* z_pack   = reinterpret_cast<Element*>(&smem.p_shared[0][0]);
                    auto     s_m_pack = cute::make_tensor(cute::make_smem_ptr(m_pack),
                                                      FusedGdrGmmaStateRowLayout<Element, kFusedGdrBlockDv>());
                    FusedGdrStoreFragmentBf16Stsm<T, Element>(tCrM, s_m_pack, thr_m, role_tid);

                    // Acquire: WG1 waits only for WG0's completed state update
                    // before reusing the aliased O/P scratch for M recurrence.
                    cute::wait_barrier(smem.state_update_done_bar, phase);
                    auto s_m_t = cute::make_tensor(cute::make_smem_ptr(m_pack),
                                                   FusedGdrGmmaStateTLayout<Element, kFusedGdrBlockDv>());
                    auto s_z_c = cute::make_tensor(
                        cute::make_smem_ptr(z_pack),
                        cute::make_layout(cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<kFusedGdrBlockDv>{}),
                                          cute::make_stride(cute::Int<kFusedGdrBlockDv>{}, cute::Int<1>{})));
                    auto tCsZ = thr_z.partition_C(s_z_c);
                    auto tCrZ = thr_z.make_fragment_C(tCsZ);
                    FusedGdrGmmaSs(z_mma, role_tid, s_k, s_m_t, tCrZ, cute::SM90::GMMA::ScaleOut::Zero);
                    auto s_z_store = cute::make_tensor(cute::make_smem_ptr(z_pack),
                                                       FusedGdrGmmaVdRowLayout<Element, kFusedGdrBlockDv>());
                    FusedGdrStoreFragmentBf16Stsm<T, Element>(tCrZ, s_z_store, thr_z, role_tid);

                    auto s_x_t =
                        cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.q_stage[stage][0][0])),
                                          FusedGdrGmmaQkTransposeALayout<Element>());
                    auto s_z_t = cute::make_tensor(cute::make_smem_ptr(z_pack),
                                                   FusedGdrGmmaVdTLayout<Element, kFusedGdrBlockDv>());
                    FusedGdrStateUpdateFragmentGmmaBf16Vd<kFusedGdrBlockDv, T>(role_tid, s_x_t, s_z_t, tCrM);
                }
                // Release: WG1's 128 lanes release their final stage use to WG3.
                cute::arrive_barrier(smem.stage_free_bar[stage]);
            }

            if constexpr (CalcM) {
                const float g_scale = FastExp(g_prod);
#pragma unroll
                for (int i = 0; i < cute::size(tCrM); ++i) {
                    tCrM(i) *= g_scale;
                }
                __nv_bfloat16* segment_m_head =
                    segment_m + (static_cast<int64_t>(segment_id) * hv + value_head) * kHeadDim * kHeadDim
                    + kFusedGdrBlockDv;
                auto g_m = cute::make_tensor(
                    cute::make_gmem_ptr(segment_m_head),
                    cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<kFusedGdrBlockDv>{}),
                                      cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{})));
                FusedGdrStoreStateFragmentGlobal<__nv_bfloat16>(tCrM, g_m, thr_m, role_tid);
            }
            return;
        }

        if (wg_idx == 2) {
            cutlass::arch::warpgroup_reg_alloc<kFusedGdrHYRegisters>();

            using YTileShape = cute::Shape<cute::Int<64>, cute::Int<kWideGdrBlockDv>, cute::Int<64>>;
            using YGmmaAtom  = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                                        Element,
                                                                        float,
                                                                        YTileShape,
                                                                        cute::SM90::GMMA::Major::K,
                                                                        cute::SM90::GMMA::Major::MN>());
            using MTileShape = cute::Shape<cute::Int<64>, cute::Int<kFusedGdrBlockDv>, cute::Int<64>>;
            using MGmmaAtom  = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                                        Element,
                                                                        float,
                                                                        MTileShape,
                                                                        cute::SM90::GMMA::Major::MN,
                                                                        cute::SM90::GMMA::Major::MN>());
            using ZGmmaAtom  = decltype(cute::SM90::GMMA::ss_op_selector<Element,
                                                                        Element,
                                                                        float,
                                                                        MTileShape,
                                                                        cute::SM90::GMMA::Major::K,
                                                                        cute::SM90::GMMA::Major::MN>());
            auto y_mma       = cute::make_tiled_mma(YGmmaAtom{});
            auto m_mma       = cute::make_tiled_mma(MGmmaAtom{});
            auto z_mma       = cute::make_tiled_mma(ZGmmaAtom{});
            auto thr_y       = y_mma.get_thread_slice(role_tid);
            auto thr_m       = m_mma.get_thread_slice(role_tid);
            auto thr_z       = z_mma.get_thread_slice(role_tid);

            auto s_y_c = cute::make_tensor(
                cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.vn_shared[0][0])),
                cute::make_layout(cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<kWideGdrBlockDv>{}),
                                  cute::make_stride(cute::Int<kWideGdrBlockDv>{}, cute::Int<1>{})));
            auto c_y  = cute::make_identity_tensor(cute::shape(s_y_c));
            auto tCsY = thr_y.partition_C(s_y_c);
            auto tCcY = thr_y.partition_C(c_y);
            auto tCrY = thr_y.make_fragment_C(tCsY);

            auto s_m_c = cute::make_tensor(
                cute::make_smem_ptr(&smem.state_stage[0][0]),
                cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<kFusedGdrBlockDv>{}),
                                  cute::make_stride(cute::Int<kFusedGdrBlockDv>{}, cute::Int<1>{})));
            auto c_m  = cute::make_identity_tensor(cute::shape(s_m_c));
            auto tCsM = thr_m.partition_C(s_m_c);
            auto tCcM = thr_m.partition_C(c_m);
            auto tCrM = thr_m.make_fragment_C(tCsM);
            if constexpr (CalcM) {
#pragma unroll
                for (int i = 0; i < cute::size(tCrM); ++i) {
                    auto      coord = tCcM(i);
                    const int dk    = cute::get<0>(coord);
                    const int dv    = cute::get<1>(coord);
                    tCrM(i)         = dk == dv ? 1.0f : 0.0f;
                }
            }
            float g_prod = 0.0f;

            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int segment_token0 = chunk * kChunkSize;
                const int remaining      = seq_len - segment_token0;
                const int valid          = remaining < kChunkSize ? remaining : kChunkSize;
                const int last_row       = valid - 1;
                const int stage          = chunk & 1;
                const int phase          = chunk & 1;
                const int stage_phase    = (chunk >> 1) & 1;
                const int gate_lane      = value_head & 3;

                // Acquire: WG2 waits for the complete K/V/resolvent/gate stage
                // before deriving gate data, Y, and optional M recurrence.
                cute::wait_barrier(smem.stage_ready_mbar[stage], stage_phase);
                // Rendezvous: WG2 arrives and waits with WGs 0/1 and the observer.
                cute::arrive_barrier(smem.iteration_entry_bar);
                cute::wait_barrier(smem.iteration_entry_bar, phase);

                const float last_g_value = smem.gate_stage[stage][0][last_row][gate_lane];
                const float last_g_exp   = FastExp(last_g_value);
                if (role_tid < kChunkSize) {
                    const float g_value      = role_tid < valid ? smem.gate_stage[stage][0][role_tid][gate_lane] : 0.0f;
                    smem.g_rev_exp[role_tid] = role_tid < valid ? FastExp(last_g_value - g_value) : 0.0f;
                }
                // Release: WG2's 128 lanes publish derived gate data to WG0 and
                // contribute the arrivals observed by the wait-only observer warp.
                cute::arrive_barrier(smem.h_gate_ready_bar);
                // Acquire: WG2 waits for WG0's 128 H-publication arrivals.
                cute::wait_barrier(smem.h_gate_ready_bar, phase);

                auto s_k =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.k_stage[stage][0][0])),
                                      FusedGdrGmmaQkKLayout<Element>());
                auto s_h = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.h_shared[0][0])),
                                             FusedGdrGmmaStateTLayout<Element, kWideGdrBlockDv>());
                FusedGdrGmmaSs(y_mma, role_tid, s_k, s_h, tCrY, cute::SM90::GMMA::ScaleOut::Zero);

                auto s_v =
                    cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.v_stage[stage][0][0])),
                                      FusedGdrSwizzledVTLayout<kWideGdrBlockDv>());
                Element* y_stage = stage == 0 ? reinterpret_cast<Element*>(&smem.vn_shared[0][0]) :
                                                reinterpret_cast<Element*>(&smem.vd_shared[0][0]);
#pragma unroll
                for (int i = 0; i < cute::size(tCrY); ++i) {
                    auto      coord   = tCcY(i);
                    const int row     = cute::get<0>(coord);
                    const int dv      = cute::get<1>(coord);
                    float     y_value = 0.0f;
                    if (row < valid) {
                        const float v_value = static_cast<float>(s_v(dv, row));
                        y_value             = last_g_exp * static_cast<float>(tCrY(i)) - smem.g_rev_exp[row] * v_value;
                    }
                    tCrY(i) = y_value;
                }
                auto s_y_store = cute::make_tensor(cute::make_smem_ptr(y_stage),
                                                   FusedGdrGmmaVdRowLayout<Element, kWideGdrBlockDv>());
                FusedGdrStoreFragmentBf16Stsm<T, Element>(tCrY, s_y_store, thr_y, role_tid);
                // Release: WG2 publishes Y to WG0 with 128 arrive-only handoffs;
                // it does not wait on the X/Y generation.
                cute::arrive_barrier(smem.xy_ready_bar);

                if constexpr (CalcM) {
                    g_prod += last_g_value;
                    Element* m_pack   = reinterpret_cast<Element*>(&smem.v_stage[stage][0][0]);
                    Element* z_pack   = reinterpret_cast<Element*>(&smem.a_stage[stage][0][0]);
                    auto     s_m_pack = cute::make_tensor(cute::make_smem_ptr(m_pack),
                                                      FusedGdrGmmaStateRowLayout<Element, kFusedGdrBlockDv>());
                    FusedGdrStoreFragmentBf16Stsm<T, Element>(tCrM, s_m_pack, thr_m, role_tid);

                    // Acquire: WG2 waits only for WG0's completed state update
                    // before reusing the aliased V/A stage for M recurrence.
                    cute::wait_barrier(smem.state_update_done_bar, phase);
                    auto s_m_t = cute::make_tensor(cute::make_smem_ptr(m_pack),
                                                   FusedGdrGmmaStateTLayout<Element, kFusedGdrBlockDv>());
                    auto s_z_c = cute::make_tensor(
                        cute::make_smem_ptr(z_pack),
                        cute::make_layout(cute::make_shape(cute::Int<kChunkSize>{}, cute::Int<kFusedGdrBlockDv>{}),
                                          cute::make_stride(cute::Int<kFusedGdrBlockDv>{}, cute::Int<1>{})));
                    auto tCsZ = thr_z.partition_C(s_z_c);
                    auto tCrZ = thr_z.make_fragment_C(tCsZ);
                    FusedGdrGmmaSs(z_mma, role_tid, s_k, s_m_t, tCrZ, cute::SM90::GMMA::ScaleOut::Zero);
                    auto s_z_store = cute::make_tensor(cute::make_smem_ptr(z_pack),
                                                       FusedGdrGmmaVdRowLayout<Element, kFusedGdrBlockDv>());
                    FusedGdrStoreFragmentBf16Stsm<T, Element>(tCrZ, s_z_store, thr_z, role_tid);

                    auto s_x_t =
                        cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.q_stage[stage][0][0])),
                                          FusedGdrGmmaQkTransposeALayout<Element>());
                    auto s_z_t = cute::make_tensor(cute::make_smem_ptr(z_pack),
                                                   FusedGdrGmmaVdTLayout<Element, kFusedGdrBlockDv>());
                    FusedGdrStateUpdateFragmentGmmaBf16Vd<kFusedGdrBlockDv, T>(role_tid, s_x_t, s_z_t, tCrM);
                }
                // Release: WG2's 128 lanes release their final stage use to WG3.
                cute::arrive_barrier(smem.stage_free_bar[stage]);
            }

            if constexpr (CalcM) {
                const float g_scale = FastExp(g_prod);
#pragma unroll
                for (int i = 0; i < cute::size(tCrM); ++i) {
                    tCrM(i) *= g_scale;
                }
                __nv_bfloat16* segment_m_head =
                    segment_m + (static_cast<int64_t>(segment_id) * hv + value_head) * kHeadDim * kHeadDim;
                auto g_m = cute::make_tensor(
                    cute::make_gmem_ptr(segment_m_head),
                    cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<kFusedGdrBlockDv>{}),
                                      cute::make_stride(cute::Int<kHeadDim>{}, cute::Int<1>{})));
                FusedGdrStoreStateFragmentGlobal<__nv_bfloat16>(tCrM, g_m, thr_m, role_tid);
            }
            return;
        }
    }

    static constexpr int kThreads   = kFusedGdrThreads;
    static constexpr int kMinBlocks = 1;

    static __device__ __forceinline__ void Run(const CUtensorMap* __restrict__ tma_desc_workspace,
                                               const float* __restrict__ g_cumsum,
                                               const float* __restrict__ beta,
                                               __nv_bfloat16* __restrict__ segment_state,
                                               __nv_bfloat16* __restrict__ segment_m,
                                               const int32_t* __restrict__ q_offsets,
                                               const int32_t* __restrict__ cp_source_indices,
                                               const int32_t* __restrict__ cp_q_offsets,
                                               const bool* __restrict__ cp_finished,
                                               bool* __restrict__ cp_fallback,
                                               int            token_num,
                                               int            sequence_num,
                                               int            hq,
                                               int            hv,
                                               int64_t        gate_stride,
                                               int64_t        gate_batch_stride,
                                               int64_t        beta_stride,
                                               int64_t        beta_batch_stride,
                                               unsigned char* smem_raw)
    {
        auto& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

        const int segment_id = static_cast<int>(blockIdx.x);
        const int value_head = static_cast<int>(blockIdx.y);
        auto&     warmup     = *reinterpret_cast<WarmupMetadata*>(smem_raw);
        if (threadIdx.x == 0) {
            warmup                                    = ComputeWarmupMetadata(g_cumsum,
                                           q_offsets,
                                           cp_source_indices,
                                           cp_q_offsets,
                                           segment_id,
                                           value_head,
                                           token_num,
                                           sequence_num,
                                           gate_stride,
                                           gate_batch_stride);
            cp_fallback[segment_id * hv + value_head] = warmup.fallback != 0;
        }
        __syncthreads();
        const int  warmup_chunks = warmup.chunks;
        const bool calc_m        = warmup.fallback != 0;
        // All lanes must consume the temporary metadata before Body reuses the
        // same dynamic shared-memory arena for its pipeline storage.
        __syncthreads();
        if (calc_m) {
            Body<true>(smem,
                       tma_desc_workspace,
                       g_cumsum,
                       beta,
                       segment_state,
                       segment_m,
                       q_offsets,
                       cp_source_indices,
                       cp_q_offsets,
                       cp_finished,
                       warmup_chunks,
                       token_num,
                       sequence_num,
                       hq,
                       hv,
                       gate_stride,
                       gate_batch_stride,
                       beta_stride,
                       beta_batch_stride);
        }
        else {
            Body<false>(smem,
                        tma_desc_workspace,
                        g_cumsum,
                        beta,
                        segment_state,
                        segment_m,
                        q_offsets,
                        cp_source_indices,
                        cp_q_offsets,
                        cp_finished,
                        warmup_chunks,
                        token_num,
                        sequence_num,
                        hq,
                        hv,
                        gate_stride,
                        gate_batch_stride,
                        beta_stride,
                        beta_batch_stride);
        }
    }
};

template<class T>
__global__ __launch_bounds__(Sm90FusedGdrH<T>::kThreads, Sm90FusedGdrH<T>::kMinBlocks) void Sm90FusedGdrHKernel(
    const CUtensorMap* __restrict__ tma_desc_workspace,
    const float* __restrict__ g_cumsum,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ segment_state,
    __nv_bfloat16* __restrict__ segment_m,
    const int32_t* __restrict__ q_offsets,
    const int32_t* __restrict__ cp_source_indices,
    const int32_t* __restrict__ cp_q_offsets,
    const bool* __restrict__ cp_finished,
    bool* __restrict__ cp_fallback,
    int     token_num,
    int     sequence_num,
    int     hq,
    int     hv,
    int64_t gate_stride,
    int64_t gate_batch_stride,
    int64_t beta_stride,
    int64_t beta_batch_stride)
{
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    Sm90FusedGdrH<T>::Run(tma_desc_workspace,
                          g_cumsum,
                          beta,
                          segment_state,
                          segment_m,
                          q_offsets,
                          cp_source_indices,
                          cp_q_offsets,
                          cp_finished,
                          cp_fallback,
                          token_num,
                          sequence_num,
                          hq,
                          hv,
                          gate_stride,
                          gate_batch_stride,
                          beta_stride,
                          beta_batch_stride,
                          smem_raw);
}

void SetFusedGdrHSharedMemoryLimit(size_t smem_bytes)
{
    static const cudaError_t status = cudaFuncSetAttribute(
        Sm90FusedGdrHKernel<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
    TM_CUDA_CHECK(status);
}

template<int BlockDv>
void LaunchSm90FusedGdrHTyped(const core::Tensor&        k,
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
    using Kernel = Sm90FusedGdrH<__nv_bfloat16>;
    static_cast<void>(k);
    static_cast<void>(v);
    static_cast<void>(resolvent);

    const dim3   grid(cp.total_segments, problem.hv, 1);
    const dim3   block(Kernel::kThreads);
    const size_t smem_bytes = Kernel::SharedBytes();

    SetFusedGdrHSharedMemoryLimit(smem_bytes);
    Sm90FusedGdrHKernel<__nv_bfloat16>
        <<<grid, block, smem_bytes, stream>>>(reinterpret_cast<CUtensorMap*>(tma_desc_workspace),
                                              g_cumsum.data<float>(),
                                              beta.data<float>(),
                                              segment_state.data<__nv_bfloat16>(),
                                              segment_m.data<__nv_bfloat16>(),
                                              q_offsets.data<int32_t>(),
                                              cp_source_indices.data<int32_t>(),
                                              cp_q_offsets.data<int32_t>(),
                                              cp_finished.data<bool>(),
                                              cp_fallback.data<bool>(),
                                              problem.token_num,
                                              problem.sequence_num,
                                              problem.hq,
                                              problem.hv,
                                              problem.gate_stride,
                                              problem.gate_batch_stride,
                                              problem.beta_stride,
                                              problem.beta_batch_stride);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
