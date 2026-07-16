// Inspired by
// https://github.com/QwenLM/FlashQLA/blob/60f81453143e724bcaf3fc7921e71e7328f6ebcd/flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_h.py

#pragma once

#include "src/turbomind/kernels/linear_attn/kernel/sm_120/common.h"

namespace turbomind::linear_attn::delta_rule {
namespace {

constexpr const char* kSm120FusedGdrHUnsupportedMessage = "chunk32 fused GDR H requires fixed CP segment tensors";

template<class T, int BlockDv>
struct alignas(1024) Sm120FusedGdrHEarlyStage {
    alignas(1024) T a[kChunk32Size][kChunk32Size];
    alignas(1024) T v[kChunk32Size][BlockDv];
    alignas(1024) float gate_stage[2][kChunk32Size][4];
    alignas(1024) float g[kChunk32Size];
    alignas(1024) float g_exp[kChunk32Size];
};

template<class T, int BlockDv>
struct alignas(1024) Sm120FusedGdrHSharedStorage {
    alignas(1024) T k_stage[2][kChunk32Size][kHeadDim];
    alignas(1024) Sm120FusedGdrHEarlyStage<T, BlockDv> early[2];
    alignas(1024) float vd[kHeadDim][BlockDv];
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

template<class T, int BlockDv>
constexpr size_t Sm120FusedGdrHSharedBytes()
{
    static_assert(BlockDv == kContextParallelGdrBlockDv);
    return sizeof(Sm120FusedGdrHSharedStorage<T, BlockDv>);
}

static_assert(Sm120FusedGdrHSharedBytes<__nv_bfloat16, kContextParallelGdrBlockDv>() <= kFusedGdrMaxDynamicSharedBytes);

template<class T, int BlockDv>
__global__ __launch_bounds__(kFusedGdrThreads,
                             1) void Sm120FusedGdrHKernel(const CUtensorMap* __restrict__ tma_desc_workspace,
                                                          const float* __restrict__ beta,
                                                          const int32_t* __restrict__ q_offsets,
                                                          const int32_t* __restrict__ cp_source_indices,
                                                          const int32_t* __restrict__ cp_q_offsets,
                                                          const bool* __restrict__ cp_finished,
                                                          int     sequence_num,
                                                          int     token_num,
                                                          int     hq,
                                                          int     hv,
                                                          int64_t beta_stride,
                                                          int64_t beta_batch_stride)
{
    static_assert(BlockDv == kContextParallelGdrBlockDv);
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    auto& smem       = *reinterpret_cast<Sm120FusedGdrHSharedStorage<T, BlockDv>*>(smem_raw);
    using MmaElement = typename FusedGdrMmaTraits<T>::Element;

    const int  tid             = static_cast<int>(threadIdx.x);
    const int  wg_idx          = cutlass::canonical_warp_group_idx();
    const int  role_tid        = tid % kFusedGdrRoleThreads;
    const bool producer_leader = wg_idx == 3 && role_tid == 0;
    const int  segment_id      = static_cast<int>(blockIdx.x);
    const int  sequence_id     = cp_source_indices[segment_id];
    if (sequence_id < 0 || sequence_id >= sequence_num) {
        return;
    }
    const int sequence_begin = q_offsets[sequence_id];
    const int segment_begin  = cp_q_offsets[segment_id];
    const int segment_end    = cp_q_offsets[segment_id + 1];
    const int seq_len        = segment_end - segment_begin;
    if (seq_len <= 0 || cp_finished[segment_id]) {
        return;
    }
    const int     token_base             = segment_begin - sequence_begin;
    const int     physical_batch         = sequence_begin / token_num;
    const int     local_sequence_begin   = sequence_begin - physical_batch * token_num;
    constexpr int kDvTilesPerHead        = kHeadDim / BlockDv;
    const int     head_tile              = static_cast<int>(blockIdx.y);
    const int     value_head             = head_tile / kDvTilesPerHead;
    const int     dv_tile                = head_tile - value_head * kDvTilesPerHead;
    const int     dv0                    = dv_tile * BlockDv;
    const int     qk_head                = value_head / (hv / hq);
    const int     value_tma_coord        = FusedGdrValueTmaCoord(value_head, dv0);
    const int     resolvent_tma_coord    = FusedGdrResolventTmaCoord<kChunk32Size>(value_head);
    const int     gate_tma_coord         = FusedGdrGateTmaCoord(value_head);
    const int     qk_tma_head_coord      = qk_head;
    const int     chunks                 = CeilDivDevice(seq_len, kChunk32Size);
    const auto    slices                 = MakeFusedGdrHTmaDescriptorSlices(tma_desc_workspace, sequence_num);
    const auto*   data_desc              = slices.data + sequence_id * kFusedGdrHDataDescCount;
    const auto*   k_tma_desc             = &data_desc[kFusedGdrHKDesc];
    const auto*   v_tma_desc             = &data_desc[kFusedGdrHVDesc];
    const auto*   g_tma_desc             = &data_desc[kFusedGdrHGDesc];
    const auto*   resolvent_tma_desc     = &data_desc[kFusedGdrHResolventDesc];
    const auto*   segment_state_tma_desc = slices.segment_state;
    const auto*   segment_m_tma_desc     = slices.segment_m;

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
        cute::initialize_barrier(smem.state_ready_mbar, kFusedGdrGateWriterThreads);
        cute::initialize_barrier(smem.early_ready_mbar[0], 1);
        cute::initialize_barrier(smem.early_ready_mbar[1], 1);
        cute::initialize_barrier(smem.early_free_bar[0], kFusedGdrConsumerThreads);
        cute::initialize_barrier(smem.early_free_bar[1], kFusedGdrConsumerThreads);
        cute::initialize_barrier(smem.k_ready_mbar[0], 1);
        cute::initialize_barrier(smem.k_ready_mbar[1], 1);
        cute::initialize_barrier(smem.k_free_bar[0], kFusedGdrConsumerThreads);
        cute::initialize_barrier(smem.k_free_bar[1], kFusedGdrConsumerThreads);
        cute::initialize_barrier(smem.h_final_ready_bar, kFusedGdrRoleThreads);
        cute::initialize_barrier(smem.m_final_ready_bar, kFusedGdrRoleThreads);
        cute::initialize_barrier(smem.h_pack_ready_bar, kFusedGdrRoleThreads);
        cute::initialize_barrier(smem.m_pack_ready_bar, kFusedGdrRoleThreads);
        cute::initialize_barrier(smem.scratch_free_bar, kFusedGdrRoleThreads);
        cute::initialize_barrier(smem.a_read_done_bar, kFusedGdrRoleThreads);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (wg_idx == 3) {
        cutlass::arch::warpgroup_reg_dealloc<kSm120FusedGdrHProducerRegisters>();

        if (role_tid < kCudaWarpThreads && chunks > 0) {
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
                cute::SM90_TMA_LOAD_2D::copy(
                    v_tma_desc, &smem.early_ready_mbar[0], kTmaNoCacheHint, &early0.v[0][0], value_tma_coord, token0);
                cute::SM90_TMA_LOAD_2D::copy(resolvent_tma_desc,
                                             &smem.early_ready_mbar[0],
                                             kTmaNoCacheHint,
                                             &early0.a[0][0],
                                             resolvent_tma_coord,
                                             token0);
                cute::SM90_TMA_LOAD_2D::copy(g_tma_desc,
                                             &smem.early_ready_mbar[0],
                                             kTmaNoCacheHint,
                                             &early0.gate_stage[0][0][0],
                                             gate_tma_coord,
                                             token0);

                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&smem.k_ready_mbar[0],
                                                                               kChunk32Size * kQkTmaBytesPerRow);
                cute::SM90_TMA_LOAD_4D::copy(k_tma_desc,
                                             &smem.k_ready_mbar[0],
                                             kTmaNoCacheHint,
                                             &smem.k_stage[0][0][0],
                                             0,
                                             0,
                                             qk_tma_head_coord,
                                             token0);
            }
        }

        int early_free_phase0 = 0;
        int early_free_phase1 = 0;
        int k_free_phase0     = 0;
        int k_free_phase1     = 0;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int next_chunk = chunk + 1;

            if (role_tid < kCudaWarpThreads) {
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
                        cute::SM90_TMA_LOAD_2D::copy(v_tma_desc,
                                                     &smem.early_ready_mbar[next_buf],
                                                     kTmaNoCacheHint,
                                                     &next_early.v[0][0],
                                                     value_tma_coord,
                                                     next_token0);
                        cute::SM90_TMA_LOAD_2D::copy(resolvent_tma_desc,
                                                     &smem.early_ready_mbar[next_buf],
                                                     kTmaNoCacheHint,
                                                     &next_early.a[0][0],
                                                     resolvent_tma_coord,
                                                     next_token0);
                        cute::SM90_TMA_LOAD_2D::copy(g_tma_desc,
                                                     &smem.early_ready_mbar[next_buf],
                                                     kTmaNoCacheHint,
                                                     &next_early.gate_stage[0][0][0],
                                                     gate_tma_coord,
                                                     next_token0);

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
                        cute::SM90_TMA_LOAD_4D::copy(k_tma_desc,
                                                     &smem.k_ready_mbar[next_buf],
                                                     kTmaNoCacheHint,
                                                     &smem.k_stage[next_buf][0][0],
                                                     0,
                                                     0,
                                                     qk_tma_head_coord,
                                                     next_token0);
                    }
                }
            }
        }
        if (producer_leader) {
            auto* h_state_stage_ptr = &smem.vd[0][0];
            cute::wait_barrier(smem.h_final_ready_bar, 0);
            cute::tma_store_fence();
            cute::SM90_TMA_STORE_4D::copy(segment_state_tma_desc, h_state_stage_ptr, dv0, 0, value_head, segment_id);
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();

            auto* m_state_stage_ptr = reinterpret_cast<float*>(&smem.k_stage[0][0][0]);
            cute::wait_barrier(smem.m_final_ready_bar, 0);
            cute::tma_store_fence();
            cute::SM90_TMA_STORE_4D::copy(segment_m_tma_desc, m_state_stage_ptr, dv0, 0, value_head, segment_id);
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
        return;
    }
    else if (wg_idx == 0) {
        cutlass::arch::warpgroup_reg_alloc<kSm120FusedGdrHStateRegisters>();
        using Element = typename FusedGdrMmaTraits<T>::Element;

        using MmaAtom = typename FusedGdrMmaTraits<T>::Atom;
        using Mma     = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
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
            const int gate_warp     = role_tid / kCudaWarpThreads;
            const int gate_warp_tid = role_tid % kCudaWarpThreads;
            if (gate_warp_tid < kFusedGdrGateRowsPerWarp) {
#pragma unroll
                for (int pass = 0; pass < kSm120FusedGdrGatePasses; ++pass) {
                    const int row =
                        pass * kFusedGdrGateWriterThreads + gate_warp * kFusedGdrGateRowsPerWarp + gate_warp_tid;
                    const float g_value = early.gate_stage[0][row][gate_lane];
                    const float g_exp   = FastExp(g_value);
                    early.g_exp[row]    = g_exp;
                    early.g[row]        = 1.0f / g_exp;
                }
            }
            const float state_decay = FastExp(early.gate_stage[0][last_row][gate_lane]);
            if (gate_warp_tid < kFusedGdrGateRowsPerWarp) {
                cute::arrive_barrier(smem.state_ready_mbar);
            }

            if (chunk > 0) {
                cute::wait_barrier(smem.scratch_free_bar, (chunk - 1) & 1);
            }
            Element* state_pack = reinterpret_cast<Element*>(&smem.vd[0][0]);
            Sm120FusedGdrStoreStateFragmentBf16Stsm<T, BlockDv>(tCrState, state_pack, thr_mma, role_tid);
            cute::arrive_barrier(smem.h_pack_ready_bar);

            FusedGdrDecayStateFragment(tCrState, state_decay);

            const int k_phase = (chunk >> 1) & 1;
            cute::wait_barrier(smem.k_ready_mbar[data_phase], k_phase);
            cutlass::arch::NamedBarrier::sync(2 * kFusedGdrRoleThreads, kFusedGdrBarrierHVdReady);
            Element* k_stage  = reinterpret_cast<Element*>(&smem.k_stage[data_phase][0][0]);
            Element* vd_stage = reinterpret_cast<Element*>(&early.v[0][0]);
            auto     s_k_t    = cute::make_tensor(cute::make_smem_ptr(k_stage), Sm120FusedGdrQkTransposedLayout());
            auto     s_vd_t   = cute::make_tensor(cute::make_smem_ptr(vd_stage), Sm120FusedGdrVTLayout<BlockDv>());

            FusedGdrStateUpdateFragmentFromScaledVd<T>(role_tid, mma, s_k_t, s_vd_t, tCrState);
            cute::arrive_barrier(smem.early_free_bar[data_phase]);
            cute::arrive_barrier(smem.k_free_bar[data_phase]);
        }

        if (chunks > 0) {
            const int final_data_phase = (chunks - 1) & 1;
            const int final_k_phase    = ((chunks - 1) >> 1) & 1;
            cute::wait_barrier(smem.k_free_bar[final_data_phase], final_k_phase);
        }
        auto* state_stage_ptr = &smem.vd[0][0];
        auto  s_final_stage =
            cute::make_tensor(cute::make_smem_ptr(state_stage_ptr),
                              cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<BlockDv>{}),
                                                cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{})));
        FusedGdrStoreStateFragmentFloat(tCrState, tCcState, s_final_stage);
        FusedGdrMmaSyncNamed<kFusedGdrBarrierStateUpdate>();
        cute::arrive_barrier(smem.h_final_ready_bar);
        return;
    }
    else if (wg_idx == 1) {
        cutlass::arch::warpgroup_reg_alloc<kSm120FusedGdrHStateRegisters>();
        using Element = typename FusedGdrMmaTraits<T>::Element;

        using MmaAtom = typename FusedGdrMmaTraits<T>::Atom;
        using Mma     = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
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
            cutlass::arch::NamedBarrier::sync(2 * kFusedGdrRoleThreads, kFusedGdrBarrierHStateReadDone);

            const float state_decay = early.g_exp[last_row];
            Element*    state_pack  = reinterpret_cast<Element*>(&smem.vd[0][0]);
            Sm120FusedGdrStoreStateFragmentBf16Stsm<T, BlockDv>(tCrState, state_pack, thr_mma, role_tid);
            cute::arrive_barrier(smem.m_pack_ready_bar);

            FusedGdrDecayStateFragment(tCrState, state_decay);

            cute::wait_barrier(smem.k_ready_mbar[data_phase], stage_phase);
            cutlass::arch::NamedBarrier::sync(2 * kFusedGdrRoleThreads, kFusedGdrBarrierMVdReady);
            Element* k_stage  = reinterpret_cast<Element*>(&smem.k_stage[data_phase][0][0]);
            Element* vd_stage = reinterpret_cast<Element*>(&early.a[0][0]);
            auto     s_k_t    = cute::make_tensor(cute::make_smem_ptr(k_stage), Sm120FusedGdrQkTransposedLayout());
            auto     s_vd_t   = cute::make_tensor(cute::make_smem_ptr(vd_stage), Sm120FusedGdrVTLayout<BlockDv>());

            FusedGdrStateUpdateFragmentFromScaledVd<T>(role_tid, mma, s_k_t, s_vd_t, tCrState);
            cute::arrive_barrier(smem.early_free_bar[data_phase]);
            cute::arrive_barrier(smem.k_free_bar[data_phase]);
        }

        if (chunks > 0) {
            const int final_data_phase = (chunks - 1) & 1;
            const int final_k_phase    = ((chunks - 1) >> 1) & 1;
            cute::wait_barrier(smem.k_free_bar[final_data_phase], final_k_phase);
        }
        auto* state_stage_ptr = reinterpret_cast<float*>(&smem.k_stage[0][0][0]);
        auto  s_final_stage =
            cute::make_tensor(cute::make_smem_ptr(state_stage_ptr),
                              cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<BlockDv>{}),
                                                cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{})));
        FusedGdrStoreStateFragmentFloat(tCrState, tCcState, s_final_stage);
        FusedGdrMmaSyncNamed<kFusedGdrBarrierOutputState>();
        cute::arrive_barrier(smem.m_final_ready_bar);
        return;
    }
    else if (wg_idx == 2) {
        cutlass::arch::warpgroup_reg_alloc<kSm120FusedGdrHValueRegisters>();
        using Element = typename FusedGdrMmaTraits<T>::Element;

        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int segment_token0 = chunk * kChunk32Size;
            const int remaining      = seq_len - segment_token0;
            const int valid          = remaining < kChunk32Size ? remaining : kChunk32Size;
            const int last_row       = valid - 1;
            const int data_phase     = chunk & 1;
            const int stage_phase    = (chunk >> 1) & 1;
            auto&     early          = smem.early[data_phase];
            Element*  k_stage        = reinterpret_cast<Element*>(&smem.k_stage[data_phase][0][0]);
            auto      s_k_smem       = cute::make_tensor(cute::make_smem_ptr(k_stage), Sm120FusedGdrQkLayout());
            auto      s_a_smem = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&early.a[0][0])),
                                              Sm120FusedGdrSquareLayout());
            Element*  w_pack   = reinterpret_cast<Element*>(&early.v[0][0]);

            cute::wait_barrier(smem.early_ready_mbar[data_phase], stage_phase);
            cute::wait_barrier(smem.state_ready_mbar, chunk & 1);

            const int gate_lane = value_head & 3;
            using MmaAtom       = typename FusedGdrMmaTraits<T>::Atom;
            using Mma           = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
                                       cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                       cute::Tile<cute::Underscore, cute::Int<32>, cute::Underscore>>;

            cute::wait_barrier(smem.k_ready_mbar[data_phase], stage_phase);

#pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                Mma      mma;
                auto     thr_mma     = mma.get_thread_slice(role_tid);
                Element* state_stage = reinterpret_cast<Element*>(&smem.vd[0][0]);
                Element* w_stage     = pass == 0 ? w_pack : state_stage;
                Element* vd_stage =
                    pass == 0 ? reinterpret_cast<Element*>(&early.v[0][0]) : reinterpret_cast<Element*>(&early.a[0][0]);
                auto s_w_row = cute::make_tensor(cute::make_smem_ptr(w_stage), Sm120FusedGdrVRowLayout<BlockDv>());
                auto s_state =
                    cute::make_tensor(cute::make_smem_ptr(state_stage), Sm120FusedGdrStateTLayout<BlockDv>());
                auto s_k_state = cute::make_tensor(
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
                auto        s_w  = cute::make_tensor(cute::make_smem_ptr(w_stage), Sm120FusedGdrVTLayout<BlockDv>());
                auto        s_vd = cute::make_tensor(cute::make_smem_ptr(vd_stage), Sm120FusedGdrVRowLayout<BlockDv>());
                auto        tCsVdStore = smem_thr_copy_C.partition_D(s_vd);
                const float last_g_exp = early.g_exp[last_row];

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
                    cutlass::arch::NamedBarrier::arrive(2 * kFusedGdrRoleThreads, kFusedGdrBarrierHStateReadDone);

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
                    FusedGdrMmaSyncNamed<kFusedGdrBarrierValueU>();
                }
#pragma unroll
                for (int i = 0; i < cute::size(tCrC); ++i) {
                    auto        coord       = tCcC(i);
                    const int   row         = cute::get<0>(coord);
                    const float state_v     = early.g_exp[row] * tCrC(i);
                    const float input_v     = pass == 0 ? static_cast<float>(tCrW(i)) : 0.0f;
                    const float input_scale = early.g[row] * early.gate_stage[1][row][gate_lane];
                    tCrW(i)                 = Element(CastFromFloat<T>((input_v - state_v) * input_scale));
                }
                cute::copy(smem_tiled_copy_C, tCrWStoreView, tCsWStore);
                FusedGdrMmaSyncNamed<kFusedGdrBarrierValueU>();

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
                    tCrW(i)          = Element(CastFromFloat<T>(gate * tCrC(i)));
                }
                if (pass == 0) {
                    FusedGdrMmaSyncNamed<kFusedGdrBarrierValueU>();
                    cute::copy(smem_tiled_copy_C, tCrWStoreView, tCsVdStore);
                    cutlass::arch::NamedBarrier::arrive(2 * kFusedGdrRoleThreads, kFusedGdrBarrierHVdReady);
                }
                else {
                    cute::arrive_barrier(smem.scratch_free_bar);
                    cute::wait_barrier(smem.a_read_done_bar, chunk & 1);
                    cute::copy(smem_tiled_copy_C, tCrWStoreView, tCsVdStore);
                    cutlass::arch::NamedBarrier::arrive(2 * kFusedGdrRoleThreads, kFusedGdrBarrierMVdReady);
                }
            }
            cute::arrive_barrier(smem.early_free_bar[data_phase]);
            cute::arrive_barrier(smem.k_free_bar[data_phase]);
        }
        return;
    }
}

template<int BlockDv>
void SetFusedGdrHSharedMemoryLimit(size_t smem_bytes)
{
    static_assert(BlockDv == kContextParallelGdrBlockDv);
    static const cudaError_t status = cudaFuncSetAttribute(Sm120FusedGdrHKernel<__nv_bfloat16, BlockDv>,
                                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                           static_cast<int>(smem_bytes));
    TM_CUDA_CHECK(status);
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
                               const core::Tensor&        cp_fallback,
                               void*                      tma_desc_workspace,
                               cudaStream_t               stream)
{
    static_assert(BlockDv == kContextParallelGdrBlockDv);
    static_cast<void>(k);
    static_cast<void>(v);
    static_cast<void>(g_cumsum);
    static_cast<void>(resolvent);
    static_cast<void>(segment_state);
    static_cast<void>(segment_m);
    static_cast<void>(cp_fallback);

    if (problem.arch != 1200 || problem.input_dtype != kBfloat16 || problem.hv % problem.hq != 0
        || problem.head_dim != kHeadDim || problem.chunk_size != kChunk32Size) {
        throw std::invalid_argument(kSm120FusedGdrHUnsupportedMessage);
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
    const int     dv_tiles = CeilDiv(kHeadDim, block_dv);
    const dim3    grid(cp.total_segments, problem.hv * dv_tiles, 1);
    const dim3    block(kFusedGdrThreads);
    const size_t  smem_bytes = Sm120FusedGdrHSharedBytes<__nv_bfloat16, block_dv>();

    SetFusedGdrHSharedMemoryLimit<block_dv>(smem_bytes);
    Sm120FusedGdrHKernel<__nv_bfloat16, block_dv>
        <<<grid, block, smem_bytes, stream>>>(reinterpret_cast<CUtensorMap*>(tma_desc_workspace),
                                              beta.data<float>(),
                                              q_offsets.data<int32_t>(),
                                              cp_source_indices.data<int32_t>(),
                                              cp_q_offsets.data<int32_t>(),
                                              cp_finished.data<bool>(),
                                              problem.sequence_num,
                                              problem.token_num,
                                              problem.hq,
                                              problem.hv,
                                              problem.beta_stride,
                                              problem.beta_batch_stride);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace
}  // namespace turbomind::linear_attn::delta_rule
