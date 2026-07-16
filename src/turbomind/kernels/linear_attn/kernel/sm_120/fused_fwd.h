// Inspired by
// https://github.com/QwenLM/FlashQLA/blob/60f81453143e724bcaf3fc7921e71e7328f6ebcd/flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py

#pragma once

#include "src/turbomind/kernels/linear_attn/kernel/sm_120/common.h"

namespace turbomind::linear_attn::delta_rule {
namespace {

constexpr const char* kSm120FusedGdrFwdUnsupportedMessage =
    "chunk32 fused GDR forward supports only the SM120 bf16 chunked target shape "
    "(int32 q_offsets, bool finished mask, head_dim=128, chunk_size=32, "
    "gate stride divisible by 4, Hv % Hq == 0)";

// SM120 grouped-bf16 chunked GDR CTA. State-ptr chunked only; StateT is the external state dtype.
template<class T, class StateT, int BlockDv>
__global__ __launch_bounds__(kFusedGdrThreads,
                             1) void Sm120FusedGdrFwdKernel(const CUtensorMap* __restrict__ tma_desc_workspace,
                                                            const int32_t* __restrict__ q_offsets,
                                                            const bool* __restrict__ finished,
                                                            const int32_t* __restrict__ data_q_offsets,
                                                            const int32_t* __restrict__ cp_source_indices,
                                                            int data_sequence_num,
                                                            int hq,
                                                            int hv)
{
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    static_assert(kFusedGdrValidStateT<StateT>, "fused chunk GDR StateT must be float or bfloat16");
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    auto& smem       = *reinterpret_cast<Sm120FusedGdrSharedStorage<T, BlockDv>*>(smem_raw);
    using MmaElement = typename FusedGdrMmaTraits<T>::Element;

    const int  tid              = static_cast<int>(threadIdx.x);
    const int  wg_idx           = cutlass::canonical_warp_group_idx();
    const int  role_tid         = tid % kFusedGdrRoleThreads;
    const int  batch_id         = static_cast<int>(blockIdx.x);
    const int  value_head       = static_cast<int>(blockIdx.y);
    const int  dv_tile          = static_cast<int>(blockIdx.z);
    const int  dv0              = dv_tile * BlockDv;
    const int  qk_head          = value_head / (hv / hq);
    const bool context_parallel = cp_source_indices != nullptr;
    const int  segment_id       = batch_id;
    const int  sequence_id      = context_parallel ? cp_source_indices[segment_id] : batch_id;
    if (context_parallel && (sequence_id < 0 || sequence_id >= data_sequence_num)) {
        return;
    }
    const int seq_start = q_offsets[batch_id];
    const int seq_end   = q_offsets[batch_id + 1];
    const int seq_len   = seq_end - seq_start;
    if (seq_len <= 0) {
        return;
    }
    const int sequence_begin = context_parallel ? data_q_offsets[sequence_id] : seq_start;
    if (context_parallel) {
        const int data_seq_end = data_q_offsets[sequence_id + 1];
        if (seq_start < sequence_begin || seq_end > data_seq_end) {
            return;
        }
    }
    const int  token_base        = context_parallel ? seq_start - sequence_begin : 0;
    const int  qk_tma_head_coord = qk_head;
    const int  gate_tma_coord    = FusedGdrGateTmaCoord(value_head);
    const int  chunks            = CeilDivDevice(seq_len, kChunk32Size);
    const auto direct_slices     = MakeFusedGdrTmaDescriptorSlices(tma_desc_workspace, data_sequence_num);
    const auto context_parallel_fused_gdr_slices =
        MakeContextParallelFusedGdrTmaDescriptorSlices(tma_desc_workspace, data_sequence_num);
    const auto* data_desc  = context_parallel ?
                                 context_parallel_fused_gdr_slices.data + sequence_id * kFusedGdrDataDescCount :
                                 direct_slices.data + batch_id * kFusedGdrDataDescCount;
    const auto* state_desc = context_parallel ?
                                 context_parallel_fused_gdr_slices.cp_state :
                                 direct_slices.state + (batch_id * hv + value_head) * kFusedGdrStateDescCount;
    FusedGdrAcquireAndPrefetchDataTmaDescriptors(data_desc, tid);
    FusedGdrAcquireAndPrefetchStateTmaDescriptor(state_desc, tid);
    const CUtensorMap* q_desc         = &data_desc[kFusedGdrQDesc];
    const CUtensorMap* k_desc         = &data_desc[kFusedGdrKDesc];
    const CUtensorMap* v_desc         = &data_desc[kFusedGdrVDesc];
    const CUtensorMap* g_desc         = &data_desc[kFusedGdrGDesc];
    const CUtensorMap* beta_desc      = &data_desc[kFusedGdrBetaDesc];
    const CUtensorMap* resolvent_desc = &data_desc[kFusedGdrResolventDesc];
    const CUtensorMap* out_desc       = &data_desc[kFusedGdrOutDesc];

    if (tid == 0) {
        cute::initialize_barrier(smem.state_tma_mbar, 1);

        cute::initialize_barrier(smem.state_ready_mbar, kFusedGdrProducerThreads);
        cute::initialize_barrier(smem.gate_ready_mbar0, 1);
        cute::initialize_barrier(smem.gate_ready_mbar1, 1);
        cute::initialize_barrier(smem.early_ready_mbar0, 1);
        cute::initialize_barrier(smem.early_ready_mbar1, 1);
        cute::initialize_barrier(smem.q_ready_mbar0, 1);
        cute::initialize_barrier(smem.q_ready_mbar1, 1);
        cute::initialize_barrier(smem.k_ready_mbar0, 1);
        cute::initialize_barrier(smem.k_ready_mbar1, 1);
        cute::initialize_barrier(smem.q_store_done_bar0, 1);
        cute::initialize_barrier(smem.q_store_done_bar1, 1);
        cute::initialize_barrier(smem.out_ready_bar0, kFusedGdrRoleThreads);
        cute::initialize_barrier(smem.out_ready_bar1, kFusedGdrRoleThreads);
        cute::initialize_barrier(smem.early_free_bar0, kFusedGdrConsumerThreads);
        cute::initialize_barrier(smem.early_free_bar1, kFusedGdrConsumerThreads);
        cute::initialize_barrier(smem.compute_done_bar0, 2);
        cute::initialize_barrier(smem.compute_done_bar1, 2);
        cute::initialize_barrier(smem.update_ready_bar, kFusedGdrRoleThreads);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (wg_idx == 3) {
        cutlass::arch::warpgroup_reg_dealloc<kSm120FusedGdrProducerRegisters>();
        const bool store_leader = role_tid == 0;
        const bool q_leader     = role_tid == kCudaWarpThreads;
        const bool k_leader     = role_tid == 2 * kCudaWarpThreads;
        const bool early_leader = role_tid == 3 * kCudaWarpThreads;

        if (store_leader) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&smem.state_tma_mbar,
                                                                           FusedGdrStateTmaBytes<StateT, BlockDv>());
            if (context_parallel) {
                cute::SM90_TMA_LOAD_4D::copy(state_desc,
                                             &smem.state_tma_mbar,
                                             kTmaNoCacheHint,
                                             reinterpret_cast<StateT*>(&smem.state_stage[0][0]),
                                             dv0,
                                             0,
                                             value_head,
                                             segment_id);
            }
            else {
                cute::SM90_TMA_LOAD_2D::copy(state_desc,
                                             &smem.state_tma_mbar,
                                             kTmaNoCacheHint,
                                             reinterpret_cast<StateT*>(&smem.state_stage[0][0]),
                                             dv0,
                                             0);
            }
        }
        cute::wait_barrier(smem.state_tma_mbar, 0);
        FusedGdrUnpackStateTma<StateT, BlockDv>(smem.state_stage, role_tid, kFusedGdrProducerThreads);
        cute::arrive_barrier(smem.state_ready_mbar);

        if (store_leader) {
            cute::wait_barrier(smem.state_ready_mbar, 0);
        }

        constexpr int kQkTmaBytesPerRow   = kHeadDim * static_cast<int>(sizeof(T));
        constexpr int kGateTmaBytesPerRow = 2 * 4 * static_cast<int>(sizeof(float));
        constexpr int kEarlyTmaBytesPerRow =
            BlockDv * static_cast<int>(sizeof(T)) + kChunk32Size * static_cast<int>(sizeof(T));
        constexpr int kTmaBoxRows = kChunk32Size;

        // The physical Q/K/early/gate slots use buffer_phase. The single-slot vd scratch
        // is serialized by update_ready, while WG1/WG2 rendezvous around packed Vd.
        // Slot reuse gates: p/gate <- early_free, q <- q_store_done, k <- compute_done.
        if (early_leader) {
            for (int load_chunk = 0; load_chunk < chunks; ++load_chunk) {
                const int data_buf     = load_chunk & 1;
                const int buffer_phase = (load_chunk >> 1) & 1;
                if (load_chunk >= 2) {
                    auto& early_free_bar = data_buf == 0 ? smem.early_free_bar0 : smem.early_free_bar1;
                    cute::wait_barrier(early_free_bar, buffer_phase ^ 1);
                }
                auto&       gate_ready_mbar  = data_buf == 0 ? smem.gate_ready_mbar0 : smem.gate_ready_mbar1;
                auto&       early_ready_mbar = data_buf == 0 ? smem.early_ready_mbar0 : smem.early_ready_mbar1;
                const int   token0           = token_base + load_chunk * kChunk32Size;
                MmaElement* w_pack =
                    reinterpret_cast<MmaElement*>(&smem.p_stage[data_buf][0][0]) + kChunk32Size * kChunk32Size;
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&gate_ready_mbar,
                                                                               kTmaBoxRows * kGateTmaBytesPerRow);
                cute::SM90_TMA_LOAD_3D::copy(g_desc,
                                             &gate_ready_mbar,
                                             kTmaNoCacheHint,
                                             &smem.gate_stage[data_buf][0][0][0],
                                             gate_tma_coord,
                                             token0,
                                             0);
                cute::SM90_TMA_LOAD_3D::copy(beta_desc,
                                             &gate_ready_mbar,
                                             kTmaNoCacheHint,
                                             &smem.gate_stage[data_buf][1][0][0],
                                             gate_tma_coord,
                                             token0,
                                             0);

                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&early_ready_mbar,
                                                                               kTmaBoxRows * kEarlyTmaBytesPerRow);
                cute::SM90_TMA_LOAD_4D::copy(
                    v_desc, &early_ready_mbar, kTmaNoCacheHint, w_pack, dv0, value_head, token0, 0);
                cute::SM90_TMA_LOAD_4D::copy(resolvent_desc,
                                             &early_ready_mbar,
                                             kTmaNoCacheHint,
                                             &smem.p_stage[data_buf][0][0],
                                             0,
                                             value_head,
                                             token0,
                                             0);
            }
        }

        if (q_leader) {
            for (int load_chunk = 0; load_chunk < chunks; ++load_chunk) {
                const int data_buf     = load_chunk & 1;
                const int buffer_phase = (load_chunk >> 1) & 1;
                if (load_chunk >= 2) {
                    auto& q_store_done_bar = data_buf == 0 ? smem.q_store_done_bar0 : smem.q_store_done_bar1;
                    cute::wait_barrier(q_store_done_bar, buffer_phase ^ 1);
                }
                auto&     q_ready_mbar = data_buf == 0 ? smem.q_ready_mbar0 : smem.q_ready_mbar1;
                const int token0       = token_base + load_chunk * kChunk32Size;
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&q_ready_mbar,
                                                                               kTmaBoxRows * kQkTmaBytesPerRow);
                cute::SM90_TMA_LOAD_5D::copy(q_desc,
                                             &q_ready_mbar,
                                             kTmaNoCacheHint,
                                             &smem.q_stage[data_buf][0][0],
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
                    auto& compute_done_bar = data_buf == 0 ? smem.compute_done_bar0 : smem.compute_done_bar1;
                    cute::wait_barrier(compute_done_bar, buffer_phase ^ 1);
                }
                auto&     k_ready_mbar = data_buf == 0 ? smem.k_ready_mbar0 : smem.k_ready_mbar1;
                const int token0       = token_base + load_chunk * kChunk32Size;
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&k_ready_mbar,
                                                                               kTmaBoxRows * kQkTmaBytesPerRow);
                cute::SM90_TMA_LOAD_5D::copy(k_desc,
                                             &k_ready_mbar,
                                             kTmaNoCacheHint,
                                             &smem.k_stage[data_buf][0][0],
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

        if (context_parallel) {
            return;
        }

        if (store_leader && !finished[batch_id]) {
            const int final_buf      = (chunks - 1) & 1;
            const int final_phase    = ((chunks - 1) >> 1) & 1;
            auto&     final_done_bar = final_buf == 0 ? smem.compute_done_bar0 : smem.compute_done_bar1;
            cute::wait_barrier(final_done_bar, final_phase);
            cute::tma_store_fence();
            cute::SM90_TMA_STORE_2D::copy(state_desc, reinterpret_cast<StateT*>(&smem.state_stage[0][0]), dv0, 0);
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }

        return;
    }

    if (wg_idx == 0) {
        cutlass::arch::warpgroup_reg_alloc<kFusedGdrStateRegisters>();
        using Element = typename FusedGdrMmaTraits<T>::Element;

        cute::wait_barrier(smem.state_ready_mbar, 0);

        using MmaAtom = typename FusedGdrMmaTraits<T>::Atom;
        using Mma     = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
                                   cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                   cute::Tile<cute::Underscore, cute::Int<32>, cute::Underscore>>;

        Mma  mma;
        auto thr_mma = mma.get_thread_slice(role_tid);
        auto s_state_stage =
            cute::make_tensor(cute::make_smem_ptr(&smem.state_stage[0][0]),
                              cute::make_layout(cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<BlockDv>{}),
                                                cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{})));
        auto c_state  = cute::make_identity_tensor(cute::shape(s_state_stage));
        auto tCsState = thr_mma.partition_C(s_state_stage);
        auto tCcState = thr_mma.partition_C(c_state);
        auto tCrState = thr_mma.make_fragment_C(tCsState);
        FusedGdrLoadStateFragment(tCrState, tCcState, s_state_stage);

        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int token0       = chunk * kChunk32Size;
            const int remaining    = seq_len - token0;
            const int valid        = remaining < kChunk32Size ? remaining : kChunk32Size;
            const int last_row     = valid - 1;
            const int data_buf     = chunk & 1;
            const int buffer_phase = (chunk >> 1) & 1;
            const int chunk_phase  = chunk & 1;

            auto& gate_ready_mbar = data_buf == 0 ? smem.gate_ready_mbar0 : smem.gate_ready_mbar1;
            cute::wait_barrier(gate_ready_mbar, buffer_phase);

            const int   gate_lane     = value_head & 3;
            const int   gate_warp     = role_tid / kCudaWarpThreads;
            const int   gate_warp_tid = role_tid % kCudaWarpThreads;
            const float last_g_value  = smem.gate_stage[data_buf][0][last_row][gate_lane];
            if (gate_warp_tid < kFusedGdrGateRowsPerWarp) {
#pragma unroll
                for (int pass = 0; pass < kSm120FusedGdrGatePasses; ++pass) {
                    const int row =
                        pass * kFusedGdrGateWriterThreads + gate_warp * kFusedGdrGateRowsPerWarp + gate_warp_tid;
                    const float g_value = row < valid ? smem.gate_stage[data_buf][0][row][gate_lane] : last_g_value;
                    const float g_exp   = FastExp(g_value);
                    smem.g_exp[data_buf][row]     = g_exp;
                    smem.g[data_buf][row]         = g_value;
                    smem.g_rev_exp[data_buf][row] = row < valid ? FastExp(last_g_value - g_value) : 0.0f;
                }
            }
            const float state_decay    = FastExp(last_g_value);
            auto&       early_free_bar = data_buf == 0 ? smem.early_free_bar0 : smem.early_free_bar1;
            cute::arrive_barrier(early_free_bar);

            Element* state_pack = reinterpret_cast<Element*>(&smem.vd[0][0]);
            Sm120FusedGdrStoreStateFragmentBf16Stsm<T, BlockDv>(tCrState, state_pack, thr_mma, role_tid);
            FusedGdrConsumerArrive();

            FusedGdrDecayStateFragment(tCrState, state_decay);

            cute::wait_barrier(smem.update_ready_bar, chunk_phase);

            auto& k_ready_mbar = data_buf == 0 ? smem.k_ready_mbar0 : smem.k_ready_mbar1;
            cute::wait_barrier(k_ready_mbar, buffer_phase);
            auto s_k_t =
                cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.k_stage[data_buf][0][0])),
                                  Sm120FusedGdrQkTransposedLayout());
            auto s_vn_t = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.vd[0][0])),
                                            Sm120FusedGdrVTLayout<BlockDv>());

            FusedGdrStateUpdateFragmentFromScaledVd<T>(role_tid, mma, s_k_t, s_vn_t, tCrState);
            FusedGdrMmaSyncNamed<kFusedGdrBarrierStateUpdate>();
            if (chunk == chunks - 1 && !context_parallel) {
                if (chunk > 0) {
                    auto& q_ready_mbar = data_buf == 0 ? smem.q_ready_mbar0 : smem.q_ready_mbar1;
                    cute::wait_barrier(q_ready_mbar, buffer_phase);
                }
                if constexpr (std::is_same_v<StateT, float>) {
                    FusedGdrStoreStateFragmentFloat(tCrState, tCcState, s_state_stage);
                }
                else {
                    FusedGdrStoreStateFragmentBf16Tma<BlockDv>(
                        tCrState, tCcState, reinterpret_cast<__nv_bfloat16*>(&smem.state_stage[0][0]));
                }
                FusedGdrMmaSyncNamed<kFusedGdrBarrierStateUpdate>();
            }
            if (role_tid == 0) {
                auto& compute_done_bar = data_buf == 0 ? smem.compute_done_bar0 : smem.compute_done_bar1;
                cute::arrive_barrier(compute_done_bar);
            }
        }
        return;
    }
    else if (wg_idx == 1) {
        cutlass::arch::warpgroup_reg_alloc<kFusedGdrValueRegisters>();
        using Element = typename FusedGdrMmaTraits<T>::Element;

        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int token0       = chunk * kChunk32Size;
            const int remaining    = seq_len - token0;
            const int valid        = remaining < kChunk32Size ? remaining : kChunk32Size;
            const int data_buf     = chunk & 1;
            const int buffer_phase = (chunk >> 1) & 1;

            auto s_k_smem =
                cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.k_stage[data_buf][0][0])),
                                  Sm120FusedGdrQkLayout());
            auto s_a_smem =
                cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.p_stage[data_buf][0][0])),
                                  Sm120FusedGdrSquareLayout());
            Element* w_pack = reinterpret_cast<Element*>(&smem.p_stage[data_buf][0][0]) + kChunk32Size * kChunk32Size;
            auto&    early_ready_mbar = data_buf == 0 ? smem.early_ready_mbar0 : smem.early_ready_mbar1;
            cute::wait_barrier(early_ready_mbar, buffer_phase);

            FusedGdrConsumerSync();

            using MmaAtom = typename FusedGdrMmaTraits<T>::Atom;
            using Mma     = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
                                       cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                       cute::Tile<cute::Underscore, cute::Int<32>, cute::Underscore>>;

            Mma      mma;
            auto     thr_mma     = mma.get_thread_slice(role_tid);
            Element* packed_base = reinterpret_cast<Element*>(&smem.q_stage[data_buf][0][0]);
            Element* packed_vd   = packed_base + kChunk32PackedVdOffset;
            auto     s_w_row     = cute::make_tensor(cute::make_smem_ptr(w_pack), Sm120FusedGdrVRowLayout<BlockDv>());

            auto s_state = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.vd[0][0])),
                                             Sm120FusedGdrStateTLayout<BlockDv>());
            auto s_k_state =
                cute::make_tensor(cute::make_smem_ptr(&smem.vd[0][0]),
                                  cute::make_layout(cute::make_shape(cute::Int<kChunk32Size>{}, cute::Int<BlockDv>{}),
                                                    cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{})));
            auto c_k_state = cute::make_identity_tensor(cute::shape(s_k_state));

            auto     tCsC         = thr_mma.partition_C(s_k_state);
            auto     tCcC         = thr_mma.partition_C(c_k_state);
            auto     tCrC         = thr_mma.make_fragment_C(tCsC);
            auto     tCrW         = cute::make_fragment_like<Element>(tCrC);
            Element* vn_bf16      = reinterpret_cast<Element*>(&smem.vd[0][0]);
            auto&    k_ready_mbar = data_buf == 0 ? smem.k_ready_mbar0 : smem.k_ready_mbar1;
            cute::wait_barrier(k_ready_mbar, buffer_phase);
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
            FusedGdrMmaSyncNamed<kFusedGdrBarrierValueU>();
#pragma unroll
            for (int i = 0; i < cute::size(tCrC); ++i) {
                const int   row     = cute::get<0>(tCcC(i));
                const float v_value = static_cast<float>(tCrW(i));
                const float delta   = row < valid ? v_value - smem.g_exp[data_buf][row] * tCrC(i) : 0.0f;
                tCrW(i)             = Element(CastFromFloat<T>(delta));
            }
            Sm120FusedGdrStoreValueFragmentBf16Stsm<T, BlockDv>(tCrW, w_pack, thr_mma, role_tid);
            FusedGdrMmaSyncNamed<kFusedGdrBarrierValueU>();
            // Transformed A is independent of the state snapshot. Acquire it
            // without waiting for WG2's Q@state reads to finish.
            Sm120FusedGdrAgReadySync();

            auto s_w = cute::make_tensor(cute::make_smem_ptr(w_pack), Sm120FusedGdrVTLayout<BlockDv>());
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
            auto& early_free_bar = data_buf == 0 ? smem.early_free_bar0 : smem.early_free_bar1;
            cute::arrive_barrier(early_free_bar);

            // WG2 has stopped reading Q before WG1 reuses the disjoint packed-Vd
            // region of q_stage. WG2 packs P concurrently with this STSM.
            Sm120FusedGdrPackedVdSync();
            Sm120FusedGdrStoreValueFragmentBf16Stsm<T, BlockDv>(tCrC, packed_vd, thr_mma, role_tid);
            // Publish packed Vd before WG2 starts the local output GEMM.
            Sm120FusedGdrPackedVdSync();

#pragma unroll
            for (int i = 0; i < cute::size(tCrC); ++i) {
                const int   row = cute::get<0>(tCcC(i));
                const float vn  = smem.g_rev_exp[data_buf][row] * static_cast<float>(tCrC(i));
                tCrW(i)         = Element(CastFromFloat<T>(vn));
            }
            Sm120FusedGdrStoreValueFragmentBf16Stsm<T, BlockDv>(tCrW, vn_bf16, thr_mma, role_tid);
            cute::arrive_barrier(smem.update_ready_bar);
        }
        return;
    }
    else if (wg_idx == 2) {
        using Element = typename FusedGdrMmaTraits<T>::Element;

        cutlass::arch::warpgroup_reg_alloc<kFusedGdrOutputRegisters>();
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int token0       = chunk * kChunk32Size;
            const int remaining    = seq_len - token0;
            const int valid        = remaining < kChunk32Size ? remaining : kChunk32Size;
            const int data_buf     = chunk & 1;
            const int buffer_phase = (chunk >> 1) & 1;

            auto s_q_smem =
                cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.q_stage[data_buf][0][0])),
                                  Sm120FusedGdrQkLayout());
            auto s_k_smem =
                cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.k_stage[data_buf][0][0])),
                                  Sm120FusedGdrQkLayout());
            auto s_a_smem =
                cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.p_stage[data_buf][0][0])),
                                  Sm120FusedGdrSquareLayout());
            auto& early_ready_mbar = data_buf == 0 ? smem.early_ready_mbar0 : smem.early_ready_mbar1;
            cute::wait_barrier(early_ready_mbar, buffer_phase);

            FusedGdrConsumerSync();

            const int gate_lane = value_head & 3;
            using MmaAtom       = typename FusedGdrMmaTraits<T>::Atom;
            using Mma           = cute::TiledMMA<cute::MMA_Atom<MmaAtom>,
                                       cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                       cute::Tile<cute::Underscore, cute::Int<32>, cute::Underscore>>;

            Mma      mma;
            auto     thr_mma     = mma.get_thread_slice(role_tid);
            Element* packed_base = reinterpret_cast<Element*>(&smem.q_stage[data_buf][0][0]);
            Element* packed_p    = packed_base + kChunk32PackedPOffset;
            Element* packed_vd   = packed_base + kChunk32PackedVdOffset;

            auto s_p_float = cute::make_tensor(
                cute::make_smem_ptr(reinterpret_cast<float*>(&smem.q_stage[data_buf][0][0])),
                cute::make_layout(cute::make_shape(cute::Int<kChunk32Size>{}, cute::Int<kChunk32Size>{}),
                                  cute::make_stride(cute::Int<kChunk32Size>{}, cute::Int<1>{})));
            auto s_packed_p = cute::make_tensor(cute::make_smem_ptr(packed_p), Sm120FusedGdrPackedP128BRowLayout());
            auto c_p        = cute::make_identity_tensor(cute::shape(s_p_float));
            auto tCsP       = thr_mma.partition_C(s_p_float);
            auto tCcP       = thr_mma.partition_C(c_p);
            auto tCrGRel    = thr_mma.make_fragment_C(tCsP);

#pragma unroll
            for (int i = 0; i < cute::size(tCrGRel); ++i) {
                auto      coord = tCcP(i);
                const int row   = cute::get<0>(coord);
                const int col   = cute::get<1>(coord);
                float     g_rel = 0.0f;
                if (col <= row) {
                    float ag_value = 0.0f;
                    if (row < valid && col < valid) {
                        g_rel                = FastExp(smem.g[data_buf][row] - smem.g[data_buf][col]);
                        const float beta_col = smem.gate_stage[data_buf][1][col][gate_lane];
                        ag_value             = static_cast<float>(s_a_smem(row, col)) * g_rel * beta_col;
                    }
                    s_a_smem(row, col) = Element(CastFromFloat<T>(ag_value));
                }
                tCrGRel(i) = g_rel;
            }
            Sm120FusedGdrAgReadyArrive();
            auto& early_free_bar = data_buf == 0 ? smem.early_free_bar0 : smem.early_free_bar1;
            cute::arrive_barrier(early_free_bar);

            auto s_q_state_c =
                cute::make_tensor(cute::make_smem_ptr(&smem.p_stage[data_buf][0][0]),
                                  cute::make_layout(cute::make_shape(cute::Int<kChunk32Size>{}, cute::Int<BlockDv>{}),
                                                    cute::make_stride(cute::Int<BlockDv>{}, cute::Int<1>{})));
            auto tCsQState = thr_mma.partition_C(s_q_state_c);
            auto c_q_state = cute::make_identity_tensor(cute::shape(s_q_state_c));
            auto tCcQState = thr_mma.partition_C(c_q_state);
            auto tCrQState = thr_mma.make_fragment_C(tCsQState);
            auto s_state   = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element*>(&smem.vd[0][0])),
                                             Sm120FusedGdrStateTLayout<BlockDv>());

            auto& q_ready_mbar = data_buf == 0 ? smem.q_ready_mbar0 : smem.q_ready_mbar1;
            cute::wait_barrier(q_ready_mbar, buffer_phase);
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
                auto      coord = tCcQState(i);
                const int row   = cute::get<0>(coord);
                tCrQState(i)    = row < valid ? tCrQState(i) * kHeadScale * smem.g_exp[data_buf][row] : 0.0f;
            }

            auto  tCrC         = thr_mma.make_fragment_C(tCsP);
            auto& k_ready_mbar = data_buf == 0 ? smem.k_ready_mbar0 : smem.k_ready_mbar1;
            cute::wait_barrier(k_ready_mbar, buffer_phase);
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
            FusedGdrMmaSyncNamed<kFusedGdrBarrierOutputP>();
            // Release the packed-Vd region after all Q reads have completed.
            Sm120FusedGdrPackedVdSync();
#pragma unroll
            for (int i = 0; i < cute::size(tCrC); i += 2) {
                auto        coord    = tCcP(i);
                const int   row      = cute::get<0>(coord);
                const int   col      = cute::get<1>(coord);
                const int   next_col = col + 1;
                const float p0 = row < valid && col < valid && col <= row ? kHeadScale * tCrGRel(i) * tCrC(i) : 0.0f;
                const float p1 = row < valid && next_col < valid && next_col <= row ?
                                     kHeadScale * tCrGRel(i + 1) * tCrC(i + 1) :
                                     0.0f;
                // The C-fragment layout emits adjacent-column pairs, and even columns
                // stay contiguous under Swizzle<3,3,3>.
                FusedGdrStoreBf16Pair(&s_packed_p(row, col), make_float2(p0, p1));
            }
            if (role_tid == 0) {
                auto& compute_done_bar = data_buf == 0 ? smem.compute_done_bar0 : smem.compute_done_bar1;
                cute::arrive_barrier(compute_done_bar);
            }
            // Acquire WG1's packed Vd after this warp group has finished packing P.
            Sm120FusedGdrPackedVdSync();

            auto s_packed_p_local =
                cute::make_tensor(cute::make_smem_ptr(packed_p), Sm120FusedGdrPackedP128BRowLayout());
            auto  s_packed_vd = cute::make_tensor(cute::make_smem_ptr(packed_vd), Sm120FusedGdrVTLayout<BlockDv>());
            auto* out_stage   = reinterpret_cast<Element*>(&smem.q_stage[data_buf][0][0]);
            auto  s_out       = cute::make_tensor(cute::make_smem_ptr(out_stage), Sm120FusedGdrVRowLayout<BlockDv>());

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
            FusedGdrMmaSyncNamed<kFusedGdrBarrierOutputLocal>();

#pragma unroll
            for (int i = 0; i < cute::size(tCrLocal); ++i) {
                const float out_value = tCrQState(i) + tCrLocal(i);
                tCsOut(i)             = Element(CastFromFloat<T>(out_value));
            }
            auto& out_ready_bar = data_buf == 0 ? smem.out_ready_bar0 : smem.out_ready_bar1;
            cute::arrive_barrier(out_ready_bar);
        }
        return;
    }
}

template<class StateT, int BlockDv>
void SetFusedGdrFwdSharedMemoryLimit(size_t smem_bytes)
{
    static_assert(kFusedGdrValidStateT<StateT>, "fused chunk GDR StateT must be float or bfloat16");
    static const cudaError_t status = cudaFuncSetAttribute(Sm120FusedGdrFwdKernel<__nv_bfloat16, StateT, BlockDv>,
                                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                           static_cast<int>(smem_bytes));
    TM_CUDA_CHECK(status);
}

template<class StateT, int BlockDv>
void LaunchSm120FusedGdrFwdTyped(const core::Tensor& q,
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
    static_cast<void>(q);
    static_cast<void>(k);
    static_cast<void>(v);
    static_cast<void>(g_cumsum);
    static_cast<void>(beta);
    static_cast<void>(resolvent);
    static_cast<void>(state_ptrs);
    static_cast<void>(cp_state_ptrs);
    static_cast<void>(out);
    static_cast<void>(state_layer_offset);

    const bool  context_parallel        = cp_source_indices != nullptr;
    const int   descriptor_sequence_num = context_parallel ? data_sequence_num : problem.sequence_num;
    const auto* q_offsets_ptr           = q_offsets.data<int32_t>();
    const auto* finished_ptr            = finished.data<bool>();
    const auto* data_q_offsets_ptr      = context_parallel ? data_q_offsets->data<int32_t>() : nullptr;
    const auto* cp_source_indices_ptr   = context_parallel ? cp_source_indices->data<int32_t>() : nullptr;

    constexpr int block_dv     = BlockDv;
    const int     dv_tiles     = CeilDiv(kHeadDim, block_dv);
    const dim3    grid         = dim3(problem.sequence_num, problem.hv, dv_tiles);
    const dim3    block        = dim3(kFusedGdrThreads);
    const size_t  smem_bytes   = Sm120FusedGdrSharedBytes<__nv_bfloat16, block_dv>();
    auto*         tma_desc_ptr = reinterpret_cast<CUtensorMap*>(tma_desc_workspace);

    SetFusedGdrFwdSharedMemoryLimit<StateT, block_dv>(smem_bytes);
    Sm120FusedGdrFwdKernel<__nv_bfloat16, StateT, block_dv>
        <<<grid, block, smem_bytes, stream>>>(tma_desc_ptr,
                                              q_offsets_ptr,
                                              finished_ptr,
                                              data_q_offsets_ptr,
                                              cp_source_indices_ptr,
                                              descriptor_sequence_num,
                                              problem.hq,
                                              problem.hv);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<class StateT, int BlockDv>
void LaunchSm120FusedGdrFwdRegistered(const core::Tensor& q,
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
    LaunchSm120FusedGdrFwdTyped<StateT, BlockDv>(q,
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
