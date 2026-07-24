#include "src/turbomind/kernels/linear_attn/kernel/sm_120/tma_desc_prepare.h"

#include <type_traits>

namespace turbomind::linear_attn::delta_rule::detail {

template<class StateT, int BlockDv>
void PrepareSm120GdrTmaDescriptors(const core::Tensor&        q,
                                   const core::Tensor&        k,
                                   const core::Tensor&        v,
                                   const core::Tensor&        g_cumsum,
                                   const core::Tensor&        resolvent,
                                   const core::Tensor&        state_ptrs,
                                   const core::Tensor&        q_offsets,
                                   const core::Tensor&        finished,
                                   core::Tensor*              out,
                                   core::Tensor&              workspace,
                                   const Problem&             problem,
                                   const ContextParallelPlan& cp,
                                   Sm120GdrTmaMode            mode,
                                   Sm120GdrTmaLayout          layout,
                                   int64_t                    state_layer_offset,
                                   cudaStream_t               stream)
{
    constexpr int ChunkSize = 32;
    using Kernel            = Sm120GdrTmaDescPrepare<StateT, ChunkSize>;
    static_assert(std::is_same_v<StateT, float> || std::is_same_v<StateT, __nv_bfloat16>,
                  "SM120 descriptor StateT must be float or bfloat16");
    static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
    const bool needs_fused_desc = mode == Sm120GdrTmaMode::kFusedOnly || mode == Sm120GdrTmaMode::kAllDirectFused
                                  || mode == Sm120GdrTmaMode::kAllContextParallel;
    const auto* state_ptrs_ptr = needs_fused_desc ? reinterpret_cast<const int64_t*>(state_ptrs.raw_data()) : nullptr;

    using ChunkedKktTma           = typename Kernel::ChunkedKktTma;
    const auto kkt_k_desc         = ChunkedKktTma::MakeKDesc(k);
    const auto kkt_resolvent_desc = ChunkedKktTma::MakeResolventDesc(resolvent);

    CUtensorMap fused_q_desc{};
    CUtensorMap fused_q_hi_desc{};
    CUtensorMap fused_k_desc{};
    CUtensorMap fused_k_hi_desc{};
    CUtensorMap fused_gdr_h_k_desc{};
    CUtensorMap fused_v_desc{};
    CUtensorMap fused_g_desc{};
    CUtensorMap fused_resolvent_desc{};
    CUtensorMap fused_state_desc{};
    CUtensorMap fused_out_desc{};
    CUtensorMap fused_gdr_h_v_desc{};
    CUtensorMap context_parallel_fused_gdr_v_desc{};
    CUtensorMap context_parallel_fused_gdr_out_desc{};
    if (needs_fused_desc) {
        fused_q_desc                             = Kernel::MakeFusedGdrQkTmaDesc(q);
        fused_q_hi_desc                          = Kernel::MakeFusedGdrQkTmaDesc(q);
        fused_k_desc                             = Kernel::MakeFusedGdrQkTmaDesc(k);
        fused_k_hi_desc                          = Kernel::MakeFusedGdrQkTmaDesc(k);
        const uint32_t fused_gdr_h_k_box_dims[5] = {64u, 2u, 1u, static_cast<uint32_t>(ChunkSize), 1u};
        fused_gdr_h_k_desc =
            Kernel::template MakeQkTmaDesc<__nv_bfloat16>(k, fused_gdr_h_k_box_dims, CU_TENSOR_MAP_SWIZZLE_128B);
        fused_v_desc                      = Kernel::MakeFusedGdrValueTmaDesc(v, BlockDv);
        fused_gdr_h_v_desc                = Kernel::MakeFusedGdrValueTmaDesc(v, kFusedGdrBlockDv);
        context_parallel_fused_gdr_v_desc = Kernel::MakeFusedGdrValueTmaDesc(v, BlockDv);
        fused_g_desc                      = Kernel::MakeFusedGdrGateTmaDesc(g_cumsum);
        fused_resolvent_desc              = Kernel::MakeFusedGdrResolventTmaDesc(resolvent);
        fused_state_desc =
            Kernel::MakeFusedGdrStateHeadTmaDesc(reinterpret_cast<StateT*>(workspace.raw_data()), BlockDv);
        fused_out_desc                      = Kernel::MakeFusedGdrOutputTmaDesc(*out, BlockDv);
        context_parallel_fused_gdr_out_desc = Kernel::MakeFusedGdrOutputTmaDesc(*out, BlockDv);
    }

    CUtensorMap context_parallel_segment_state_desc{};
    CUtensorMap context_parallel_segment_m_desc{};
    CUtensorMap correct_initial_states_cp_state_desc{};
    CUtensorMap correct_initial_states_segment_state_desc{};
    CUtensorMap correct_initial_states_segment_m_desc{};
    CUtensorMap correct_initial_states_external_state_desc{};
    CUtensorMap context_parallel_fused_gdr_state_desc{};
    if (mode == Sm120GdrTmaMode::kAllContextParallel) {
        auto* workspace_base = static_cast<char*>(workspace.raw_data());
        auto* cp_state_ptr   = reinterpret_cast<float*>(workspace_base + layout.cp_state_offset);
        auto* context_parallel_segment_state_ptr =
            reinterpret_cast<float*>(workspace_base + layout.segment_state_offset);
        auto* context_parallel_segment_m_ptr = reinterpret_cast<float*>(workspace_base + layout.segment_m_offset);
        constexpr bool kBf16State            = std::is_same_v<StateT, __nv_bfloat16>;
        constexpr int  prefix_block_dv =
            kBf16State ? Kernel::kCorrectInitialStatesBf16BlockDv : Kernel::kCorrectInitialStatesF32BlockDv;
        constexpr int prefix_external_block_dv =
            kBf16State ? Kernel::kCorrectInitialStatesBf16ExternalTmaBlockDv : prefix_block_dv;
        context_parallel_segment_state_desc = Kernel::MakeContextParallelStateTmaDesc(
            context_parallel_segment_state_ptr, cp.total_segments, problem.hv, kFusedGdrBlockDv);
        context_parallel_segment_m_desc = Kernel::MakeFusedGdrHSegmentMatrixTmaDesc(
            context_parallel_segment_m_ptr, cp.total_segments, problem.hv, kFusedGdrBlockDv);
        correct_initial_states_cp_state_desc =
            Kernel::MakeContextParallelStateTmaDesc(cp_state_ptr, cp.total_segments, problem.hv, prefix_block_dv);
        correct_initial_states_segment_state_desc = Kernel::MakeContextParallelStateTmaDesc(
            context_parallel_segment_state_ptr, cp.total_segments, problem.hv, prefix_block_dv);
        correct_initial_states_segment_m_desc = Kernel::MakeCorrectInitialStatesSegmentMatrixTmaDesc(
            context_parallel_segment_m_ptr, cp.total_segments, problem.hv);
        correct_initial_states_external_state_desc = Kernel::MakeFusedGdrStateHeadTmaDesc(
            reinterpret_cast<StateT*>(workspace.raw_data()), prefix_external_block_dv);
        context_parallel_fused_gdr_state_desc =
            Kernel::MakeContextParallelStateTmaDesc(cp_state_ptr, cp.total_segments, problem.hv, BlockDv);
    }
    if (mode == Sm120GdrTmaMode::kAllContextParallel) {
        fused_v_desc   = context_parallel_fused_gdr_v_desc;
        fused_out_desc = context_parallel_fused_gdr_out_desc;
    }

    constexpr int kContextParallelTensorDescTasks = 1;
    const bool    needs_direct_desc  = mode == Sm120GdrTmaMode::kAllDirectFused || mode == Sm120GdrTmaMode::kFusedOnly;
    const int     direct_data_tasks  = needs_direct_desc ? problem.sequence_num : 0;
    const int     direct_state_tasks = needs_direct_desc ? problem.sequence_num * problem.hv : 0;
    const int     direct_desc_tasks  = direct_data_tasks + direct_state_tasks;

    const int fused_gdr_h_data_tasks                  = cp.enabled ? problem.sequence_num : 0;
    const int fused_gdr_h_tensor_tasks                = cp.enabled ? kContextParallelTensorDescTasks : 0;
    const int correct_initial_states_tensor_tasks     = cp.enabled ? kContextParallelTensorDescTasks : 0;
    const int correct_initial_states_external_tasks   = cp.enabled ? problem.sequence_num * problem.hv : 0;
    const int context_parallel_fused_gdr_data_tasks   = cp.enabled ? problem.sequence_num : 0;
    const int context_parallel_fused_gdr_tensor_tasks = cp.enabled ? kContextParallelTensorDescTasks : 0;
    const int context_parallel_desc_tasks =
        mode == Sm120GdrTmaMode::kAllContextParallel ?
            fused_gdr_h_data_tasks + fused_gdr_h_tensor_tasks + correct_initial_states_tensor_tasks
                + correct_initial_states_external_tasks + context_parallel_fused_gdr_data_tasks
                + context_parallel_fused_gdr_tensor_tasks :
            0;
    const bool needs_kkt_desc = mode == Sm120GdrTmaMode::kSolveKkt || mode == Sm120GdrTmaMode::kAllDirectFused
                                || mode == Sm120GdrTmaMode::kAllContextParallel;
    const int  kkt_tasks = needs_kkt_desc ? problem.sequence_num : 0;
    const int  blocks    = kkt_tasks + direct_desc_tasks + context_parallel_desc_tasks;
    const auto q_base    = Kernel::template MakeStridedTensorBase<__nv_bfloat16>(q);
    const auto k_base    = Kernel::template MakeStridedTensorBase<__nv_bfloat16>(k);
    const auto v_base    = Kernel::template MakeStridedTensorBase<__nv_bfloat16>(v);
    const auto g_base    = Kernel::template MakeStridedTensorBase<float>(g_cumsum);
    const typename Kernel::template StridedTensorBase<__nv_bfloat16> resolvent_base{
        const_cast<__nv_bfloat16*>(resolvent.data<__nv_bfloat16>()), resolvent.stride(0), resolvent.stride(1)};
    const typename Kernel::template StridedTensorBase<__nv_bfloat16> out_base =
        out == nullptr ? typename Kernel::template StridedTensorBase<__nv_bfloat16>{} :
                         Kernel::template MakeStridedTensorBase<__nv_bfloat16>(*out);

    Sm120GdrTmaDescPrepareKernel<StateT, ChunkSize>
        <<<blocks, Kernel::kThreads, 0, stream>>>(mode,
                                                  layout,
                                                  kkt_k_desc,
                                                  kkt_resolvent_desc,
                                                  fused_q_desc,
                                                  fused_q_hi_desc,
                                                  fused_k_desc,
                                                  fused_k_hi_desc,
                                                  fused_gdr_h_k_desc,
                                                  fused_v_desc,
                                                  fused_g_desc,
                                                  fused_resolvent_desc,
                                                  fused_state_desc,
                                                  fused_out_desc,
                                                  fused_gdr_h_v_desc,
                                                  context_parallel_segment_state_desc,
                                                  context_parallel_segment_m_desc,
                                                  correct_initial_states_cp_state_desc,
                                                  correct_initial_states_segment_state_desc,
                                                  correct_initial_states_segment_m_desc,
                                                  correct_initial_states_external_state_desc,
                                                  context_parallel_fused_gdr_state_desc,
                                                  q_base,
                                                  k_base,
                                                  v_base,
                                                  g_base,
                                                  resolvent_base,
                                                  out_base,
                                                  state_ptrs_ptr,
                                                  q_offsets.data<int32_t>(),
                                                  finished.data<bool>(),
                                                  workspace.raw_data(),
                                                  problem.sequence_num,
                                                  problem.hq,
                                                  problem.hv,
                                                  problem.num_head_groups,
                                                  problem.heads_per_block,
                                                  problem.token_num,
                                                  cp.total_segments,
                                                  cp.segment_tokens,
                                                  problem.gate_stride,
                                                  problem.gate_batch_stride,
                                                  state_layer_offset);
    TM_CUDA_CHECK(cudaGetLastError());
}

#define TM_INSTANTIATE_SM120_TMA_PREPARE(STATE_T, BLOCK_DV)                                                            \
    template void PrepareSm120GdrTmaDescriptors<STATE_T, BLOCK_DV>(const core::Tensor&,                                \
                                                                   const core::Tensor&,                                \
                                                                   const core::Tensor&,                                \
                                                                   const core::Tensor&,                                \
                                                                   const core::Tensor&,                                \
                                                                   const core::Tensor&,                                \
                                                                   const core::Tensor&,                                \
                                                                   const core::Tensor&,                                \
                                                                   core::Tensor*,                                      \
                                                                   core::Tensor&,                                      \
                                                                   const Problem&,                                     \
                                                                   const ContextParallelPlan&,                         \
                                                                   Sm120GdrTmaMode,                                    \
                                                                   Sm120GdrTmaLayout,                                  \
                                                                   int64_t,                                            \
                                                                   cudaStream_t)

TM_INSTANTIATE_SM120_TMA_PREPARE(float, kContextParallelGdrBlockDv);
TM_INSTANTIATE_SM120_TMA_PREPARE(float, kFusedGdrBlockDv);
TM_INSTANTIATE_SM120_TMA_PREPARE(__nv_bfloat16, kContextParallelGdrBlockDv);
TM_INSTANTIATE_SM120_TMA_PREPARE(__nv_bfloat16, kFusedGdrBlockDv);

#undef TM_INSTANTIATE_SM120_TMA_PREPARE

}  // namespace turbomind::linear_attn::delta_rule::detail
