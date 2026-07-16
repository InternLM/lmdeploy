#include "src/turbomind/kernels/linear_attn/kernel/sm_120/tma_desc_prepare.h"

namespace turbomind::linear_attn::delta_rule::detail {

void PrepareSm120GdrTmaDescriptors(const core::Tensor& q,
                                     const core::Tensor& k,
                                     const core::Tensor& v,
                                     const core::Tensor& g_cumsum,
                                     const core::Tensor& beta,
                                     const core::Tensor& resolvent,
                                     const core::Tensor& state_ptrs,
                                     const core::Tensor& q_offsets,
                                     const core::Tensor& finished,
                                     core::Tensor* out,
                                     core::Tensor& workspace,
                                     const Problem& problem,
                                     const ContextParallelPlan& cp,
                                     Sm120GdrTmaMode mode,
                                     Sm120GdrTmaLayout layout,
                                     int64_t state_layer_offset,
                                     DataType state_dtype,
                                     cudaStream_t stream)
{
    constexpr int ChunkSize = 32;
    const bool needs_fused_desc = mode == Sm120GdrTmaMode::kFusedOnly
                                  || mode == Sm120GdrTmaMode::kAllDirectFused
                                  || mode == Sm120GdrTmaMode::kAllContextParallel;
    const auto* state_ptrs_ptr = needs_fused_desc
                                     ? reinterpret_cast<const int64_t*>(state_ptrs.raw_data())
                                     : nullptr;

    const auto kkt_k_desc = MakeChunkedKktTmaDesc<ChunkSize>(k);
    const auto kkt_beta_desc = MakeChunkedKktGateTmaDesc<ChunkSize>(beta);
    const auto kkt_resolvent_desc = MakeChunkedKktResolventTmaDesc<ChunkSize>(resolvent);

    CUtensorMap fused_q_desc{};
    CUtensorMap fused_k_desc{};
    CUtensorMap fused_v_desc{};
    CUtensorMap fused_g_desc{};
    CUtensorMap fused_beta_desc{};
    CUtensorMap fused_resolvent_desc{};
    CUtensorMap fused_state_desc{};
    CUtensorMap fused_out_desc{};
    CUtensorMap fused_gdr_h_v_desc{};
    CUtensorMap context_parallel_fused_gdr_v_desc{};
    CUtensorMap context_parallel_fused_gdr_out_desc{};
    if (needs_fused_desc) {
        fused_q_desc = MakeFusedGdrQkTmaDesc<ChunkSize>(q);
        fused_k_desc = MakeFusedGdrQkTmaDesc<ChunkSize>(k);
        fused_v_desc = MakeFusedGdrValueTmaDesc<ChunkSize>(v, kFusedGdrBlockDv);
        fused_gdr_h_v_desc =
            MakeFusedGdrValueTmaDesc<ChunkSize>(v, kContextParallelGdrBlockDv);
        context_parallel_fused_gdr_v_desc =
            MakeFusedGdrValueTmaDesc<ChunkSize>(v, kContextParallelGdrBlockDv);
        fused_g_desc = MakeFusedGdrGateTmaDesc<ChunkSize>(g_cumsum);
        fused_beta_desc = MakeFusedGdrGateTmaDesc<ChunkSize>(beta);
        fused_resolvent_desc = MakeFusedGdrResolventTmaDesc<ChunkSize>(resolvent);
        fused_state_desc = state_dtype == kBfloat16
                               ? MakeFusedGdrStateHeadTmaDesc(
                                   reinterpret_cast<__nv_bfloat16*>(workspace.raw_data()), kFusedGdrBlockDv)
                               : MakeFusedGdrStateHeadTmaDesc(
                                   reinterpret_cast<float*>(workspace.raw_data()), kFusedGdrBlockDv);
        fused_out_desc = MakeFusedGdrOutputTmaDesc<ChunkSize>(*out, kFusedGdrBlockDv);
        context_parallel_fused_gdr_out_desc = MakeFusedGdrOutputTmaDesc<ChunkSize>(*out, kContextParallelGdrBlockDv);
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
        auto* cp_state_ptr = reinterpret_cast<float*>(workspace_base + layout.cp_state_offset);
        auto* context_parallel_segment_state_ptr = reinterpret_cast<float*>(workspace_base + layout.segment_state_offset);
        auto* context_parallel_segment_m_ptr = reinterpret_cast<float*>(workspace_base + layout.segment_m_offset);
        const int prefix_block_dv = state_dtype == kBfloat16 ? kCorrectInitialStatesBf16BlockDv : kCorrectInitialStatesF32BlockDv;
        const int prefix_external_block_dv =
            state_dtype == kBfloat16 ? kCorrectInitialStatesBf16ExternalTmaBlockDv : prefix_block_dv;
        context_parallel_segment_state_desc =
            MakeContextParallelStateTmaDesc(context_parallel_segment_state_ptr, cp.total_segments, problem.hv, kContextParallelGdrBlockDv);
        context_parallel_segment_m_desc =
            MakeFusedGdrHSegmentMatrixTmaDesc(
                context_parallel_segment_m_ptr, cp.total_segments, problem.hv, kContextParallelGdrBlockDv);
        correct_initial_states_cp_state_desc =
            MakeContextParallelStateTmaDesc(cp_state_ptr, cp.total_segments, problem.hv, prefix_block_dv);
        correct_initial_states_segment_state_desc =
            MakeContextParallelStateTmaDesc(context_parallel_segment_state_ptr, cp.total_segments, problem.hv, prefix_block_dv);
        correct_initial_states_segment_m_desc =
            MakeCorrectInitialStatesSegmentMatrixTmaDesc(context_parallel_segment_m_ptr, cp.total_segments, problem.hv);
        correct_initial_states_external_state_desc =
            state_dtype == kBfloat16
                ? MakeFusedGdrStateHeadTmaDesc(
                    reinterpret_cast<__nv_bfloat16*>(workspace.raw_data()), prefix_external_block_dv)
                : MakeFusedGdrStateHeadTmaDesc(reinterpret_cast<float*>(workspace.raw_data()), prefix_external_block_dv);
        context_parallel_fused_gdr_state_desc =
            MakeContextParallelStateTmaDesc(cp_state_ptr, cp.total_segments, problem.hv, kContextParallelGdrBlockDv);
    }
    if (mode == Sm120GdrTmaMode::kAllContextParallel) {
        fused_v_desc   = context_parallel_fused_gdr_v_desc;
        fused_out_desc = context_parallel_fused_gdr_out_desc;
    }

    constexpr int kContextParallelTensorDescTasks = 1;
    const bool needs_direct_desc = mode == Sm120GdrTmaMode::kAllDirectFused
                                   || mode == Sm120GdrTmaMode::kFusedOnly;
    const int direct_data_tasks = needs_direct_desc ? problem.sequence_num : 0;
    const int direct_state_tasks = needs_direct_desc ? problem.sequence_num * problem.hv : 0;
    const int direct_desc_tasks = direct_data_tasks + direct_state_tasks;

    const int fused_gdr_h_data_tasks = cp.enabled ? problem.sequence_num : 0;
    const int fused_gdr_h_tensor_tasks = cp.enabled ? kContextParallelTensorDescTasks : 0;
    const int correct_initial_states_tensor_tasks = cp.enabled ? kContextParallelTensorDescTasks : 0;
    const int correct_initial_states_external_tasks = cp.enabled ? problem.sequence_num * problem.hv : 0;
    const int context_parallel_fused_gdr_data_tasks = cp.enabled ? problem.sequence_num : 0;
    const int context_parallel_fused_gdr_tensor_tasks = cp.enabled ? kContextParallelTensorDescTasks : 0;
    const int context_parallel_desc_tasks = mode == Sm120GdrTmaMode::kAllContextParallel
                                     ? fused_gdr_h_data_tasks + fused_gdr_h_tensor_tasks
                                           + correct_initial_states_tensor_tasks + correct_initial_states_external_tasks
                                           + context_parallel_fused_gdr_data_tasks + context_parallel_fused_gdr_tensor_tasks
                                     : 0;
    const bool needs_kkt_desc = mode == Sm120GdrTmaMode::kSolveKkt
                                || mode == Sm120GdrTmaMode::kAllDirectFused
                                || mode == Sm120GdrTmaMode::kAllContextParallel;
    const int kkt_tasks = needs_kkt_desc ? problem.sequence_num : 0;
    const int blocks = kkt_tasks + direct_desc_tasks + context_parallel_desc_tasks;
    const auto q_base = MakeStridedTensorBase<__nv_bfloat16>(q);
    const auto k_base = MakeStridedTensorBase<__nv_bfloat16>(k);
    const auto v_base = MakeStridedTensorBase<__nv_bfloat16>(v);
    const auto g_base = MakeStridedTensorBase<float>(g_cumsum);
    const auto beta_base = MakeStridedTensorBase<float>(beta);
    const StridedTensorBase<__nv_bfloat16> resolvent_base{
        const_cast<__nv_bfloat16*>(resolvent.data<__nv_bfloat16>()),
        resolvent.stride(0),
        resolvent.stride(1)};
    const StridedTensorBase<__nv_bfloat16> out_base =
        out == nullptr ? StridedTensorBase<__nv_bfloat16>{}
                       : MakeStridedTensorBase<__nv_bfloat16>(*out);

    if (state_dtype == kBfloat16) {
        Sm120GdrTmaDescPrepareKernel<__nv_bfloat16, ChunkSize><<<blocks, 32, 0, stream>>>(mode,
                                                                                       layout,
                                                                                       kkt_k_desc,
                                                                                       kkt_beta_desc,
                                                                                       kkt_resolvent_desc,
                                                                                       fused_q_desc,
                                                                                       fused_k_desc,
                                                                                       fused_v_desc,
                                                                                       fused_g_desc,
                                                                                       fused_beta_desc,
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
                                                                                       beta_base,
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
    }
    else {
        Sm120GdrTmaDescPrepareKernel<float, ChunkSize><<<blocks, 32, 0, stream>>>(mode,
                                                                               layout,
                                                                               kkt_k_desc,
                                                                               kkt_beta_desc,
                                                                               kkt_resolvent_desc,
                                                                               fused_q_desc,
                                                                               fused_k_desc,
                                                                               fused_v_desc,
                                                                               fused_g_desc,
                                                                               fused_beta_desc,
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
                                                                               beta_base,
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
    }
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind::linear_attn::delta_rule::detail
