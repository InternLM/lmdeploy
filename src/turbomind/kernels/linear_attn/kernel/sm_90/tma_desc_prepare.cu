#include "src/turbomind/kernels/linear_attn/kernel/sm_90/tma_desc_prepare.h"

#include <cstdint>

namespace turbomind::linear_attn::delta_rule::detail {

void PrepareSm90GdrTmaDescriptorsAndCumsum(const core::Tensor&        q,
                                           const core::Tensor&        k,
                                           const core::Tensor&        v,
                                           const core::Tensor&        g,
                                           core::Tensor&              g_cumsum,
                                           const core::Tensor&        beta,
                                           core::Tensor&              resolvent,
                                           const core::Tensor&        q_offsets,
                                           const core::Tensor&        finished,
                                           core::Tensor&              out,
                                           core::Tensor&              workspace,
                                           const Problem&             problem,
                                           const ContextParallelPlan& cp,
                                           Sm90GdrTmaLayout        layout,
                                           DataType                   state_dtype,
                                           cudaStream_t               stream)
{
    const bool context_parallel = cp.enabled;
    constexpr auto kVectorAlignment = alignof(float4);
    if (reinterpret_cast<std::uintptr_t>(g.raw_data()) % kVectorAlignment != 0
        || reinterpret_cast<std::uintptr_t>(g_cumsum.raw_data()) % kVectorAlignment != 0) {
        throw std::invalid_argument("SM90 chunk64 GDR requires 16-byte-aligned gate and gate-cumsum tensors");
    }
    const auto kkt_k_desc          = MakeChunkedKktTmaDesc(k);
    const auto kkt_resolvent_desc = MakeChunkedKktResolventTmaDesc(resolvent);
    const auto fused_q_desc        = MakeFusedGdrQkTmaDesc(q);
    const auto fused_k_desc        = MakeFusedGdrQkTmaDesc(k);
    auto fused_selector_problem = problem;
    if (context_parallel) {
        fused_selector_problem.sequence_num = cp.total_segments;
        fused_selector_problem.total_chunks = cp.total_chunks;
    }
    const int fused_block_dv = FusedChunkGdrBlockDv(fused_selector_problem, context_parallel);
    const auto fused_v_desc        = MakeFusedGdrValueTmaDesc(v, fused_block_dv);
    const auto fused_resolvent_desc = MakeFusedGdrResolventTmaDesc(resolvent);
    const auto fused_out_desc       = MakeFusedGdrOutputTmaDesc(out, fused_block_dv);

    CUtensorMap fused_gdr_h_beta_desc{};
    CUtensorMap fused_gdr_h_g_desc{};
    CUtensorMap fused_gdr_h_v_desc{};
    CUtensorMap fused_gdr_h_resolvent_desc{};
    CUtensorMap context_parallel_segment_state_desc{};
    CUtensorMap context_parallel_segment_m_desc{};
    CUtensorMap correct_initial_states_cp_state_desc{};
    CUtensorMap correct_initial_states_segment_state_desc{};
    CUtensorMap correct_initial_states_segment_m_desc{};
    if (context_parallel) {
        fused_gdr_h_beta_desc = MakeFusedGdrHGateTmaDesc(beta);
        fused_gdr_h_g_desc = MakeFusedGdrHGateTmaDesc(g_cumsum);
        fused_gdr_h_v_desc = MakeFusedGdrValueTmaDesc(v, kFusedGdrHBlockDv);
        fused_gdr_h_resolvent_desc = MakeFusedGdrHResolventTmaDesc(resolvent);

        auto* workspace_base = static_cast<char*>(workspace.raw_data());
        auto* cp_state_ptr = reinterpret_cast<float*>(workspace_base + layout.cp_state_offset);
        auto* segment_state_ptr =
            reinterpret_cast<__nv_bfloat16*>(workspace_base + layout.segment_state_offset);
        auto* segment_m_ptr =
            reinterpret_cast<__nv_bfloat16*>(workspace_base + layout.segment_m_offset);
        const int prefix_block_dv =
            state_dtype == kBfloat16 ? kCorrectInitialStatesBf16BlockDv : kCorrectInitialStatesF32BlockDv;
        context_parallel_segment_state_desc = MakeContextParallelStateTmaDesc(
            segment_state_ptr, cp.total_segments, problem.hv, kFusedGdrHBlockDv);
        context_parallel_segment_m_desc = MakeFusedGdrHSegmentMatrixTmaDesc(
            segment_m_ptr, cp.total_segments, problem.hv, kFusedGdrHMBlockDv);
        correct_initial_states_cp_state_desc =
            MakeContextParallelStateTmaDesc(cp_state_ptr, cp.total_segments, problem.hv, prefix_block_dv);
        correct_initial_states_segment_state_desc = MakeContextParallelStateTmaDesc(
            segment_state_ptr, cp.total_segments, problem.hv, prefix_block_dv);
        correct_initial_states_segment_m_desc =
            MakeCorrectInitialStatesSegmentMatrixTmaDesc(segment_m_ptr, cp.total_segments, problem.hv);
    }

    const auto q_base = MakeStridedTensorBase<__nv_bfloat16>(q);
    const auto k_base = MakeStridedTensorBase<__nv_bfloat16>(k);
    const auto v_base = MakeStridedTensorBase<__nv_bfloat16>(v);
    const auto beta_base = MakeStridedTensorBase<float>(beta);
    const StridedTensorBase<__nv_bfloat16> resolvent_base{
        resolvent.data<__nv_bfloat16>(), resolvent.stride(0), resolvent.stride(1)};
    const auto out_base = MakeStridedTensorBase<__nv_bfloat16>(out);

    using Kernel = Sm90GdrTmaDescPrepare<>;
    const int head_quads =
        (problem.hv + Kernel::kSetupHeadsPerScan - 1) / Kernel::kSetupHeadsPerScan;
    const int cumsum_tasks = problem.total_chunks * head_quads;
    const int descriptor_tasks = Kernel::DescriptorTaskCount(context_parallel, problem.sequence_num);
    const int descriptor_blocks =
        (descriptor_tasks + Kernel::kSetupDescriptorsPerBlock - 1) / Kernel::kSetupDescriptorsPerBlock;
    const int blocks = std::max(cumsum_tasks, descriptor_blocks);
    if (blocks == 0) {
        return;
    }

    Sm90GdrPrepareAndCumsumKernel<>
        <<<blocks, Kernel::kSetupThreads, 0, stream>>>(context_parallel,
                                                       layout,
                                                       kkt_k_desc,
                                                       kkt_resolvent_desc,
                                                       fused_gdr_h_beta_desc,
                                                       fused_gdr_h_g_desc,
                                                       fused_q_desc,
                                                       fused_k_desc,
                                                       fused_v_desc,
                                                       fused_resolvent_desc,
                                                       fused_out_desc,
                                                       fused_gdr_h_v_desc,
                                                       fused_gdr_h_resolvent_desc,
                                                       context_parallel_segment_state_desc,
                                                       context_parallel_segment_m_desc,
                                                       correct_initial_states_cp_state_desc,
                                                       correct_initial_states_segment_state_desc,
                                                       correct_initial_states_segment_m_desc,
                                                       q_base,
                                                       k_base,
                                                       v_base,
                                                       beta_base,
                                                       resolvent_base,
                                                       out_base,
                                                       g.data<float>(),
                                                       g_cumsum.data<float>(),
                                                       q_offsets.data<int32_t>(),
                                                       finished.data<bool>(),
                                                       workspace.raw_data(),
                                                       problem.total_chunks,
                                                       problem.sequence_num,
                                                       problem.token_num,
                                                       problem.hv,
                                                       cp.total_segments,
                                                       cp.segment_chunks,
                                                       cp.segment_tokens,
                                                       problem.gate_stride,
                                                       problem.gate_batch_stride);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind::linear_attn::delta_rule::detail
