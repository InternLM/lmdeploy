#include "src/turbomind/kernels/linear_attn/kernel/sm_90/internal.h"

#include "src/turbomind/kernels/linear_attn/kernel/plan.h"

#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace turbomind::linear_attn::delta_rule::detail {
namespace {

constexpr int kHeadDim = 128;
constexpr int kTmaDescriptorBytes = 128;
constexpr int kKktTmaDescCount = 4;
constexpr int kFusedGdrDataDescCount = 7;
constexpr int kFusedGdrStateDescCount = 1;
constexpr int kFusedGdrHDataDescCount = 5;
constexpr int kFusedGdrHTensorDescCount = 2;
constexpr int kCorrectInitialStatesTensorDescCount = 3;
constexpr int kContextParallelFusedGdrTensorDescCount = 1;

int CeilDiv(int value, int divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

size_t KktTmaDescriptorBytes(const Problem& problem)
{
    return size_t(problem.sequence_num) * kKktTmaDescCount * kTmaDescriptorBytes;
}

size_t FusedGdrTmaDescriptorBytes(const Problem& problem)
{
    const size_t data_descs = size_t(problem.sequence_num) * kFusedGdrDataDescCount;
    const size_t state_descs = size_t(problem.sequence_num) * problem.hv * kFusedGdrStateDescCount;
    return (data_descs + state_descs) * kTmaDescriptorBytes;
}

size_t FusedGdrHTmaDescriptorBytes(const Problem& problem)
{
    return (size_t(problem.sequence_num) * kFusedGdrHDataDescCount
            + kFusedGdrHTensorDescCount)
           * kTmaDescriptorBytes;
}

size_t CorrectInitialStatesTmaDescriptorBytes()
{
    return kCorrectInitialStatesTensorDescCount * kTmaDescriptorBytes;
}

size_t ContextParallelFusedGdrTmaDescriptorBytes(const Problem& problem)
{
    return (size_t(problem.sequence_num) * kFusedGdrDataDescCount
            + kContextParallelFusedGdrTensorDescCount)
           * kTmaDescriptorBytes;
}

int CpSegmentChunks(const Problem& problem)
{
    const double scaled = std::sqrt(double(problem.hv) * double(std::max(1, problem.total_chunks))
                                    / double(problem.sm_count))
                          * 3.0;
    const int power = std::max(0, int(std::round(std::log2(std::max(1.0, scaled)))));
    return std::max(4, 1 << power);
}

size_t BuildCpWorkspaceBytes(const Problem& problem, const ContextParallelPlan& cp)
{
    size_t bytes = 0;
    const size_t segments = size_t(cp.total_segments);
    const size_t heads = size_t(problem.hv);
    const size_t state_elements = segments * heads * kHeadDim * kHeadDim;
    AddWorkspaceBytes(&bytes, state_elements * sizeof(float));
    AddWorkspaceBytes(&bytes, state_elements * sizeof(__nv_bfloat16));
    AddWorkspaceBytes(&bytes, state_elements * sizeof(__nv_bfloat16));
    AddWorkspaceBytes(&bytes, (segments + 1) * sizeof(int32_t));
    AddWorkspaceBytes(&bytes, segments * sizeof(int32_t));
    AddWorkspaceBytes(&bytes, size_t(problem.sequence_num + 1) * sizeof(int32_t));
    AddWorkspaceBytes(&bytes, segments * sizeof(int64_t));
    AddWorkspaceBytes(&bytes, segments * sizeof(bool));
    AddWorkspaceBytes(&bytes, segments * heads * sizeof(bool));
    const size_t descriptor_bytes = KktTmaDescriptorBytes(problem)
                                    + FusedGdrHTmaDescriptorBytes(problem)
                                    + CorrectInitialStatesTmaDescriptorBytes()
                                    + ContextParallelFusedGdrTmaDescriptorBytes(problem) + 127;
    AddWorkspaceBytes(&bytes, descriptor_bytes, kTmaDescriptorBytes);
    return bytes;
}

ContextParallelPlan BuildContextParallelPlan(const Problem& problem,
                                             const std::vector<int32_t>& q_offsets)
{
    auto cp = BuildDisabledContextParallelPlan(problem);
    int max_chunks = 0;
    int total_raw_chunks = 0;
    for (int sequence = 0; sequence < problem.sequence_num; ++sequence) {
        const int tokens = q_offsets[sequence + 1] - q_offsets[sequence];
        const int chunks = tokens <= 0 ? 0 : CeilDiv(tokens, problem.chunk_size);
        max_chunks = std::max(max_chunks, chunks);
        total_raw_chunks += chunks;
    }
    const double effective_batch = double(total_raw_chunks) / double(max_chunks);
    const double effective_heads = effective_batch * double(problem.hv);
    const bool use_cp = effective_heads <= 40.0 || (effective_heads <= 56.0 && max_chunks >= 128);
    if (!use_cp) {
        return cp;
    }

    cp.enabled = true;
    cp.segment_chunks = CpSegmentChunks(problem);
    cp.segment_tokens = cp.segment_chunks * problem.chunk_size;
    cp.total_segments = 0;
    cp.total_chunks = 0;
    for (int sequence = 0; sequence < problem.sequence_num; ++sequence) {
        const int tokens = q_offsets[sequence + 1] - q_offsets[sequence];
        if (tokens > 0) {
            cp.total_segments += CeilDiv(tokens, cp.segment_tokens);
            cp.total_chunks += CeilDiv(tokens, problem.chunk_size);
        }
    }
    cp.cp_q_offsets = TensorPlan{core::Layout{{cp.total_segments + 1}}, kInt32};
    cp.cp_source_indices = TensorPlan{core::Layout{{cp.total_segments}}, kInt32};
    cp.cp_sequence_starts = TensorPlan{core::Layout{{problem.sequence_num + 1}}, kInt32};
    cp.cp_state_ptrs = TensorPlan{core::Layout{{cp.total_segments}}, kInt64};
    cp.cp_finished = TensorPlan{core::Layout{{cp.total_segments}}, kBool};
    cp.cp_fallback = TensorPlan{core::Layout{{cp.total_segments, problem.hv}}, kBool};
    cp.workspace_bytes = BuildCpWorkspaceBytes(problem, cp);
    return cp;
}

std::uintptr_t AlignUpAddress(std::uintptr_t value, std::uintptr_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

core::Tensor WorkspaceTensor(char* base,
                             size_t offset,
                             core::Layout layout,
                             DataType dtype,
                             core::Device device)
{
    return core::Tensor{base + offset, std::move(layout), dtype, device};
}

size_t StateBytes(int batch, int hv, DataType dtype)
{
    return size_t(batch) * size_t(hv) * kHeadDim * kHeadDim * byte_size(dtype, 1);
}

size_t AlignDescriptorOffset(char* base, size_t offset)
{
    const auto address = reinterpret_cast<std::uintptr_t>(base) + offset;
    return size_t(AlignUpAddress(address, kTmaDescriptorBytes)
                  - reinterpret_cast<std::uintptr_t>(base));
}

}  // namespace

bool PlanSm90Operation(const GdrKernelSpec& spec,
                       const Operation& operation,
                       const PlanningContext& context,
                       Plan* plan)
{
    plan->problem = BuildProblem(context, spec);
    if (IsChunkedGdr(plan->problem)
        && (plan->problem.gate_stride % 4 != 0 || plan->problem.gate_batch_stride % 4 != 0)) {
        return false;
    }
    plan->cp = operation.mode == GdrMode::kChunked
                       && operation.cp_mode == ContextParallelMode::kAuto
                   ? BuildContextParallelPlan(plan->problem, context.q_offsets)
                   : BuildDisabledContextParallelPlan(plan->problem);
    const size_t descriptor_bytes = KktTmaDescriptorBytes(plan->problem)
                                    + FusedGdrTmaDescriptorBytes(plan->problem) + 127;
    BuildOptimizedTensorPlans(plan, descriptor_bytes);
    plan->state_tma_desc_bytes_per_layer_group =
        IsRecurrentGdr(plan->problem)
            ? size_t(plan->problem.sequence_num) * plan->problem.num_head_groups * sizeof(CUtensorMap)
            : 0;
    return true;
}

Sm90DirectChunkWorkspace PartitionSm90DirectChunkWorkspace(const Arguments& args, const Plan& plan)
{
    auto* base = static_cast<char*>(args.workspace->raw_data());
    size_t offset = 0;
    Sm90DirectChunkWorkspace out{};
    out.g_cumsum = WorkspaceTensor(
        base, offset, plan.g_cumsum.layout, plan.g_cumsum.dtype, args.workspace->device());
    AddWorkspaceBytes(&offset, plan.g_cumsum.storage_size * sizeof(float));
    const size_t resolvent_offset = offset == 0 ? 0 : AlignUp(offset, 16);
    out.resolvent = WorkspaceTensor(base,
                                    resolvent_offset,
                                    plan.resolvent.layout,
                                    plan.resolvent.dtype,
                                    args.workspace->device());
    AddWorkspaceBytes(&offset,
                      plan.resolvent.storage_size * byte_size(plan.resolvent.dtype, 1));
    offset = AlignDescriptorOffset(base, offset);
    out.layout.kkt_desc_offset = offset;
    out.kkt_tma_desc = base + offset;
    offset += KktTmaDescriptorBytes(plan.problem);
    out.layout.direct_fused_desc_offset = offset;
    out.fused_tma_desc = base + offset;
    return out;
}

Sm90ContextParallelWorkspace PartitionSm90ContextParallelWorkspace(const Arguments& args, const Plan& plan)
{
    const auto& problem = plan.problem;
    const auto& cp = plan.cp;
    auto* base = static_cast<char*>(args.workspace->raw_data());
    size_t offset = 0;
    Sm90ContextParallelWorkspace out{};
    out.g_cumsum = WorkspaceTensor(
        base, offset, plan.g_cumsum.layout, plan.g_cumsum.dtype, args.workspace->device());
    AddWorkspaceBytes(&offset, plan.g_cumsum.storage_size * sizeof(float));
    const size_t resolvent_offset = offset == 0 ? 0 : AlignUp(offset, 16);
    out.resolvent = WorkspaceTensor(base,
                                    resolvent_offset,
                                    plan.resolvent.layout,
                                    plan.resolvent.dtype,
                                    args.workspace->device());
    AddWorkspaceBytes(&offset,
                      plan.resolvent.storage_size * byte_size(plan.resolvent.dtype, 1));

    offset = offset == 0 ? 0 : AlignUp(offset, 16);
    out.layout.cp_state_offset = offset;
    out.cp_state = WorkspaceTensor(base,
                                   offset,
                                   core::Layout{{cp.total_segments, problem.hv, kHeadDim, kHeadDim}},
                                   kFloat32,
                                   args.workspace->device());
    AddWorkspaceBytes(&offset, StateBytes(cp.total_segments, problem.hv, kFloat32));
    out.layout.segment_state_offset = offset;
    out.segment_state = WorkspaceTensor(base,
                                        offset,
                                        core::Layout{{cp.total_segments, problem.hv, kHeadDim, kHeadDim}},
                                        kBfloat16,
                                        args.workspace->device());
    AddWorkspaceBytes(&offset, StateBytes(cp.total_segments, problem.hv, kBfloat16));
    out.layout.segment_m_offset = offset;
    out.segment_m = WorkspaceTensor(base,
                                    offset,
                                    core::Layout{{cp.total_segments, problem.hv, kHeadDim, kHeadDim}},
                                    kBfloat16,
                                    args.workspace->device());
    AddWorkspaceBytes(&offset, StateBytes(cp.total_segments, problem.hv, kBfloat16));

    const auto partition = [&](size_t& layout_offset,
                               core::Tensor& tensor,
                               const TensorPlan& tensor_plan,
                               DataType dtype,
                               size_t count) {
        layout_offset = offset == 0 ? 0 : AlignUp(offset, 16);
        offset = layout_offset;
        tensor = WorkspaceTensor(
            base, offset, tensor_plan.layout, tensor_plan.dtype, args.workspace->device());
        offset += byte_size(dtype, count);
    };
    partition(out.layout.cp_q_offsets_offset,
              out.cp_q_offsets,
              cp.cp_q_offsets,
              kInt32,
              cp.total_segments + 1);
    partition(out.layout.cp_source_indices_offset,
              out.cp_source_indices,
              cp.cp_source_indices,
              kInt32,
              cp.total_segments);
    partition(out.layout.cp_sequence_starts_offset,
              out.cp_sequence_starts,
              cp.cp_sequence_starts,
              kInt32,
              problem.sequence_num + 1);
    partition(out.layout.cp_state_ptrs_offset,
              out.cp_state_ptrs,
              cp.cp_state_ptrs,
              kInt64,
              cp.total_segments);
    partition(out.layout.cp_finished_offset,
              out.cp_finished,
              cp.cp_finished,
              kBool,
              cp.total_segments);
    partition(out.layout.cp_fallback_offset,
              out.cp_fallback,
              cp.cp_fallback,
              kBool,
              cp.total_segments * problem.hv);

    offset = AlignDescriptorOffset(base, offset);
    out.layout.kkt_desc_offset = offset;
    out.kkt_tma_desc = base + offset;
    offset += KktTmaDescriptorBytes(problem);
    out.layout.fused_gdr_h_desc_offset = offset;
    out.fused_gdr_h_tma_desc = base + offset;
    offset += FusedGdrHTmaDescriptorBytes(problem);
    out.layout.correct_initial_states_desc_offset = offset;
    out.correct_initial_states_tma_desc = base + offset;
    offset += CorrectInitialStatesTmaDescriptorBytes();
    out.layout.context_parallel_fused_gdr_desc_offset = offset;
    out.context_parallel_fused_gdr_tma_desc = base + offset;
    return out;
}

}  // namespace turbomind::linear_attn::delta_rule::detail
