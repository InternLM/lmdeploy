#include "src/turbomind/kernels/linear_attn/kernel/plan.h"

#include <algorithm>

namespace turbomind::linear_attn::delta_rule::detail {
namespace {

int CeilDiv(int value, int divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

}  // namespace

size_t AlignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

void AddWorkspaceBytes(size_t* total, size_t bytes, size_t alignment)
{
    if (bytes == 0) {
        return;
    }
    *total = *total == 0 ? bytes : AlignUp(*total, alignment) + bytes;
}

bool MatchesGdrSpec(const GdrKernelSpec& spec, const Operation& operation, const PlanningContext& context)
{
    const int requested_chunk = operation.mode == GdrMode::kRecurrent     ? kRecurrentGdrChunkSize :
                                operation.chunk_size == kAutoGdrChunkSize ? spec.chunk_size :
                                                                            operation.chunk_size;
    return operation.mode == spec.mode && context.input_dtype == spec.input_dtype
           && context.state_dtype == spec.state_dtype && context.head_dim == spec.head_dim
           && requested_chunk == spec.chunk_size
           && (operation.mode != GdrMode::kRecurrent || operation.chunk_size == kAutoGdrChunkSize
               || operation.chunk_size == kRecurrentGdrChunkSize);
}

Problem BuildProblem(const PlanningContext& context, const GdrKernelSpec& spec)
{
    Problem problem{};
    problem.arch              = context.arch;
    problem.sm_count          = context.sm_count;
    problem.input_dtype       = context.input_dtype;
    problem.state_dtype       = context.state_dtype;
    problem.batch             = context.physical_batch;
    problem.token_num         = context.token_slots;
    problem.hq                = context.hq;
    problem.hv                = context.hv;
    problem.gate_stride       = context.gate_stride;
    problem.gate_batch_stride = context.gate_batch_stride;
    problem.beta_stride       = context.beta_stride;
    problem.beta_batch_stride = context.beta_batch_stride;
    problem.head_dim          = context.head_dim;
    problem.chunk_size        = spec.chunk_size;
    problem.num_head_groups   = context.num_head_groups;
    problem.heads_per_block   = context.heads_per_block;
    if (spec.mode == GdrMode::kRecurrent) {
        problem.sequence_num        = context.physical_batch;
        problem.total_chunks        = context.physical_batch;
        problem.max_sequence_chunks = context.physical_batch > 0 ? 1 : 0;
        return problem;
    }

    problem.sequence_num = static_cast<int>(context.q_offsets.size() - 1);
    int total_chunks         = 0;
    int max_sequence_chunks = 0;
    for (int sequence = 0; sequence < problem.sequence_num; ++sequence) {
        const int tokens = context.q_offsets[sequence + 1] - context.q_offsets[sequence];
        const int chunks = CeilDiv(tokens, spec.chunk_size);
        total_chunks += chunks;
        max_sequence_chunks = std::max(max_sequence_chunks, chunks);
    }
    problem.total_chunks        = total_chunks;
    problem.max_sequence_chunks = max_sequence_chunks;
    return problem;
}

ContextParallelPlan BuildDisabledContextParallelPlan(const Problem& problem)
{
    TensorPlan          zero{core::Layout{{0}}, kInt32};
    ContextParallelPlan cp{};
    cp.total_segments     = problem.sequence_num;
    cp.segment_tokens     = problem.token_num;
    cp.segment_chunks     = IsRecurrentGdr(problem) ? CeilDiv(problem.token_num, problem.chunk_size) : 0;
    cp.total_chunks       = problem.total_chunks;
    cp.cp_q_offsets       = zero;
    cp.cp_source_indices  = zero;
    cp.cp_sequence_starts = zero;
    cp.cp_state_ptrs      = TensorPlan{core::Layout{{0}}, kInt64};
    cp.cp_finished        = TensorPlan{core::Layout{{0}}, kBool};
    cp.cp_fallback        = TensorPlan{core::Layout{{0}}, kBool};
    return cp;
}

void BuildOptimizedTensorPlans(Plan* plan, size_t direct_descriptor_bytes)
{
    auto&               problem        = plan->problem;
    const core::ssize_t value_row      = core::ssize_t(problem.hv) * 128;
    const core::ssize_t value_batch    = core::ssize_t(problem.token_num) * value_row;
    const size_t        value_elements = size_t(problem.batch) * size_t(value_batch);
    plan->out =
        TensorPlan{core::Layout{{problem.batch, problem.token_num, problem.hv, 128}, {value_batch, value_row, 128, 1}},
                   problem.input_dtype,
                   value_elements};
    const bool          chunked     = IsChunkedGdr(problem);
    const core::ssize_t gate_stride = chunked ? core::ssize_t(AlignUp(size_t(problem.hv), 4)) : problem.gate_stride;
    const core::ssize_t gate_batch_stride =
        chunked ? core::ssize_t(problem.token_num) * gate_stride : problem.gate_batch_stride;
    plan->g_cumsum = TensorPlan{
        core::Layout{{problem.batch, problem.token_num, problem.hv}, {gate_batch_stride, gate_stride, 1}}, kFloat32};
    plan->g_cumsum.storage_size = size_t(problem.batch - 1) * gate_batch_stride
                                  + size_t(problem.token_num - 1) * gate_stride + AlignUp(size_t(problem.hv), 4);
    const core::ssize_t resolvent_head     = problem.chunk_size;
    const core::ssize_t resolvent_token    = core::ssize_t(problem.hv) * resolvent_head;
    const core::ssize_t resolvent_batch    = core::ssize_t(problem.token_num) * resolvent_token;
    const size_t        resolvent_elements = size_t(problem.batch) * size_t(resolvent_batch);
    plan->resolvent       = TensorPlan{core::Layout{{problem.batch, problem.token_num, problem.hv, problem.chunk_size},
                                              {resolvent_batch, resolvent_token, resolvent_head, 1}},
                                 problem.input_dtype,
                                 resolvent_elements};
    plan->workspace_bytes = 0;
    if (chunked) {
        AddWorkspaceBytes(&plan->workspace_bytes, plan->g_cumsum.storage_size * sizeof(float));
        AddWorkspaceBytes(&plan->workspace_bytes, resolvent_elements * byte_size(problem.input_dtype, 1));
        AddWorkspaceBytes(&plan->workspace_bytes,
                          plan->cp.enabled ? plan->cp.workspace_bytes : direct_descriptor_bytes,
                          plan->cp.enabled ? 16 : 128);
    }
    plan->workspace = TensorPlan{core::Layout{{core::ssize_t(plan->workspace_bytes)}}, kUint8, plan->workspace_bytes};
}

}  // namespace turbomind::linear_attn::delta_rule::detail
