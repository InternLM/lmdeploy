#include "src/turbomind/kernels/linear_attn/kernel/sm_120/internal.h"

#include "src/turbomind/kernels/linear_attn/kernel/plan.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_120/cp_fwd.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_120/fused_fwd.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_120/prepare_h.h"
#include "src/turbomind/kernels/linear_attn/registrar.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace turbomind::linear_attn::delta_rule {
namespace {

template<GdrMode Mode, DataType StateType>
class Sm120GdrKernel final: public GdrKernel {
    static_assert(StateType == kFloat32 || StateType == kBfloat16);

    using StateT = std::conditional_t<StateType == kFloat32, float, __nv_bfloat16>;

    static int SelectFusedBlockDv(const Problem& problem, const ContextParallelPlan& cp)
    {
        const int total_chunks = cp.enabled ? cp.total_chunks : problem.total_chunks;
        const int max_chunks = cp.enabled ? (cp.total_chunks > 0 ? cp.segment_chunks : 0) : problem.max_sequence_chunks;
        if (total_chunks <= 0 || max_chunks <= 0 || problem.hv <= 0) {
            return kContextParallelGdrBlockDv;
        }

        // A CTA owns one sequence/head/Dv tile for the entire sequence. The raw
        // sequence count therefore overstates useful parallelism for imbalanced
        // varlen batches. Normalize total work by the longest sequence so dispatch
        // follows the effective head grid that controls the launch tail.
        const int64_t effective_grid_numerator = int64_t(total_chunks) * problem.hv;
        const int64_t narrow_grid_limit        = std::max((problem.sm_count + 4) / 5, 1);
        return effective_grid_numerator <= int64_t(max_chunks) * narrow_grid_limit ? kContextParallelGdrBlockDv :
                                                                                     kFusedGdrBlockDv;
    }

    template<int BlockDv>
    static void RunDirectChunk32(const Arguments& args, const delta_rule::Plan& plan, cudaStream_t stream)
    {
        auto        workspace               = detail::PartitionSm120DirectChunkWorkspace(args, plan);
        const auto& beta                    = args.beta;
        auto        execution_problem       = plan.problem;
        execution_problem.gate_stride       = workspace.g_cumsum.stride(1);
        execution_problem.gate_batch_stride = workspace.g_cumsum.stride(0);
        execution_problem.beta_stride       = beta.stride(1);
        execution_problem.beta_batch_stride = beta.stride(0);
        detail::LaunchChunk32LocalCumsumAndPrepareDirect<BlockDv>(args.q,
                                                                  args.k,
                                                                  args.v,
                                                                  args.g,
                                                                  args.q_offsets,
                                                                  workspace.g_cumsum,
                                                                  workspace.resolvent,
                                                                  *args.out,
                                                                  workspace.kkt_tma_desc,
                                                                  workspace.fused_tma_desc,
                                                                  execution_problem,
                                                                  stream);
        detail::LaunchSm120KktSolve(args.k,
                                    beta,
                                    args.q_offsets,
                                    &workspace.g_cumsum,
                                    args.finished,
                                    workspace.resolvent,
                                    execution_problem,
                                    workspace.kkt_tma_desc,
                                    stream);
        LaunchSm120FusedGdrFwd<StateT, BlockDv, false>(args.q,
                                                       args.k,
                                                       args.v,
                                                       workspace.g_cumsum,
                                                       beta,
                                                       workspace.resolvent,
                                                       args.state_ptrs,
                                                       args.q_offsets,
                                                       args.finished,
                                                       *args.out,
                                                       execution_problem,
                                                       args.state_layer_offset,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       plan.problem.sequence_num,
                                                       workspace.fused_tma_desc,
                                                       stream);
    }

    template<int BlockDv>
    static void RunContextParallelChunk32(const Arguments& args, const delta_rule::Plan& plan, cudaStream_t stream)
    {
        auto        workspace               = detail::PartitionSm120ContextParallelWorkspace(args, plan);
        const auto& beta                    = args.beta;
        auto        execution_problem       = plan.problem;
        execution_problem.gate_stride       = workspace.g_cumsum.stride(1);
        execution_problem.gate_batch_stride = workspace.g_cumsum.stride(0);
        execution_problem.beta_stride       = beta.stride(1);
        execution_problem.beta_batch_stride = beta.stride(0);
        detail::PrepareSm120GdrTmaDescriptors<StateT, BlockDv>(args.q,
                                                               args.k,
                                                               args.v,
                                                               workspace.g_cumsum,
                                                               workspace.resolvent,
                                                               args.state_ptrs,
                                                               args.q_offsets,
                                                               args.finished,
                                                               args.out,
                                                               *args.workspace,
                                                               execution_problem,
                                                               plan.cp,
                                                               Sm120GdrTmaMode::kAllContextParallel,
                                                               workspace.layout,
                                                               args.state_layer_offset,
                                                               stream);
        detail::LaunchChunk32LocalCumsum(args.g, args.q_offsets, workspace.g_cumsum, execution_problem, stream);
        detail::LaunchSm120KktSolve(args.k,
                                    beta,
                                    args.q_offsets,
                                    &workspace.g_cumsum,
                                    args.finished,
                                    workspace.resolvent,
                                    execution_problem,
                                    workspace.kkt_tma_desc,
                                    stream);
        LaunchSm120FusedGdrHTyped<kFusedGdrHBlockDv>(args.k,
                                                     args.v,
                                                     workspace.g_cumsum,
                                                     beta,
                                                     workspace.resolvent,
                                                     workspace.segment_state,
                                                     workspace.segment_m,
                                                     execution_problem,
                                                     plan.cp,
                                                     args.q_offsets,
                                                     workspace.cp_source_indices,
                                                     workspace.cp_q_offsets,
                                                     workspace.cp_finished,
                                                     workspace.cp_fallback,
                                                     workspace.fused_gdr_h_tma_desc,
                                                     stream);
        LaunchSm120CorrectInitialStatesTyped<StateT>(workspace.cp_state,
                                                     args.state_ptrs,
                                                     workspace.cp_sequence_starts,
                                                     workspace.segment_state,
                                                     workspace.segment_m,
                                                     workspace.cp_fallback,
                                                     execution_problem,
                                                     plan.cp,
                                                     args.state_layer_offset,
                                                     stream);
        auto context_parallel_problem = detail::MakeSm120ContextParallelProblem(execution_problem, plan.cp);
        LaunchSm120FusedGdrFwd<StateT, BlockDv, true>(args.q,
                                                      args.k,
                                                      args.v,
                                                      workspace.g_cumsum,
                                                      beta,
                                                      workspace.resolvent,
                                                      args.state_ptrs,
                                                      workspace.cp_q_offsets,
                                                      workspace.cp_finished,
                                                      *args.out,
                                                      context_parallel_problem,
                                                      args.state_layer_offset,
                                                      &args.q_offsets,
                                                      &workspace.cp_source_indices,
                                                      &workspace.cp_state_ptrs,
                                                      plan.problem.sequence_num,
                                                      workspace.context_parallel_fused_gdr_tma_desc,
                                                      stream);
    }

    template<int BlockDv>
    static void RunChunk32(const Arguments& args, const delta_rule::Plan& plan, cudaStream_t stream)
    {
        static_assert(BlockDv == kContextParallelGdrBlockDv || BlockDv == kFusedGdrBlockDv);
        if (plan.cp.enabled) {
            RunContextParallelChunk32<BlockDv>(args, plan, stream);
        }
        else {
            RunDirectChunk32<BlockDv>(args, plan, stream);
        }
    }

public:
    const GdrKernelSpec& spec() const noexcept override
    {
        static constexpr GdrKernelSpec kSpec{
            "sm120", Mode, kBfloat16, StateType, 128, Mode == GdrMode::kRecurrent ? 1 : 32};
        return kSpec;
    }

    const char* name() const noexcept override
    {
        if constexpr (Mode == GdrMode::kRecurrent && StateType == kFloat32) {
            return "sm120_delta_rule_recurrent_f32_state";
        }
        else if constexpr (Mode == GdrMode::kRecurrent) {
            return "sm120_delta_rule_recurrent_bf16_state";
        }
        else if constexpr (StateType == kFloat32) {
            return "sm120_delta_rule_chunk32_f32_state";
        }
        else {
            return "sm120_delta_rule_chunk32_bf16_state";
        }
    }

    bool Match(const Operation& operation, const PlanningContext& context) const override
    {
        return detail::MatchesGdrSpec(spec(), operation, context);
    }

    bool Plan(const Operation& operation, const PlanningContext& context, delta_rule::Plan* plan) const override
    {
        return detail::PlanSm120Operation(spec(), operation, context, plan);
    }

    void PrepareState(const core::Tensor&     state_ptrs,
                      core::Tensor&           state_tma_descs,
                      int                     layer_groups,
                      int                     layers_per_block,
                      const delta_rule::Plan& plan,
                      cudaStream_t            stream) const override
    {
        if constexpr (Mode == GdrMode::kRecurrent) {
            detail::PrepareSm120RecurrentStateTmaDescriptors<StateT>(
                state_ptrs, state_tma_descs, layer_groups, layers_per_block, plan, stream);
        }
    }

    void Run(const Arguments& args, const delta_rule::Plan& plan, cudaStream_t stream) const override
    {
        if constexpr (Mode == GdrMode::kRecurrent) {
            detail::LaunchSm120Recurrent<StateT>(args.q,
                                                 args.k,
                                                 args.v,
                                                 args.g,
                                                 args.beta,
                                                 args.finished,
                                                 args.state_tma_descs,
                                                 *args.out,
                                                 plan.problem,
                                                 args.state_layer_offset,
                                                 stream);
        }
        else if (SelectFusedBlockDv(plan.problem, plan.cp) == kContextParallelGdrBlockDv) {
            RunChunk32<kContextParallelGdrBlockDv>(args, plan, stream);
        }
        else {
            RunChunk32<kFusedGdrBlockDv>(args, plan, stream);
        }
    }
};

Registrar reg([](Collector& c) {
    c.add<Sm120GdrKernel<GdrMode::kRecurrent, kFloat32>>();
    c.add<Sm120GdrKernel<GdrMode::kRecurrent, kBfloat16>>();
    c.add<Sm120GdrKernel<GdrMode::kChunked, kFloat32>>();
    c.add<Sm120GdrKernel<GdrMode::kChunked, kBfloat16>>();
});

}  // namespace

}  // namespace turbomind::linear_attn::delta_rule
