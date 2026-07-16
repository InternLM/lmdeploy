#include "src/turbomind/kernels/linear_attn/kernel/sm_90/internal.h"

#include "src/turbomind/kernels/linear_attn/kernel/plan.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_90/cp_fwd.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_90/fused_fwd.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_90/prepare_h.h"
#include "src/turbomind/kernels/linear_attn/registry.h"

#include <cuda_bf16.h>

#include <memory>

namespace turbomind::linear_attn::delta_rule {
namespace detail {

void LaunchSm90FusedChunk(const core::Tensor& q,
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
                          DataType            state_dtype,
                          const core::Tensor* data_q_offsets,
                          const core::Tensor* cp_source_indices,
                          const core::Tensor* cp_state_ptrs,
                          int                 data_sequence_num,
                          void*               tma_desc_workspace,
                          cudaStream_t        stream)
{
    const bool context_parallel = cp_source_indices != nullptr;
    const int  block_dv         = FusedChunkGdrBlockDv(problem, context_parallel);
    if (state_dtype == kFloat32) {
        if (block_dv == 32) {
            LaunchSm90FusedGdrFwdRegistered<float, 32>(q,
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
        else if (block_dv == 128) {
            LaunchSm90FusedGdrFwdRegistered<float, 128>(q,
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
        else {
            LaunchSm90FusedGdrFwdRegistered<float, 64>(q,
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
    }
    else if (block_dv == 32) {
        LaunchSm90FusedGdrFwdRegistered<__nv_bfloat16, 32>(q,
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
    else if (block_dv == 128) {
        LaunchSm90FusedGdrFwdRegistered<__nv_bfloat16, 128>(q,
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
    else {
        LaunchSm90FusedGdrFwdRegistered<__nv_bfloat16, 64>(q,
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
}

void LaunchSm90FusedGdrH(const core::Tensor&        k,
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
    LaunchSm90FusedGdrHTyped<kFusedGdrHBlockDv>(k,
                                                v,
                                                g_cumsum,
                                                beta,
                                                resolvent,
                                                segment_state,
                                                segment_m,
                                                problem,
                                                cp,
                                                q_offsets,
                                                cp_source_indices,
                                                cp_q_offsets,
                                                cp_finished,
                                                cp_fallback,
                                                tma_desc_workspace,
                                                stream);
}

void LaunchSm90CorrectInitialStates(core::Tensor&              cp_state,
                                    const core::Tensor&        state_ptrs,
                                    const core::Tensor&        finished,
                                    const core::Tensor&        cp_sequence_starts,
                                    const core::Tensor&        segment_state,
                                    const core::Tensor&        segment_m,
                                    const core::Tensor&        cp_fallback,
                                    const Problem&             problem,
                                    const ContextParallelPlan& cp,
                                    int64_t                    state_layer_offset,
                                    DataType                   state_dtype,
                                    void*                      tma_desc_workspace,
                                    cudaStream_t               stream)
{
    if (state_dtype == kFloat32) {
        LaunchSm90CorrectInitialStatesTyped<float>(cp_state,
                                                   state_ptrs,
                                                   finished,
                                                   cp_sequence_starts,
                                                   segment_state,
                                                   segment_m,
                                                   cp_fallback,
                                                   problem,
                                                   cp,
                                                   state_layer_offset,
                                                   tma_desc_workspace,
                                                   stream);
    }
    else {
        LaunchSm90CorrectInitialStatesTyped<__nv_bfloat16>(cp_state,
                                                           state_ptrs,
                                                           finished,
                                                           cp_sequence_starts,
                                                           segment_state,
                                                           segment_m,
                                                           cp_fallback,
                                                           problem,
                                                           cp,
                                                           state_layer_offset,
                                                           tma_desc_workspace,
                                                           stream);
    }
}

namespace {

void RunSm90RecurrentEntry(const Arguments& args, const Plan& plan, cudaStream_t stream)
{
    LaunchSm90Recurrent(args.q,
                        args.k,
                        args.v,
                        args.g,
                        args.beta,
                        args.finished,
                        args.state_ptrs,
                        args.state_tma_descs,
                        *args.out,
                        plan.problem,
                        args.state_layer_offset,
                        plan.problem.state_dtype,
                        stream);
}

void RunSm90Chunk64WithoutContextParallel(const Arguments& args, const Plan& plan, cudaStream_t stream)
{
    auto workspace = PartitionSm90DirectChunkWorkspace(args, plan);
    PrepareSm90GdrTmaDescriptorsAndCumsum(args.q,
                                          args.k,
                                          args.v,
                                          args.g,
                                          workspace.g_cumsum,
                                          args.beta,
                                          workspace.resolvent,
                                          args.q_offsets,
                                          args.finished,
                                          *args.out,
                                          *args.workspace,
                                          plan.problem,
                                          plan.cp,
                                          workspace.layout,
                                          plan.problem.state_dtype,
                                          stream);
    LaunchSm90KktSolve(args.k,
                       args.beta,
                       args.q_offsets,
                       &workspace.g_cumsum,
                       args.finished,
                       workspace.resolvent,
                       plan.problem,
                       workspace.kkt_tma_desc,
                       stream);
    LaunchSm90FusedChunk(args.q,
                         args.k,
                         args.v,
                         workspace.g_cumsum,
                         args.beta,
                         workspace.resolvent,
                         args.state_ptrs,
                         args.q_offsets,
                         args.finished,
                         *args.out,
                         plan.problem,
                         args.state_layer_offset,
                         plan.problem.state_dtype,
                         nullptr,
                         nullptr,
                         nullptr,
                         plan.problem.sequence_num,
                         workspace.fused_tma_desc,
                         stream);
}

void RunSm90Chunk64Entry(const Arguments& args, const Plan& plan, cudaStream_t stream)
{
    if (!plan.cp.enabled) {
        RunSm90Chunk64WithoutContextParallel(args, plan, stream);
        return;
    }

    auto workspace = PartitionSm90ContextParallelWorkspace(args, plan);
    PrepareSm90GdrTmaDescriptorsAndCumsum(args.q,
                                          args.k,
                                          args.v,
                                          args.g,
                                          workspace.g_cumsum,
                                          args.beta,
                                          workspace.resolvent,
                                          args.q_offsets,
                                          args.finished,
                                          *args.out,
                                          *args.workspace,
                                          plan.problem,
                                          plan.cp,
                                          workspace.layout,
                                          plan.problem.state_dtype,
                                          stream);
    LaunchSm90KktSolve(args.k,
                       args.beta,
                       args.q_offsets,
                       &workspace.g_cumsum,
                       args.finished,
                       workspace.resolvent,
                       plan.problem,
                       workspace.kkt_tma_desc,
                       stream);
    LaunchSm90FusedGdrH(args.k,
                        args.v,
                        workspace.g_cumsum,
                        args.beta,
                        workspace.resolvent,
                        workspace.segment_state,
                        workspace.segment_m,
                        plan.problem,
                        plan.cp,
                        args.q_offsets,
                        workspace.cp_source_indices,
                        workspace.cp_q_offsets,
                        workspace.cp_finished,
                        workspace.cp_fallback,
                        workspace.fused_gdr_h_tma_desc,
                        stream);
    LaunchSm90CorrectInitialStates(workspace.cp_state,
                                   args.state_ptrs,
                                   args.finished,
                                   workspace.cp_sequence_starts,
                                   workspace.segment_state,
                                   workspace.segment_m,
                                   workspace.cp_fallback,
                                   plan.problem,
                                   plan.cp,
                                   args.state_layer_offset,
                                   plan.problem.state_dtype,
                                   workspace.correct_initial_states_tma_desc,
                                   stream);
    Problem context_parallel_problem = MakeContextParallelProblem(plan.problem, plan.cp);
    LaunchSm90FusedChunk(args.q,
                         args.k,
                         args.v,
                         workspace.g_cumsum,
                         args.beta,
                         workspace.resolvent,
                         args.state_ptrs,
                         workspace.cp_q_offsets,
                         workspace.cp_finished,
                         *args.out,
                         context_parallel_problem,
                         args.state_layer_offset,
                         plan.problem.state_dtype,
                         &args.q_offsets,
                         &workspace.cp_source_indices,
                         &workspace.cp_state_ptrs,
                         plan.problem.sequence_num,
                         workspace.context_parallel_fused_gdr_tma_desc,
                         stream);
}

}  // namespace
}  // namespace detail

namespace {

template<GdrMode Mode, DataType StateType>
class Sm90GdrKernel final: public GdrKernel {
public:
    const GdrKernelSpec& spec() const noexcept override
    {
        static constexpr GdrKernelSpec kSpec{
            "sm90", Mode, kBfloat16, StateType, 128, Mode == GdrMode::kRecurrent ? 1 : 64};
        return kSpec;
    }

    const char* name() const noexcept override
    {
        if constexpr (Mode == GdrMode::kRecurrent && StateType == kFloat32) {
            return "sm90_delta_rule_recurrent_f32_state";
        }
        else if constexpr (Mode == GdrMode::kRecurrent) {
            return "sm90_delta_rule_recurrent_bf16_state";
        }
        else if constexpr (StateType == kFloat32) {
            return "sm90_delta_rule_chunk64_f32_state";
        }
        else {
            return "sm90_delta_rule_chunk64_bf16_state";
        }
    }

    bool Match(const Operation& operation, const PlanningContext& context) const override
    {
        return context.arch == 900 && detail::MatchesGdrSpec(spec(), operation, context);
    }

    bool Plan(const Operation& operation, const PlanningContext& context, delta_rule::Plan* plan) const override
    {
        return detail::PlanSm90Operation(spec(), operation, context, plan);
    }

    void PrepareState(const core::Tensor&     state_ptrs,
                      core::Tensor&           state_tma_descs,
                      int                     layer_groups,
                      int                     layers_per_block,
                      const delta_rule::Plan& plan,
                      cudaStream_t            stream) const override
    {
        if constexpr (Mode == GdrMode::kRecurrent) {
            detail::PrepareSm90RecurrentStateTmaDescriptors(
                state_ptrs, state_tma_descs, layer_groups, layers_per_block, plan, stream);
        }
    }

    void Run(const Arguments& args, const delta_rule::Plan& plan, cudaStream_t stream) const override
    {
        if constexpr (Mode == GdrMode::kRecurrent) {
            detail::RunSm90RecurrentEntry(args, plan, stream);
        }
        else {
            detail::RunSm90Chunk64Entry(args, plan, stream);
        }
    }
};

template<GdrMode Mode, DataType StateType>
void Add(GdrKernelRegistry& registry)
{
    registry.Add(std::make_unique<Sm90GdrKernel<Mode, StateType>>());
}

bool RegisterSm90GdrOperations()
{
    auto& registry = GdrKernelRegistry::instance();
    Add<GdrMode::kRecurrent, kFloat32>(registry);
    Add<GdrMode::kRecurrent, kBfloat16>(registry);
    Add<GdrMode::kChunked, kFloat32>(registry);
    Add<GdrMode::kChunked, kBfloat16>(registry);
    return true;
}

[[maybe_unused]] const bool kSm90GdrOperationsRegistered = RegisterSm90GdrOperations();

}  // namespace

}  // namespace turbomind::linear_attn::delta_rule
