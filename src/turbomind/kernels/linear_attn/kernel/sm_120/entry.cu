#include "src/turbomind/kernels/linear_attn/kernel/sm_120/internal.h"

#include "src/turbomind/kernels/linear_attn/kernel/plan.h"
#include "src/turbomind/kernels/linear_attn/kernel/sm_120/fused_fwd.h"
#include "src/turbomind/kernels/linear_attn/registry.h"

#include <cuda_bf16.h>

#include <memory>

namespace turbomind::linear_attn::delta_rule {
namespace detail {

void LaunchSm120FusedChunk(const core::Tensor& q,
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
    if (state_dtype == kFloat32) {
        LaunchSm120FusedGdrFwdRegistered<float, kFusedGdrBlockDv>(q,
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
        LaunchSm120FusedGdrFwdRegistered<__nv_bfloat16, kFusedGdrBlockDv>(q,
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

namespace {

void RunSm120RecurrentEntry(const Arguments& args, const Plan& plan, cudaStream_t stream)
{
    LaunchSm120Recurrent(args.q,
                         args.k,
                         args.v,
                         args.g,
                         args.beta,
                         args.finished,
                         args.state_tma_descs,
                         *args.out,
                         plan.problem,
                         args.state_layer_offset,
                         plan.problem.state_dtype,
                         stream);
}

void RunSm120Chunk32Entry(const Arguments& args, const Plan& plan, cudaStream_t stream)
{
    auto        workspace               = PartitionSm120DirectChunkWorkspace(args, plan);
    const auto& beta                    = args.beta;
    auto        execution_problem       = plan.problem;
    execution_problem.gate_stride       = workspace.g_cumsum.stride(1);
    execution_problem.gate_batch_stride = workspace.g_cumsum.stride(0);
    execution_problem.beta_stride       = beta.stride(1);
    execution_problem.beta_batch_stride = beta.stride(0);
    PrepareSm120GdrTmaDescriptors(args.q,
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
                                  Sm120GdrTmaMode::kAllDirectFused,
                                  workspace.layout,
                                  args.state_layer_offset,
                                  plan.problem.state_dtype,
                                  stream);
    LaunchChunk32LocalCumsum(args.g, args.q_offsets, workspace.g_cumsum, execution_problem, stream);
    LaunchSm120KktSolve(args.k,
                        beta,
                        args.q_offsets,
                        &workspace.g_cumsum,
                        args.finished,
                        workspace.resolvent,
                        execution_problem,
                        workspace.kkt_tma_desc,
                        stream);
    LaunchSm120FusedChunk(args.q,
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
                          plan.problem.state_dtype,
                          nullptr,
                          nullptr,
                          nullptr,
                          plan.problem.sequence_num,
                          workspace.fused_tma_desc,
                          stream);
}

}  // namespace
}  // namespace detail

namespace {

template<GdrMode Mode, DataType StateType>
class Sm120GdrKernel final: public GdrKernel {
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
        return context.arch == 1200 && detail::MatchesGdrSpec(spec(), operation, context);
    }

    bool Plan(const Operation&, const PlanningContext& context, delta_rule::Plan* plan) const override
    {
        return detail::PlanSm120Operation(spec(), context, plan);
    }

    void PrepareState(const core::Tensor&     state_ptrs,
                      core::Tensor&           state_tma_descs,
                      int                     layer_groups,
                      int                     layers_per_block,
                      const delta_rule::Plan& plan,
                      cudaStream_t            stream) const override
    {
        if constexpr (Mode == GdrMode::kRecurrent) {
            detail::PrepareSm120RecurrentStateTmaDescriptors(
                state_ptrs, state_tma_descs, layer_groups, layers_per_block, plan, stream);
        }
    }

    void Run(const Arguments& args, const delta_rule::Plan& plan, cudaStream_t stream) const override
    {
        if constexpr (Mode == GdrMode::kRecurrent) {
            detail::RunSm120RecurrentEntry(args, plan, stream);
        }
        else {
            detail::RunSm120Chunk32Entry(args, plan, stream);
        }
    }
};

template<GdrMode Mode, DataType StateType>
void Add(GdrKernelRegistry& registry)
{
    registry.Add(std::make_unique<Sm120GdrKernel<Mode, StateType>>());
}

bool RegisterSm120GdrOperations()
{
    auto& registry = GdrKernelRegistry::instance();
    Add<GdrMode::kRecurrent, kFloat32>(registry);
    Add<GdrMode::kRecurrent, kBfloat16>(registry);
    Add<GdrMode::kChunked, kFloat32>(registry);
    Add<GdrMode::kChunked, kBfloat16>(registry);
    return true;
}

[[maybe_unused]] const bool kSm120GdrOperationsRegistered = RegisterSm120GdrOperations();

}  // namespace

}  // namespace turbomind::linear_attn::delta_rule
