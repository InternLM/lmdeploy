#include "src/turbomind/kernels/linear_attn/kernel/pre_sm90/internal.h"

#include "src/turbomind/kernels/linear_attn/kernel/plan.h"

namespace turbomind::linear_attn::delta_rule::detail {

bool PlanPreSm90Operation(const GdrKernelSpec&   spec,
                          const Operation&       operation,
                          const PlanningContext& context,
                          Plan*                  plan)
{
    plan->problem                        = BuildProblem(context, spec);
    plan->cp                             = BuildDisabledContextParallelPlan(plan->problem, operation.cp_level);
    const core::ssize_t value_row_size   = core::ssize_t(plan->problem.hv) * 128;
    const core::ssize_t value_batch_size = core::ssize_t(plan->problem.token_num) * value_row_size;
    const size_t        value_elements   = size_t(plan->problem.batch) * size_t(value_batch_size);
    plan->out       = TensorPlan{core::Layout{{plan->problem.batch, plan->problem.token_num, plan->problem.hv, 128},
                                        {value_batch_size, value_row_size, 128, 1}},
                           plan->problem.input_dtype,
                           value_elements};
    plan->g_cumsum  = TensorPlan{core::Layout{{0}}, kFloat32};
    plan->resolvent = TensorPlan{core::Layout{{0}}, plan->problem.input_dtype};
    plan->workspace = TensorPlan{core::Layout{{0}}, kUint8};
    plan->workspace_bytes                      = 0;
    plan->state_tma_desc_bytes_per_layer_group = 0;
    return true;
}

}  // namespace turbomind::linear_attn::delta_rule::detail
