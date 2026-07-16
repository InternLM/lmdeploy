#include "src/turbomind/kernels/linear_attn/delta_rule.h"

#include "src/turbomind/kernels/linear_attn/registry.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace turbomind::linear_attn::delta_rule {
namespace {

const char* ModeName(GdrMode mode)
{
    switch (mode) {
        case GdrMode::kRecurrent:
            return "recurrent";
        case GdrMode::kChunked:
            return "chunked";
    }
    return "invalid";
}

const GdrKernel& RequireGdrKernel(const Operation& operation, const PlanningContext& context)
{
    const auto* kernel = GdrKernelRegistry::instance().Find(operation, context);
    if (kernel == nullptr) {
        throw std::invalid_argument(
            std::string{"GDR operation is not registered for requested configuration arch="}
            + std::to_string(context.arch) + " mode=" + ModeName(operation.mode)
            + " input_dtype=" + to_string(context.input_dtype)
            + " state_dtype=" + to_string(context.state_dtype)
            + " head_dim=" + std::to_string(context.head_dim)
            + " chunk_size=" + std::to_string(operation.chunk_size));
    }
    return *kernel;
}

const GdrKernel& RequireGdrKernel(const Plan& plan)
{
    if (plan.kernel == nullptr) {
        throw std::invalid_argument("GDR plan does not reference a registered kernel");
    }
    return *plan.kernel;
}

}  // namespace

bool GatedDeltaRule::Plan(const Operation& operation,
                          const PlanningContext& context,
                          delta_rule::Plan* plan) const
{
    const auto& kernel = RequireGdrKernel(operation, context);
    delta_rule::Plan candidate{};
    candidate.kernel = &kernel;
    if (!kernel.Plan(operation, context, &candidate)) {
        return false;
    }
    *plan = std::move(candidate);
    return true;
}

void GatedDeltaRule::PrepareState(const core::Tensor& state_ptrs,
                                  core::Tensor& state_tma_descs,
                                  int layer_groups,
                                  int layers_per_block,
                                  const delta_rule::Plan& plan,
                                  cudaStream_t stream) const
{
    RequireGdrKernel(plan).PrepareState(
        state_ptrs, state_tma_descs, layer_groups, layers_per_block, plan, stream);
}

void GatedDeltaRule::Run(const Arguments& args,
                         const delta_rule::Plan& plan,
                         cudaStream_t stream) const
{
    RequireGdrKernel(plan).Run(args, plan, stream);
}

}  // namespace turbomind::linear_attn::delta_rule
