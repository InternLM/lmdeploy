#include "src/turbomind/kernels/linear_attn/delta_rule.h"

#include "src/turbomind/kernels/linear_attn/registry.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace turbomind::linear_attn::delta_rule {
namespace {

constexpr std::string_view kPreSm90Architecture{"pre_sm90"};
constexpr std::string_view kSm90Architecture{"sm90"};
constexpr std::string_view kSm120Architecture{"sm120"};

bool ForceLegacyGdr()
{
    struct Result {
        bool        value{};
        std::string error;
    };

    static const Result result = [] {
        const char* value = std::getenv("TM_GDR_FORCE_LEGACY");
        if (value == nullptr || std::strcmp(value, "0") == 0) {
            return Result{false, {}};
        }
        if (std::strcmp(value, "1") == 0) {
            return Result{true, {}};
        }
        return Result{false, std::string{"TM_GDR_FORCE_LEGACY must be 0 or 1, got "} + value};
    }();

    if (!result.error.empty()) {
        throw std::invalid_argument(result.error);
    }
    return result.value;
}

std::string_view SelectArchitecture(int arch, bool force_legacy)
{
    if (force_legacy || (arch > 0 && arch < 900)) {
        return kPreSm90Architecture;
    }
    if (arch == 900) {
        return kSm90Architecture;
    }
    if (arch == 1200) {
        return kSm120Architecture;
    }
    return {};
}

Operation SelectOperation(const Operation& requested, bool force_legacy)
{
    Operation selected = requested;
    if (force_legacy) {
        selected.chunk_size = kAutoGdrChunkSize;
    }
    return selected;
}

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

const GdrKernel&
RequireGdrKernel(const Operation& operation, const PlanningContext& context, std::string_view architecture)
{
    const auto* kernel = GdrKernelRegistry::instance().Find(operation, context, architecture);
    if (kernel == nullptr) {
        const std::string kernel_architecture = architecture.empty() ? "unsupported" : std::string{architecture};
        throw std::invalid_argument(
            std::string{"GDR operation is not registered for requested configuration arch="}
            + std::to_string(context.arch) + " kernel_arch=" + kernel_architecture + " mode=" + ModeName(operation.mode)
            + " input_dtype=" + to_string(context.input_dtype) + " state_dtype=" + to_string(context.state_dtype)
            + " head_dim=" + std::to_string(context.head_dim) + " chunk_size=" + std::to_string(operation.chunk_size));
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

bool GatedDeltaRule::Plan(const Operation& requested, const PlanningContext& context, delta_rule::Plan* plan) const
{
    const bool       force_legacy = ForceLegacyGdr();
    const Operation  operation    = SelectOperation(requested, force_legacy);
    const auto       architecture = SelectArchitecture(context.arch, force_legacy);
    const auto&      kernel       = RequireGdrKernel(operation, context, architecture);
    delta_rule::Plan candidate{};
    candidate.kernel = &kernel;
    if (!kernel.Plan(operation, context, &candidate)) {
        return false;
    }
    *plan = std::move(candidate);
    return true;
}

void GatedDeltaRule::PrepareState(const core::Tensor&     state_ptrs,
                                  core::Tensor&           state_tma_descs,
                                  int                     layer_groups,
                                  int                     layers_per_block,
                                  const delta_rule::Plan& plan,
                                  cudaStream_t            stream) const
{
    RequireGdrKernel(plan).PrepareState(state_ptrs, state_tma_descs, layer_groups, layers_per_block, plan, stream);
}

void GatedDeltaRule::Run(const Arguments& args, const delta_rule::Plan& plan, cudaStream_t stream) const
{
    RequireGdrKernel(plan).Run(args, plan, stream);
}

}  // namespace turbomind::linear_attn::delta_rule
