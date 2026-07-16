#include "src/turbomind/kernels/linear_attn/registry.h"

#include <utility>

namespace turbomind::linear_attn::delta_rule {
bool GdrKernelRegistry::Add(std::unique_ptr<GdrKernel> kernel)
{
    kernels_.push_back(std::move(kernel));
    return true;
}

const GdrKernel* GdrKernelRegistry::Find(const Operation& operation, const PlanningContext& context) const
{
    for (const auto& kernel : kernels_) {
        if (kernel->Match(operation, context)) {
            return kernel.get();
        }
    }
    return nullptr;
}

GdrKernelRegistry& GdrKernelRegistry::instance()
{
    static GdrKernelRegistry registry;
    return registry;
}

}  // namespace turbomind::linear_attn::delta_rule
