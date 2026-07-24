#include "src/turbomind/kernels/linear_attn/registry.h"

#include "src/turbomind/kernels/linear_attn/registrar.h"

#include <utility>

namespace turbomind::linear_attn::delta_rule {

GdrKernelRegistry::GdrKernelRegistry()
{
    for (auto& register_fn : gKernelFactories()) {
        Collector collector;
        register_fn(collector);
        for (auto& kernel : collector.release()) {
            Add(std::move(kernel));
        }
    }
}

bool GdrKernelRegistry::Add(std::unique_ptr<GdrKernel> kernel)
{
    kernels_.push_back(std::move(kernel));
    return true;
}

const GdrKernel*
GdrKernelRegistry::Find(const Operation& operation, const PlanningContext& context, std::string_view architecture) const
{
    for (const auto& kernel : kernels_) {
        if (std::string_view{kernel->spec().architecture} == architecture && kernel->Match(operation, context)) {
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
