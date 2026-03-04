// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/kernels/attention/registrar.h"
#include "src/turbomind/kernels/attention/registry.h"

namespace turbomind::attention {

Registry::Registry(std::shared_ptr<cudaDeviceProp> device_prop):
    device_prop_{std::move(device_prop)}, arch_{device_prop_->major * 100 + device_prop_->minor * 10}
{
    for (auto& register_fn : gKernelFactories()) {
        Collector collector;
        register_fn(collector);
        for (auto& k : collector.release()) {
            Add(std::move(k));
        }
    }
}

bool Registry::Add(std::unique_ptr<Kernel> kernel)
{
    bool is_valid = true;

    if (!arch::is_arch_compatible(kernel->arch(), arch_)) {
        is_valid = false;
    }

    if ((int)device_prop_->sharedMemPerBlockOptin < kernel->smem_size()) {
        is_valid = false;
    }

    if (is_valid) {
        ptrs_.push_back(kernels_.emplace_back(std::move(kernel)).get());
    }

    return is_valid;
}

const Kernel* Registry::Find(const AttnDesc& desc) const
{
    const Kernel* best = nullptr;
    for (const auto* k : ptrs_) {
        const auto& d = k->desc();
        if (d.mode != desc.mode || d.head_dim != desc.head_dim  //
            || d.is_bf16 != desc.is_bf16 || d.kv_quant != desc.kv_quant) {
            continue;
        }
        if (d.qh < desc.qh) {
            continue;
        }
        if (!best || d.qh < best->desc().qh) {
            best = k;
        }
    }
    return best;
}

Registry& Registry::instance()
{
    static auto reg = [] {
        int device_id{};
        cudaGetDevice(&device_id);
        auto prop = std::make_shared<cudaDeviceProp>();
        cudaGetDeviceProperties(prop.get(), device_id);
        return Registry{std::move(prop)};
    }();
    return reg;
}

}  // namespace turbomind::attention
