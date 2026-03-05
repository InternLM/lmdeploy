// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/registry.h"

#include <memory>
#include <tuple>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/kernels/attention/registrar.h"
#include "src/turbomind/kernels/core/math.h"

namespace turbomind::attention {

namespace {

constexpr float kMaxWasteRatio = 1.f;

template<int DeviceId>
Registry& device_instance()
{
    static Registry reg = [] {
        auto prop = std::make_shared<cudaDeviceProp>();
        cudaGetDeviceProperties(prop.get(), DeviceId);
        return Registry{std::move(prop)};
    }();
    return reg;
}

}  // namespace

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
    const int threshold = static_cast<int>(kMaxWasteRatio * desc.query_group_sz);

    const Kernel*             best = nullptr;
    std::tuple<int, int, int> cost{};

    for (const auto* k : ptrs_) {
        const auto& d = k->desc();
        if (d.mode != desc.mode || d.head_dim != desc.head_dim  //
            || d.data_type != desc.data_type || d.kv_quant != desc.kv_quant) {
            continue;
        }
        if (desc.mode == AttnDesc::kDecoding) {
            const int ctas  = cdiv(desc.query_group_sz, d.qh);
            const int waste = d.qh * ctas - desc.query_group_sz;

            const auto v = std::make_tuple(waste > threshold, ctas, waste);
            if (!best || v < cost) {
                best = k;
                cost = v;
            }
        }
        else {  // attention, return on first match
            return k;
        }
    }
    return best;
}

Registry& Registry::instance()
{
    int device_id{};
    cudaGetDevice(&device_id);
    switch (device_id) {
        case 0:
            return device_instance<0>();
        case 1:
            return device_instance<1>();
        case 2:
            return device_instance<2>();
        case 3:
            return device_instance<3>();
        case 4:
            return device_instance<4>();
        case 5:
            return device_instance<5>();
        case 6:
            return device_instance<6>();
        case 7:
            return device_instance<7>();
        default:
            TM_CHECK(false) << "unsupported device_id: " << device_id;
    }
    TM_UNREACHABLE;
}

}  // namespace turbomind::attention
