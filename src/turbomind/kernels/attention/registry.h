// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <vector>

#include "src/turbomind/kernels/attention/kernel_impl.h"

namespace turbomind::attention {

class Registry {
public:
    explicit Registry(std::shared_ptr<cudaDeviceProp> device_prop);

    template<class KernelType>
    [[maybe_unused]] bool Add()
    {
        return Add(std::make_unique<KernelImpl<KernelType>>());
    }

    const Kernel* Find(const AttnDesc& desc) const;

    [[nodiscard]] const std::vector<Kernel*>& kernels() const
    {
        return ptrs_;
    }

    static Registry& instance();


private:
    bool Add(std::unique_ptr<Kernel> kernel);

    std::shared_ptr<cudaDeviceProp>      device_prop_;
    int                                  arch_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
    std::vector<Kernel*>                 ptrs_;
};

}  // namespace turbomind::attention
