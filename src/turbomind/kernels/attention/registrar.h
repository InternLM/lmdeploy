// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "src/turbomind/kernels/attention/kernel_impl.h"

namespace turbomind::attention {

class Collector {
public:
    template<class T>
    void add()
    {
        kernels_.emplace_back(std::make_unique<KernelImpl<T>>());
        std::cout << "add kernel: " << to_string(kernels_.back()->desc()) << std::endl;
    }

    std::vector<std::unique_ptr<Kernel>> release()
    {
        return std::move(kernels_);
    }

private:
    std::vector<std::unique_ptr<Kernel>> kernels_;
};

using RegisterFn = std::function<void(Collector&)>;

inline std::vector<RegisterFn>& gKernelFactories()
{
    static std::vector<RegisterFn> v;
    return v;
}

struct Registrar {
    explicit Registrar(RegisterFn fn)
    {
        gKernelFactories().push_back(std::move(fn));
    }
};

}  // namespace turbomind::attention
