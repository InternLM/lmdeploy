// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"

namespace turbomind::gemm {

class Collector {
public:
    // Matches Registry::Add<Config>(): Config has nested ::Kernel
    template<class Config>
    void add()
    {
        kernels_.emplace_back(std::make_unique<KernelImpl<typename Config::Kernel>>());
    }

    void add(std::unique_ptr<Kernel> kernel)
    {
        kernels_.emplace_back(std::move(kernel));
    }

    std::vector<std::unique_ptr<Kernel>> release()
    {
        return std::move(kernels_);
    }

private:
    std::vector<std::unique_ptr<Kernel>> kernels_;
};

using RegisterFn = std::function<void(Collector&, int arch)>;

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

}  // namespace turbomind::gemm
