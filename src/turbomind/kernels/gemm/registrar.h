// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/turbomind/kernels/gemm/family.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"

namespace turbomind::gemm {

class Collector {
public:
    explicit Collector(const Family& family): family_{family} {}

    // Matches Registry::Add<Config>(): Config has nested ::Kernel
    template<class T, class... Args>
    void add(Args&&... args)
    {
        if constexpr (std::is_base_of_v<Kernel, T>) {
            kernels_.emplace_back(std::make_unique<T>(family_, std::forward<Args>(args)...));
        }
        else {
            static_assert(sizeof...(Args) == 0);
            kernels_.emplace_back(std::make_unique<KernelImpl<typename T::Kernel>>(family_));
        }
    }

    std::vector<std::unique_ptr<Kernel>> release()
    {
        return std::move(kernels_);
    }

private:
    const Family&                        family_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
};

using RegisterFn = std::function<void(Collector&)>;

inline std::vector<std::pair<const Family*, RegisterFn>>& gKernelFactories()
{
    static std::vector<std::pair<const Family*, RegisterFn>> v;
    return v;
}

struct Registrar {
    Registrar(const Family& family, RegisterFn fn)
    {
        gKernelFactories().emplace_back(&family, std::move(fn));
    }
};

}  // namespace turbomind::gemm
