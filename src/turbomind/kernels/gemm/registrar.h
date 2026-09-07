// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "src/turbomind/kernels/gemm/family.h"
#include "src/turbomind/kernels/gemm/kernel.h"

namespace turbomind::gemm {

class Collector {
public:
    explicit Collector(const Family& family): family_{family} {}

    template<class T, class... Args>
    void add(Args&&... args)
    {
        kernels_.emplace_back(std::make_unique<T>(family_, std::forward<Args>(args)...));
    }

    std::vector<std::unique_ptr<Kernel>> release()
    {
        return std::move(kernels_);
    }

private:
    const Family& family_;
    std::vector<std::unique_ptr<Kernel>> kernels_;
};

template<class T, class... Args>
void add(Collector& c, Args&&... args)
{
    c.add<T>(std::forward<Args>(args)...);
}

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
