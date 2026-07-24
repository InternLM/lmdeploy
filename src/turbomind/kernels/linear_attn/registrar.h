#pragma once

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "src/turbomind/kernels/linear_attn/registry.h"

namespace turbomind::linear_attn::delta_rule {

class Collector {
public:
    template<class T>
    void add()
    {
        kernels_.emplace_back(std::make_unique<T>());
    }

    std::vector<std::unique_ptr<GdrKernel>> release()
    {
        return std::move(kernels_);
    }

private:
    std::vector<std::unique_ptr<GdrKernel>> kernels_;
};

using RegisterFn = std::function<void(Collector&)>;

inline std::vector<RegisterFn>& gKernelFactories()
{
    static std::vector<RegisterFn> factories;
    return factories;
}

struct Registrar {
    explicit Registrar(RegisterFn fn)
    {
        gKernelFactories().push_back(std::move(fn));
    }
};

}  // namespace turbomind::linear_attn::delta_rule
