#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/kernels/linear_attn/delta_rule.h"

namespace turbomind::linear_attn::delta_rule {

class GdrKernel {
public:
    virtual ~GdrKernel() = default;

    virtual const GdrKernelSpec& spec() const noexcept = 0;
    virtual const char* name() const noexcept = 0;
    virtual bool Match(const Operation&, const PlanningContext&) const = 0;
    virtual bool Plan(const Operation&, const PlanningContext&, delta_rule::Plan*) const = 0;
    virtual void PrepareState(const core::Tensor&,
                              core::Tensor&,
                              int,
                              int,
                              const delta_rule::Plan&,
                              cudaStream_t) const
    {
    }
    virtual void Run(const Arguments&, const delta_rule::Plan&, cudaStream_t) const = 0;
};

class GdrKernelRegistry {
public:
    bool Add(std::unique_ptr<GdrKernel> kernel);
    const GdrKernel* Find(const Operation&, const PlanningContext&) const;
    static GdrKernelRegistry& instance();

private:
    std::vector<std::unique_ptr<GdrKernel>> kernels_;
};

}  // namespace turbomind::linear_attn::delta_rule
