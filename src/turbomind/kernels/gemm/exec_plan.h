#pragma once

#include <utility>

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/output_spec.h"

namespace turbomind {
class LlamaLinear;
}

namespace turbomind::gemm {

class ExecPlan {
public:
    const OutputSpec& output_spec() const noexcept
    {
        return output_spec_;
    }

private:
    friend class Gemm;
    friend class ::turbomind::LlamaLinear;

    ExecPlan(GemmDesc desc, LaunchSpec launch): desc_{std::move(desc)}, launch_{launch} {}

    OutputSpec       output_spec_;
    const GemmDesc   desc_;
    const LaunchSpec launch_;
};

}  // namespace turbomind::gemm
