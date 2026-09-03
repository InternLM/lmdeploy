// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <cuda_runtime.h>

#include "src/turbomind/kernels/gemm/exec_plan.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/weight_plan.h"

namespace turbomind {
class LlamaLinear;
}

namespace turbomind::gemm {

class Gemm {
public:
    static constexpr size_t kBarriersSize = 1 << 20;
    static constexpr size_t kPartialsSize = 32 << 20;

    Gemm();

    ~Gemm();

    struct Arguments {
        Operation    operation{};
        float        alpha{1.f};
        const void*  A{};
        MatrixLayout Adesc{};
        const void*  U{};
        MatrixLayout Udesc{};
        const void*  B{};
        MatrixLayout Bdesc{};
        const void*  V{};
        MatrixLayout Vdesc{};
        const void*  global_scale{};
        MatrixLayout global_scale_desc{};
        float        beta{};
        const void*  C{};
        MatrixLayout Cdesc{};
        void*        D{};
        MatrixLayout Ddesc{};
        void*        W{};
        MatrixLayout Wdesc{};
        Workspace    workspace{};
        cudaStream_t stream{};
    };

    std::optional<WeightPlan> GetWeightPlan(const WeightQuery& query) const;

    std::vector<DataType> DataTypes(const DataFormat& weight_format) const;

    [[nodiscard]] int Run(const ExecPlan& plan, const Arguments& args);

    [[maybe_unused]] int Export(std::ostream& os);

    [[maybe_unused]] int Import(std::istream& is);

    [[nodiscard]] std::vector<int> GetTuningSeq() const;

private:
    friend class ::turbomind::LlamaLinear;

    std::optional<ExecPlan> GetExecPlan(const Arguments& args);
    std::optional<ExecPlan> Tune(const Arguments& args);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind::gemm
