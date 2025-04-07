// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <istream>
#include <ostream>

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/utils/cublasMMWrapper.h"

namespace turbomind {

class LlamaLinear {
public:
    enum Type
    {
        kGemm,
        kFusedSiluFfn,
        kFusedAdd
    };

    LlamaLinear(cublasMMWrapper* cublas_wrapper, cudaStream_t stream);

    core::Tensor forward(const core::Tensor&     input,  //
                         const LlamaDenseWeight& weight,
                         Type                    type   = kGemm,
                         core::Tensor*           output = {});

    void forward_moe(core::Tensor&           output,
                     const core::Tensor&     input,
                     const int*              indexes,
                     const int*              offsets,
                     const LlamaDenseWeight& weight,
                     Type                    type,
                     gemm::Context*          context);

    void set_measure(bool measure);

    [[maybe_unused]] int Export(std::ostream& os);

    [[maybe_unused]] int Import(std::istream& is);

    std::vector<int> GetTuningSeq() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace turbomind
