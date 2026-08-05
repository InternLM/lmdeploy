// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <istream>
#include <ostream>

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind {

class LlamaLinear {
public:
    explicit LlamaLinear();

    void Forward(const Tensor&       input,  //
                 const LinearWeight& weight,
                 Ref<Tensor>         output);

    void Forward(const Tensor&       input,
                 const LinearWeight& weight,
                 const Buffer_<int>& indices,
                 const Buffer_<int>& offsets,
                 Ref<Tensor>         output);

    /// Forward with optional dynamic act-scale companions.
    /// ``input_scales``: when ``input`` is already FP8, used as GEMM U (skip QuantizeSymm).
    /// ``output_scales``: when ``weight.output_dtype()`` is FP8, filled with group-128 scales (W).
    void Forward(const Tensor&       input,
                 const Tensor&       input_scales,
                 const LinearWeight& weight,
                 const Buffer_<int>& indices,
                 const Buffer_<int>& offsets,
                 Ref<Tensor>         output,
                 Ref<Tensor>         output_scales);

    void Forward(const Tensor&       input,
                 const Tensor&       input_scales,
                 const LinearWeight& weight,
                 Ref<Tensor>         output,
                 Ref<Tensor>         output_scales);

    void set_measure(bool measure);

    [[maybe_unused]] int Export(std::ostream& os);

    [[maybe_unused]] int Import(std::istream& is);

    std::vector<int> GetTuningSeq() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace turbomind
