// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <istream>
#include <optional>
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

    // Expert-parallel variant: `scales` carries externally-computed FP8 input
    // scales (from EP dispatch) so the input is not re-quantized here.
    void Forward(const Tensor&                input,
                 const std::optional<Tensor>& scales,
                 const LinearWeight&          weight,
                 const Buffer_<int>&          indices,
                 const Buffer_<int>&          offsets,
                 Ref<Tensor>                  output);

    void set_measure(bool measure);

    [[maybe_unused]] int Export(std::ostream& os);

    [[maybe_unused]] int Import(std::istream& is);

    std::vector<int> GetTuningSeq() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace turbomind
