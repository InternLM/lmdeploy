// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <istream>
#include <ostream>

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"

namespace turbomind {

class LlamaLinear {
public:
    explicit LlamaLinear();

    Tensor Forward(const Tensor&           input,  //
                   const LlamaDenseWeight& weight,
                   std::optional<Tensor>   output = {});

    Tensor Forward(const Tensor&           input,
                   const LlamaDenseWeight& weight,
                   const Buffer_<int>&     indices,
                   const Buffer_<int>&     offsets,
                   std::optional<Tensor>   output = {});

    void set_measure(bool measure);

    [[maybe_unused]] int Export(std::ostream& os);

    [[maybe_unused]] int Import(std::istream& is);

    std::vector<int> GetTuningSeq() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace turbomind
