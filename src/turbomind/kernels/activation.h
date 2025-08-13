#pragma once

#include "src/turbomind/core/core.h"

namespace turbomind {

enum class ActivationType
{
    kSilu,
    kSiluGptOss
};

void Activation(Ref<Tensor> gate, const Tensor& up, ActivationType type, cudaStream_t stream);

}  // namespace turbomind
