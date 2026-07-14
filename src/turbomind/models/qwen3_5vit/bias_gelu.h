#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/kernels/activation.h"

#include <cuda_runtime.h>

namespace turbomind {

// In-place Qwen3.5 ViT bias + unary activation:
// x <- activation(x + bias)
void invokeQwen3_5VitBiasActivation(Tensor& x, const Tensor& bias, ActivationType type, cudaStream_t stream);

}  // namespace turbomind
