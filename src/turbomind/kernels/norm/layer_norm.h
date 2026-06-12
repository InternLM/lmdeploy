// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

void invokeLayerNorm(
    Tensor& out, const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, cudaStream_t stream);

void invokeResidualBiasLayerNorm(void*        hidden_states,
                                 void*        residual,
                                 const void*  norm_weight,
                                 const void*  norm_bias,
                                 const void*  residual_bias,
                                 DataType     dtype,
                                 int          dims,
                                 int          num,
                                 float        eps,
                                 cudaStream_t stream);

}  // namespace turbomind
