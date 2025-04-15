// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/core/tensor.h"

namespace turbomind {

void invokeRMSNorm(core::Tensor& out, const core::Tensor& x, const core::Tensor& w, float eps, cudaStream_t st);

void invokeRMSNormQK(core::Tensor& x, const core::Tensor& w, float eps, cudaStream_t st);

template<class T>
void invokeBiasResidualRMSNorm(
    T* residual, T* hidden_states, const T* weights, const T* bias, int dims, int num, float eps, cudaStream_t st);

void invokeResidualBiasRMSNorm(void*        hidden_states,
                               void*        residual,
                               const void*  weights,
                               const void*  bias,
                               DataType     dtype,
                               int          dims,
                               int          num,
                               float        eps,
                               cudaStream_t st);

}  // namespace turbomind
