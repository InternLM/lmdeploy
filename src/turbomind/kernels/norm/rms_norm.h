// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

void invokeRMSNorm(Tensor& out, const Tensor& x, const Tensor& w, float eps, cudaStream_t st);

void invokeRMSNormQK(Tensor& x, const Tensor& w, float eps, cudaStream_t st);

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

void ApplyBias(Tensor& x, const Tensor& bias, const Buffer_<int>& offsets, float scale, cudaStream_t st);

void ApplyBias(Tensor& x, const Tensor& bias, cudaStream_t st);

}  // namespace turbomind
