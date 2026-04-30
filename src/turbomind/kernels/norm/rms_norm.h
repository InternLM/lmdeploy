// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

[[nodiscard]] cudaError_t invokeRMSNorm(Tensor& out, const Tensor& x, const Tensor& w, float eps, cudaStream_t st);

[[nodiscard]] cudaError_t invokeRMSNormQK(Tensor& x, const Tensor& w, float eps, cudaStream_t st);

template<class T>
[[nodiscard]] cudaError_t invokeBiasResidualRMSNorm(
    T* residual, T* hidden_states, const T* weights, const T* bias, int dims, int num, float eps, cudaStream_t st);

[[nodiscard]] cudaError_t invokeResidualBiasRMSNorm(void*        hidden_states,
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
