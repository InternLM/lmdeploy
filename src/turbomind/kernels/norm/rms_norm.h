// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "src/turbomind/utils/Tensor.h"

namespace turbomind {

template<class T>
void invokeRMSNorm(
    T* dst, int dst_ld, const T* src, int src_ld, const T* weights, int dims, int num, float eps, cudaStream_t st);

template<class T>
void invokeRMSNorm(T* dst, const T* src, const T* weights, int dims, int num, float eps, cudaStream_t st)
{
    invokeRMSNorm(dst, dims, src, dims, weights, dims, num, eps, st);
}

void invokeQkRMSNorm(void*        data,
                     int          ld,
                     const void*  weight,
                     DataType     dtype,
                     int          head_dim,
                     int          n,
                     int          token_num,
                     float        eps,
                     cudaStream_t stream);

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
