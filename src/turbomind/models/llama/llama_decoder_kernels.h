// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
void invokeFusedAddBiasResidualRMSNorm(
    T* residual, T* in_out, const T* bias, const T* scale, float eps, int batch_size, int n_dims, cudaStream_t stream);

}  // namespace turbomind
