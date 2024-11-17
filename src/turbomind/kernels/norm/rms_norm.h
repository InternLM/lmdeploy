// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

namespace turbomind {

template<class T>
void invokeRMSNorm(
    T* dst, int dst_ld, const T* src, int src_ld, const T* weights, int dims, int num, float eps, cudaStream_t st);

}