// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::gemm {

template<class T>
void invoke(T* C, const T* A, const T* B, int m, int n, int k, cudaStream_t st);

}