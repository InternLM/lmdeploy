// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::gemm {

template<class T, class Tb>
void invoke(T* C, const T* A, const Tb* B, const T* Q, int m, int n, int k, cudaStream_t st);

}