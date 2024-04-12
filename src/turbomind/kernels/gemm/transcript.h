// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::gemm {

template<class T, class Tb>
void transcript(Tb* dst, const T* src, int n, int k, cudaStream_t st);

}