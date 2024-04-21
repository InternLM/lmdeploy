// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::gemm {

template<class T, class Ti, class To>
void transcript(To* dst_B, T* dst_Q, const Ti* src_B, const T* src_Q, int n, int k, int g, cudaStream_t st);

}