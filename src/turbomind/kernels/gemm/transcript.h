// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind::gemm {

template<class T, class Ti, class To>
void transcript(To* dst, const Ti* src, int n, int k, cudaStream_t st);

}