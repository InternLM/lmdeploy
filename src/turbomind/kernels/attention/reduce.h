// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>

namespace turbomind::attention {

template<int HeadDim, class T>
void invokeReduce(T*           out,
                  float*       partial_M,
                  float*       partial_L,
                  float*       partial_O,
                  const int*   split_cnt,
                  int          partial_len,
                  int          max_split_cnt,
                  int          query_num,
                  int          head_num,
                  float        exp_scale,
                  cudaStream_t stream);

}  // namespace turbomind::attention
