// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "cta_map.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/thread_map.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>

namespace turbomind::attention {

template<int HeadDim, class T>
void invokeReduceV3(T*           out,
                    float*       partial_ML,
                    float*       partial_O,
                    const int*   split_cnt,
                    int          partial_len,
                    int          max_split_cnt,
                    int          cp_size,
                    int          cp_rank,
                    int          query_num,
                    int          head_num,
                    float        exp_scale,
                    cudaStream_t stream);
}  // namespace turbomind::attention
