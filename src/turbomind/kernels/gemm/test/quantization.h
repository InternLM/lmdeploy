// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/types.h"
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#pragma once

namespace turbomind::gemm {

template<class D, class S>
void Quantize(const thrust::universal_vector<S>&  x,
              int                                 m,
              int                                 k,
              Order                               order,
              int                                 group_size,
              thrust::universal_vector<S>&        x_p,  // pseudo-quantized
              thrust::universal_vector<uint16_t>& x_q,  // quantized ushort
              thrust::universal_vector<S>&        x_u,  // scales & zeros (always m-major)
              cudaStream_t                        stream);

}
