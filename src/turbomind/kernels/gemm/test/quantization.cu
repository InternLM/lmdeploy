// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/test/quantization_impl.h"

namespace turbomind::gemm {

template void Quantize<uint4_t>(const thrust::universal_vector<half>& x,
                                int                                   m,
                                int                                   k,
                                Order                                 order,
                                int                                   group_size,
                                thrust::universal_vector<half>&       x_p,  // pseudo-quantized
                                thrust::universal_vector<uint16_t>&   x_q,  // quantized ushort
                                thrust::universal_vector<half>&       x_u,  // scales & zeros (always m-major)
                                cudaStream_t                          stream);

template void Quantize<uint8_t>(const thrust::universal_vector<half>& x,
                                int                                   m,
                                int                                   k,
                                Order                                 order,
                                int                                   group_size,
                                thrust::universal_vector<half>&       x_p,  // pseudo-quantized
                                thrust::universal_vector<uint16_t>&   x_q,  // quantized ushort
                                thrust::universal_vector<half>&       x_u,  // scales & zeros (always m-major)
                                cudaStream_t                          stream);

}  // namespace turbomind::gemm
