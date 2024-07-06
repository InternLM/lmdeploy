
#include "src/turbomind/kernels/core/data_type.h"
#include <cuda_runtime.h>

namespace turbomind {

void extend_to_u16(uint16_t* dst, const uint4_t* src, size_t n, cudaStream_t st = {});

void fuse_scales_and_zeros(half* fused, const half* scales, half* zeros, size_t n, cudaStream_t st = {});

}