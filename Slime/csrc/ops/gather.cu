#include "ops.h"

#include <cstdint>
#include <cuda.h>
#include <vector>

__global__ void _gather(int8_t* src, int8_t* buffer, int64_t length, int64_t* offset)
{
    int64_t iter = (length + blockDim.x - 1) / blockDim.x;

    for (int i = 0; i < iter; i++) {
        if (blockDim.x * i + threadIdx.x < length) {
            int64_t buf_idx = blockIdx.x * length + blockDim.x * i + threadIdx.x;
            int64_t src_idx = offset[blockIdx.x] + blockDim.x * i + threadIdx.x;
            buffer[buf_idx] = src[src_idx];
        }
    }
}

namespace slime {
void gather(int64_t src_ptr, int64_t buffer_ptr, int64_t length, int64_t offset_ptr, int64_t num_offset)
{
    _gather<<<num_offset, 512>>>((int8_t*)src_ptr, (int8_t*)buffer_ptr, length, (int64_t*)offset_ptr);
}
}  // namespace slime
