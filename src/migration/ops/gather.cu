#include "ops.h"

#include <cstdint>
#include <cuda.h>
#include <vector>

__global__ void _gather(int8_t* src, int8_t* buffer, int64_t length, int64_t* offset)
{
    int64_t buf_idx = blockIdx.x * length + threadIdx.x;
    int64_t src_idx = offset[blockIdx.x] + threadIdx.x;

    buffer[buf_idx] = src[src_idx];
}

namespace migration {
void gather(int64_t src_ptr, int64_t buffer_ptr, int64_t length, int64_t offset_ptr, int64_t num_offset)
{
    _gather<<<num_offset, length>>>((int8_t*)src_ptr, (int8_t*)buffer_ptr, length, (int64_t*)offset_ptr);
}
}  // namespace migration
