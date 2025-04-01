#include "ops.h"

#include <cstdint>
#include <cuda.h>

__global__ void _scatter(int8_t* des, int8_t* buffer, int64_t length, int64_t* offset)
{
    int64_t iter = (length + blockDim.x - 1) / blockDim.x;

    for (int i = 0; i < iter; i++) {
        if (blockDim.x * i + threadIdx.x < length) {
            int64_t buf_idx = blockIdx.x * length + blockDim.x * i + threadIdx.x;
            int64_t des_idx = offset[blockIdx.x] + blockDim.x * i + threadIdx.x;
            des[des_idx] = buffer[buf_idx];
        }
    }
}

namespace slime {
void scatter(int64_t des_ptr, int64_t buffer_ptr, int64_t length, int64_t offset_ptr, int64_t num_offset)
{
    _scatter<<<num_offset, 512>>>((int8_t*)des_ptr, (int8_t*)buffer_ptr, length, (int64_t*)offset_ptr);
}
}  // namespace slime
