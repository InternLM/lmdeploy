#include "ops.h"

#include <cstdint>
#include <cuda.h>
#include <vector>

__global__ void _scatter(int8_t* des, int8_t* buffer, int64_t length, int64_t* offset)
{
    int64_t buf_idx = blockIdx.x * length + threadIdx.x;
    int64_t src_idx = offset[blockIdx.x] + threadIdx.x;

    buffer[buf_idx] = des[src_idx];
}
namespace migration {
void scatter(int64_t des_ptr, int64_t buffer_ptr, int64_t length, int64_t offset_ptr, int64_t num_offset)
{
    _scatter<<<num_offset, length>>>((int8_t*)des_ptr, (int8_t*)buffer_ptr, length, (int64_t*)offset_ptr);
}
}  // namespace migration
