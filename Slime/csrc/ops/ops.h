#include <cstdint>
#include <cuda.h>

namespace slime {
void gather(int64_t src_ptr, int64_t buffer_ptr, int64_t length, int64_t offset_ptr, int64_t num_offset);
void scatter(int64_t des_ptr, int64_t buffer_ptr, int64_t length, int64_t offset_ptr, int64_t num_offset);
}  // namespace slime
