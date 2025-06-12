#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

#if __CUDACC_VER_MAJOR__ >= 12

CUtensorMap make_2d_tma_desc(void*              global_address,
                             DataType           data_type,
                             uint32_t           gmem_rows,
                             uint32_t           gmem_cols,
                             uint32_t           smem_rows,
                             uint32_t           smem_cols,
                             Order              order,
                             CUtensorMapSwizzle swizzle,
                             int                ld = 0);

CUtensorMap make_2d_tma_desc(void* ptr, const MatrixLayout& desc, uint2 smem_shape, CUtensorMapSwizzle swizzle);

constexpr CUtensorMapSwizzle get_tma_swizzle(int bytes)
{
    switch (bytes) {
        case 128:
            return CU_TENSOR_MAP_SWIZZLE_128B;
        case 64:
            return CU_TENSOR_MAP_SWIZZLE_64B;
        case 32:
            return CU_TENSOR_MAP_SWIZZLE_32B;
        case 16:  // unit swizzle is equivalent to "none"
        case 0:
            return CU_TENSOR_MAP_SWIZZLE_NONE;
        default:
            throw std::logic_error("unsupported swizzle type: " + std::to_string(bytes));
    }
    return {};
}

#endif

}  // namespace turbomind::gemm
