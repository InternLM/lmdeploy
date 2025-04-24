#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

CUtensorMap make_2d_tma_desc(void*              global_address,
                             DataType           data_type,
                             uint32_t           gmem_rows,
                             uint32_t           gmem_cols,
                             uint32_t           smem_rows,
                             uint32_t           smem_cols,
                             Order              order,
                             CUtensorMapSwizzle swizzle);

}  // namespace turbomind::gemm