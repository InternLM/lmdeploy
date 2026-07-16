#include "src/turbomind/kernels/linear_attn/kernel/tma_desc.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda_runtime.h>

namespace turbomind::linear_attn::delta_rule {
namespace {

#if CUDA_VERSION >= 13000
using CuTensorMapEncodeTiledFn = PFN_cuTensorMapEncodeTiled_v12000;
#else
using CuTensorMapEncodeTiledFn = PFN_cuTensorMapEncodeTiled;
#endif

CuTensorMapEncodeTiledFn GetCuTensorMapEncodeTiled()
{
    static const auto ptr = [] {
        cudaDriverEntryPointQueryResult driver_status;
        void*                           raw_ptr = nullptr;
#if CUDA_VERSION >= 13000
        TM_CUDA_CHECK(cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled", &raw_ptr, 12000, cudaEnableDefault, &driver_status));
#else
        TM_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &raw_ptr, cudaEnableDefault, &driver_status));
#endif
        TM_CHECK_EQ(driver_status, cudaDriverEntryPointSuccess);
        return reinterpret_cast<CuTensorMapEncodeTiledFn>(raw_ptr);
    }();
    return ptr;
}

}  // namespace

CUtensorMap MakeTmaDesc(void*              global_address,
                        CUtensorMapDataType data_type,
                        uint32_t            rank,
                        const uint64_t*     global_dim,
                        const uint64_t*     global_stride,
                        const uint32_t*     box_dim,
                        CUtensorMapSwizzle  swizzle)
{
    uint32_t    element_stride[5] = {1, 1, 1, 1, 1};
    CUtensorMap tensor_map{};
    TM_CHECK_EQ(GetCuTensorMapEncodeTiled()(&tensor_map,
                                            data_type,
                                            rank,
                                            global_address,
                                            global_dim,
                                            global_stride,
                                            box_dim,
                                            element_stride,
                                            CU_TENSOR_MAP_INTERLEAVE_NONE,
                                            swizzle,
                                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                CUDA_SUCCESS);
    return tensor_map;
}

}  // namespace turbomind::linear_attn::delta_rule
