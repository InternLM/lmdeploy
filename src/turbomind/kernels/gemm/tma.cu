
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/cuda_data_type.h"
#include "src/turbomind/kernels/gemm/tma.h"

namespace turbomind::gemm {

#if __CUDACC_VER_MAJOR__ >= 12

#if (CUDA_VERSION >= 13000) && (!defined(PFN_cuTensorMapEncodeTiled))
// PFN_cuTensorMapEncodeTiled not defined in cuda 13 headers.
#define PFN_cuTensorMapEncodeTiled PFN_cuTensorMapEncodeTiled_v12000
#endif

namespace {

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled()
{
    static const auto ptr = [] {
        // Get pointer to `cuTensorMapEncodeTiled`
        cudaDriverEntryPointQueryResult driver_status;
        void*                           cuTensorMapEncodeTiled_ptr = nullptr;

// https://github.com/NVIDIA/cutlass/pull/2086
#if CUDA_VERSION >= 13000
        cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status);
#else
        cudaGetDriverEntryPoint(
            "cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, cudaEnableDefault, &driver_status);
#endif
        TM_CHECK_EQ(driver_status, cudaDriverEntryPointSuccess);
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(cuTensorMapEncodeTiled_ptr);
    }();
    return ptr;
}

CUtensorMap make_2d_tma_desc(void*              global_address,
                             DataType           data_type,
                             uint64_t           gmem_dims[2],
                             uint64_t           stride_in_bytes,
                             uint32_t           smem_dims[2],
                             CUtensorMapSwizzle swizzle)
{
    uint64_t global_stride[1] = {stride_in_bytes};
    uint32_t elem_strides[2]  = {1, 1};

    auto encode_func = get_cuTensorMapEncodeTiled();

    CUtensorMap tensor_map = {};

    auto result = encode_func(&tensor_map,
                              to_CUtensorMap_dtype(data_type),
                              2,
                              global_address,
                              gmem_dims,
                              global_stride,
                              smem_dims,
                              elem_strides,
                              CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                              swizzle,
                              CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                              CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    TM_CHECK_EQ(result, CUDA_SUCCESS);

    return tensor_map;
}

}  // namespace

CUtensorMap make_2d_tma_desc(void*              global_address,
                             DataType           data_type,
                             uint32_t           gmem_rows,
                             uint32_t           gmem_cols,
                             uint32_t           smem_rows,
                             uint32_t           smem_cols,
                             Order              order,
                             CUtensorMapSwizzle swizzle,
                             int                ld)
{
    if (order == kRowMajor) {
        uint64_t gmem_dims[] = {gmem_cols, gmem_rows};
        uint32_t smem_dims[] = {smem_cols, smem_rows};
        return make_2d_tma_desc(
            global_address, data_type, gmem_dims, byte_size(data_type, ld ? ld : gmem_cols), smem_dims, swizzle);
    }
    else {
        uint64_t gmem_dims[] = {gmem_rows, gmem_cols};
        uint32_t smem_dims[] = {smem_rows, smem_cols};
        return make_2d_tma_desc(
            global_address, data_type, gmem_dims, byte_size(data_type, ld ? ld : gmem_rows), smem_dims, swizzle);
    }
}

CUtensorMap make_2d_tma_desc(void* ptr, const MatrixLayout& desc, uint2 smem_shape, CUtensorMapSwizzle swizzle)
{
    return make_2d_tma_desc(
        ptr, desc.type, desc.rows, desc.cols, smem_shape.x, smem_shape.y, desc.order, swizzle, desc.ld);
}

#endif

}  // namespace turbomind::gemm
