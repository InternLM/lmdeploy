#include "src/turbomind/core/typecvt.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind {

using gemm::get_data_type_v;

gemm::DataType to_gemm_dtype(DataType dtype)
{
    switch (dtype) {
        case TYPE_FP16:
            return get_data_type_v<half>;
        case TYPE_BF16:
            return get_data_type_v<nv_bfloat16>;
        case TYPE_FP32:
            return get_data_type_v<float>;
        case TYPE_UINT4:
            return get_data_type_v<uint4_t>;
        case TYPE_UINT8:
            return get_data_type_v<uint8_t>;
        default:
            TM_CHECK(0) << "not implemented";
    }
    return {};
}

DataType from_gemm_dtype(gemm::DataType dtype)
{
    switch (dtype) {
        case get_data_type_v<half>:
            return TYPE_FP16;
        case get_data_type_v<nv_bfloat16>:
            return TYPE_BF16;
        case get_data_type_v<float>:
            return TYPE_FP32;
        case get_data_type_v<uint4_t>:
            return TYPE_UINT4;
        case get_data_type_v<uint8_t>:
            return TYPE_UINT8;
        default:
            TM_CHECK(0) << "not implemented";
    }
    return {};
}

cudaDataType_t to_cuda_dtype(DataType dtype)
{
    switch (dtype) {
        case TYPE_FP16:
            return CUDA_R_16F;
        case TYPE_BF16:
            return CUDA_R_16BF;
        case TYPE_FP32:
            return CUDA_R_32F;
        default:
            TM_CHECK(0) << "not implemented";
    }
    return {};
}

}  // namespace turbomind