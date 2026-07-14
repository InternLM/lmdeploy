

#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "src/turbomind/core/data_type.h"

namespace turbomind {

// clang-format off

constexpr cudaDataType to_cuda_dtype(DataType type)
{
    switch (type) {
        case kUint8:  return CUDA_R_8U;
        case kUint16: return CUDA_R_16U;
        case kUint32: return CUDA_R_32U;
        case kUint64: return CUDA_R_64U;
        case kInt8:  return CUDA_R_8I;
        case kInt16: return CUDA_R_16I;
        case kInt32: return CUDA_R_32I;
        case kInt64: return CUDA_R_64I;
        case kFloat16: return CUDA_R_16F;
        case kFloat32: return CUDA_R_32F;
        case kFloat64: return CUDA_R_64F;
        case kBfloat16: return CUDA_R_16BF;
        case kFloat8_e4m3: return CUDA_R_8F_E4M3;
        case kFloat8_e5m2: return CUDA_R_8F_E5M2;
        default:
            throw std::runtime_error("Not supported " + std::string{to_string(type)});
    }
}

constexpr DataType from_cuda_dtype(cudaDataType type) {
    switch (type) {
        case CUDA_R_8U:  return kUint8;
        case CUDA_R_16U: return kUint16;
        case CUDA_R_32U: return kUint32;
        case CUDA_R_64U: return kUint64;
        case CUDA_R_8I:  return kInt8;
        case CUDA_R_16I: return kInt16;
        case CUDA_R_32I: return kInt32;
        case CUDA_R_64I: return kInt64;
        case CUDA_R_16F: return kFloat16;
        case CUDA_R_32F: return kFloat32;
        case CUDA_R_64F: return kFloat64;
        case CUDA_R_16BF: return kBfloat16;
        case CUDA_R_8F_E4M3: return kFloat8_e4m3;
        case CUDA_R_8F_E5M2: return kFloat8_e5m2;
        default:
            throw std::runtime_error("Not supported " + std::string{std::to_string(type)});
    }
}

#if __CUDACC_VER_MAJOR__ >= 12

constexpr CUtensorMapDataType to_CUtensorMap_dtype(DataType type) {
    switch (type) {
        case kFloat32:
            return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case kFloat16:
            return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
        case kBfloat16:
            return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        case kFloat8_e4m3:
        case kFloat8_e5m2:
            return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        default:
            throw std::runtime_error("Not supported " + std::string{to_string(type)});
    }
}

#endif

// clang-format on

}  // namespace turbomind
