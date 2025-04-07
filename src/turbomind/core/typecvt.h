#pragma once

#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind {

// ! avoid using these functions when possible, we need to unify the two data type classes

gemm::DataType to_gemm_dtype(DataType dtype);

DataType from_gemm_dtype(gemm::DataType dtype);

cudaDataType_t to_cuda_dtype(DataType dtype);

}  // namespace turbomind