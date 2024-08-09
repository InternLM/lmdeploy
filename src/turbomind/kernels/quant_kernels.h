
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
void invokeI8Quant(
    const T* input, int8_t* out, float* scale, const int token_num, const int hidden_size, cudaStream_t stream);

}  // namespace turbomind
