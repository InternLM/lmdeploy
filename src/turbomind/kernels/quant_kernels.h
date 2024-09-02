
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
void invokeI8Quant(
    const T* input, int8_t* out, float* scale, const int token_num, int hidden_size, int stride, cudaStream_t stream);

}  // namespace turbomind
