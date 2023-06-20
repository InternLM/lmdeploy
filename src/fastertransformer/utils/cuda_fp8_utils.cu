/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_fp8_utils.h"

namespace fastertransformer {
#ifdef ENABLE_FP8

template<typename T_OUT, typename T_IN, QUANTIZE_MODE quantize_mode>
__global__ void quantizeMatrix(T_OUT* output, float const* input_scale, T_IN const* input, uint32_t size, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        if (quantize_mode == QUANTIZE_MODE::PER_CHANNEL) {
            output[i] = T_OUT((float)(input[i]) * __ldg(input_scale + (i % n)));
        }
        else {
            output[i] = T_OUT((float)(input[i]) * __ldg(input_scale));
        }
    }
}

template<typename T_OUT, typename T_IN, QUANTIZE_MODE quantize_mode>
void invokeQuantizeMatrix(
    T_OUT* output, float const* input_scale, T_IN const* input, uint32_t size, uint32_t n, cudaStream_t stream)
{
    dim3 grid(32);
    dim3 block(256);
    quantizeMatrix<T_OUT, T_IN, quantize_mode><<<grid, block, 0, stream>>>(output, input_scale, input, size, n);
}

#define defineinvokeQuantizeMatrix(type_out, type_in, mode)                                                            \
    template void invokeQuantizeMatrix<type_out, type_in, mode>(type_out * output,                                     \
                                                                float const*   input_scale,                            \
                                                                type_in const* input,                                  \
                                                                uint32_t       size,                                   \
                                                                uint32_t       n,                                      \
                                                                cudaStream_t   stream);

defineinvokeQuantizeMatrix(__nv_fp8_e4m3, float, QUANTIZE_MODE::PER_CHANNEL);
defineinvokeQuantizeMatrix(__nv_fp8_e4m3, float, QUANTIZE_MODE::PER_TENSOR);
defineinvokeQuantizeMatrix(__nv_fp8_e4m3, half, QUANTIZE_MODE::PER_CHANNEL);
defineinvokeQuantizeMatrix(__nv_fp8_e4m3, half, QUANTIZE_MODE::PER_TENSOR);
defineinvokeQuantizeMatrix(half, __nv_fp8_e4m3, QUANTIZE_MODE::PER_CHANNEL);
defineinvokeQuantizeMatrix(half, __nv_fp8_e4m3, QUANTIZE_MODE::PER_TENSOR);
defineinvokeQuantizeMatrix(float, __nv_fp8_e4m3, QUANTIZE_MODE::PER_CHANNEL);
defineinvokeQuantizeMatrix(float, __nv_fp8_e4m3, QUANTIZE_MODE::PER_TENSOR);
#ifdef ENABLE_BF16
defineinvokeQuantizeMatrix(__nv_fp8_e4m3, __nv_bfloat16, QUANTIZE_MODE::PER_CHANNEL);
defineinvokeQuantizeMatrix(__nv_fp8_e4m3, __nv_bfloat16, QUANTIZE_MODE::PER_TENSOR);
defineinvokeQuantizeMatrix(__nv_bfloat16, __nv_fp8_e4m3, QUANTIZE_MODE::PER_CHANNEL);
defineinvokeQuantizeMatrix(__nv_bfloat16, __nv_fp8_e4m3, QUANTIZE_MODE::PER_TENSOR);
#endif

template<typename T_OUT, typename T_IN, typename T_FAKE>
__global__ void fakeQuantize(T_OUT* dst, const T_IN* src, const int size)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        T_FAKE tmp = (T_FAKE)((float)src[tid]);
        dst[tid]   = (T_OUT)((float)tmp);
    }
}

template<typename T_OUT, typename T_IN, typename T_FAKE>
void invokeFakeQuantize(T_OUT* dst, const T_IN* src, const int size, cudaStream_t stream)
{
    fakeQuantize<T_OUT, T_IN, T_FAKE><<<256, 256, 0, stream>>>(dst, src, size);
}

template void
invokeFakeQuantize<float, float, __nv_fp8_e4m3>(float* dst, const float* src, const int size, cudaStream_t stream);
template void
invokeFakeQuantize<half, half, __nv_fp8_e4m3>(half* dst, const half* src, const int size, cudaStream_t stream);
template void invokeFakeQuantize<__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3>(__nv_bfloat16*       dst,
                                                                              const __nv_bfloat16* src,
                                                                              const int            size,
                                                                              cudaStream_t         stream);

template<typename T_W>
__global__ void computeFP8QuantizeScale(float* quant_ptr, const T_W* weights, const int k, const int n)
{
    float max = -10000.f;
    for (int i = 0; i < k; i++) {
        float val = fabs((float)weights[i * n + blockIdx.x * blockDim.x + threadIdx.x]);
        max       = max > val ? max : val;
        if (threadIdx.x == 0 && blockIdx.x == 0 && i % 100 == 0) {
            printf("max: %f, val: %f \n", max, val);
        }
    }
    // quant_ptr[blockIdx.x * blockDim.x + threadIdx.x] = 1.0f;
    // quant_ptr[blockIdx.x * blockDim.x + threadIdx.x] = FP8_E4M3_MAX / max;
    quant_ptr[blockIdx.x * blockDim.x + threadIdx.x] = std::max(max / FP8_E4M3_MAX, 1.0f / 32.f);
}

template<typename T_W>
void invokeComputeFP8QuantizeScale(float* quant_ptr, const T_W* weights, const int k, const int n, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid;
    grid.x = (n + 255) / 256;
    computeFP8QuantizeScale<T_W><<<grid, block, 0, stream>>>(quant_ptr, weights, k, n);
}

#ifdef ENABLE_BF16
template void invokeComputeFP8QuantizeScale(
    float* quant_ptr, const __nv_bfloat16* weights, const int k, const int n, cudaStream_t stream);
#endif
template void
invokeComputeFP8QuantizeScale(float* quant_ptr, const float* weights, const int k, const int n, cudaStream_t stream);

#endif  // ENABLE_FP8
}  // namespace fastertransformer