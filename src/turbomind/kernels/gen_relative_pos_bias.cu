/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "cublas_v2.h"
#include "gen_relative_pos_bias.h"
#include "reduce_kernel_utils.cuh"
#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <cstdio>

namespace turbomind {

/*******************  invokeGenRelativePosBias  ***********************/
// relative_position_bias_table is [(2*window_size-1)*(2*window_size-1), headNum]
// relative_position_bias is [head_num, window_size^2, window_size^2]
// grid(window_size*window_size, head_num)
// block(window_size*window_size)

template<typename T, typename Tindex>
__global__ void gen_relative_pos_bias(T*            relative_position_bias,
                                      const T*      relative_position_bias_table,
                                      const Tindex* relative_position_bias_index,
                                      const int     window_size,
                                      const int     head_num)
{
    const int    h_in_window           = blockIdx.x / window_size;
    const int    w_in_window           = blockIdx.x % window_size;
    const int    h_in_token            = threadIdx.x / window_size;
    const int    w_in_token            = threadIdx.x % window_size;
    const int    head_idx              = blockIdx.y;
    const int    elements_per_window   = window_size * window_size;
    const size_t elements_per_window_2 = elements_per_window * elements_per_window;
    const size_t output_idx = head_idx * elements_per_window_2 + blockIdx.x * elements_per_window + threadIdx.x;
    if (output_idx < head_num * elements_per_window_2) {
        const Tindex idx_in_table =
            relative_position_bias_index[(h_in_window * window_size + w_in_window) * elements_per_window
                                         + h_in_token * window_size + w_in_token];
        relative_position_bias[output_idx] = relative_position_bias_table[idx_in_table * head_num + head_idx];
    }
}

template<typename T, typename Tindex>
void invokeGenRelativePosBias(T*            relative_position_bias,
                              const T*      relative_position_bias_table,
                              const Tindex* relative_position_bias_index,
                              const int     window_size,
                              const int     head_num,
                              cudaStream_t  stream)
{
    dim3 grid(window_size * window_size, head_num);
    dim3 block(window_size * window_size);

    if (block.x > 1024) {
        printf("[ERROR][invokeGenRelativePosBias] window_size*window_size > 1024.\n");
        exit(-1);
    }

    gen_relative_pos_bias<<<grid, block, 0, stream>>>(
        relative_position_bias, relative_position_bias_table, relative_position_bias_index, window_size, head_num);
}

/*******************  invokeGenRelativePosBiasV2  ***********************/
template<typename T, typename Tindex>
void invokeGenRelativePosBiasV2(T*            relative_position_bias,
                                const T*      relative_coords_table,
                                const Tindex* relative_position_bias_index,
                                const T*      cpb_mlp_weight1,
                                const T*      cpb_mlp_bias1,
                                const T*      cpb_mlp_weight2,
                                const int     window_size,
                                const int     cpb_mlp_in_dim,
                                const int     cpb_mlp_out_dim,
                                const int     head_num,
                                cudaStream_t  stream)
{

    dim3 grid(window_size * window_size, head_num);
    dim3 block(window_size * window_size);

    if (block.x > 1024) {
        printf("[ERROR][invokeGenRelativePosBias] window_size*window_size > 1024.\n");
        exit(-1);
    }

    T* relative_position_bias_table;
    check_cuda_error(cudaMalloc(&relative_position_bias_table,
                                ((2 * window_size - 1) * (2 * window_size - 1) * head_num) * sizeof(T)));
    T* cpb_mlp_1;
    check_cuda_error(
        cudaMalloc(&cpb_mlp_1, ((2 * window_size - 1) * (2 * window_size - 1) * cpb_mlp_out_dim) * sizeof(T)));
    cublasHandle_t cublas_handle;
    check_cuda_error(cublasCreate(&cublas_handle));

    int            m     = (2 * window_size - 1) * (2 * window_size - 1);
    T              alpha = (T)1.0f;
    T              beta  = (T)0.0f;
    cudaDataType_t type  = std::is_same<float, T>::value ? CUDA_R_32F : CUDA_R_16F;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t compute_type = std::is_same<float, T>::value ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F;
#else
    cudaDataType_t compute_type = std::is_same<float, T>::value ? CUDA_R_32F : CUDA_R_16F;
#endif
    cublasGemmAlgo_t algo = std::is_same<float, T>::value ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    check_cuda_error(cublasGemmEx(cublas_handle,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  cpb_mlp_out_dim,
                                  m,
                                  cpb_mlp_in_dim,
                                  &alpha,
                                  cpb_mlp_weight1,
                                  type,
                                  cpb_mlp_in_dim,
                                  relative_coords_table,
                                  type,
                                  cpb_mlp_in_dim,
                                  &beta,
                                  cpb_mlp_1,
                                  type,
                                  cpb_mlp_out_dim,
                                  compute_type,
                                  algo));

    invokeGenericActivation<ReluActivation, T, T>(
        cpb_mlp_1, cpb_mlp_bias1, nullptr, nullptr, nullptr, nullptr, m, cpb_mlp_out_dim, 0, nullptr, nullptr, stream);

    check_cuda_error(cublasGemmEx(cublas_handle,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  head_num,
                                  m,
                                  cpb_mlp_out_dim,
                                  &alpha,
                                  cpb_mlp_weight2,
                                  type,
                                  cpb_mlp_out_dim,
                                  cpb_mlp_1,
                                  type,
                                  cpb_mlp_out_dim,
                                  &beta,
                                  relative_position_bias_table,
                                  type,
                                  head_num,
                                  compute_type,
                                  algo));

    gen_relative_pos_bias<<<grid, block, 0, stream>>>(
        relative_position_bias, relative_position_bias_table, relative_position_bias_index, window_size, head_num);

    invokeSigmoid(
        relative_position_bias, window_size * window_size * window_size * window_size * head_num, 16.0f, stream);
    check_cuda_error(cudaFree(relative_position_bias_table));
    check_cuda_error(cudaFree(cpb_mlp_1));
    check_cuda_error(cublasDestroy(cublas_handle));
}

/*******************  instantiation  ***********************/

template void invokeGenRelativePosBias(float*       relative_position_bias,
                                       const float* relative_position_bias_table,
                                       const int*   relative_position_bias_index,
                                       const int    window_size,
                                       const int    head_num,
                                       cudaStream_t stream);

template void invokeGenRelativePosBias(half*        relative_position_bias,
                                       const half*  relative_position_bias_table,
                                       const int*   relative_position_bias_index,
                                       const int    window_size,
                                       const int    head_num,
                                       cudaStream_t stream);

template void invokeGenRelativePosBias(float*         relative_position_bias,
                                       const float*   relative_position_bias_table,
                                       const int64_t* relative_position_bias_index,
                                       const int      window_size,
                                       const int      head_num,
                                       cudaStream_t   stream);

template void invokeGenRelativePosBias(half*          relative_position_bias,
                                       const half*    relative_position_bias_table,
                                       const int64_t* relative_position_bias_index,
                                       const int      window_size,
                                       const int      head_num,
                                       cudaStream_t   stream);

__host__ __device__ uint32_t pow2_rounddown(uint32_t x)
{
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x >>= 1;
    return x + 1;
}

template<typename T>
__global__ void generate_alibi_slopes(T* alibi_slopes, const size_t num_heads)
{
    if (threadIdx.x < num_heads) {
        // The nearest power of 2 greater than num_heads followed by HF's implementation.
        int num_heads_pow2 = pow2_rounddown(num_heads);
        // Loop over the attention head.
        for (int h = threadIdx.x; h < num_heads; h += blockDim.x) {
            if (h < num_heads_pow2) {
                alibi_slopes[h] = static_cast<T>(powf(powf(0.5f, powf(0.5f, log2f(num_heads_pow2) - 3.f)), h + 1));
            }
            else {
                alibi_slopes[h] = static_cast<T>(
                    powf(powf(0.5f, powf(0.5f, log2f(num_heads_pow2 << 1) - 3.f)), (h - num_heads_pow2) * 2 + 1));
            }
        }
    }
}

template<typename T>
void invokeBuildAlibiSlopes(T* alibi_slopes, const size_t num_heads, cudaStream_t stream)
{
    // Generate the slopes of a linear attention linear bias.
    //
    // Paper: https://arxiv.org/abs/2108.12409
    // HF's implementation
    //   https://github.com/huggingface/transformers/blob/56ef0ba44765162f830873c140bd40bdc975cc34/src/transformers/models/bloom/modeling_bloom.py#L86
    // Author's implementation
    //   https://github.com/ofirpress/attention_with_linear_biases/blob/02aa87e7a29e9340efd28d6d169018eafb3aa57a/fairseq/models/transformer.py#L760
    //
    // alibi_slopes: [num_heads],
    //     strictly follows how HF implements. which treats power-of-2 heads, and non-power-of-2 heads differently.
    //     what paper generates differs with HF's when number of heads is not a power of 2.
    // num_heads: the number of attention heads.
    // stream: a cuda stream.

    dim3 block(min((int)num_heads, 512));
    generate_alibi_slopes<<<1, block, 0, stream>>>(alibi_slopes, num_heads);
}

template void invokeBuildAlibiSlopes(float* alibi_slopes, const size_t num_heads, cudaStream_t stream);
template void invokeBuildAlibiSlopes(half* alibi_slopes, const size_t num_heads, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBuildAlibiSlopes(__nv_bfloat16* alibi_slopes, const size_t num_heads, cudaStream_t stream);
#endif

template void invokeGenRelativePosBiasV2(float*       relative_position_bias,
                                         const float* relative_coords_table,
                                         const int*   relative_position_bias_index,
                                         const float* cpb_mlp_weight1,
                                         const float* cpb_mlp_bias1,
                                         const float* cpb_mlp_weight2,
                                         const int    window_size,
                                         const int    cpb_mlp_in_dim,
                                         const int    cpb_mlp_out_dim,
                                         const int    head_num,
                                         cudaStream_t stream);

template void invokeGenRelativePosBiasV2(half*        relative_position_bias,
                                         const half*  relative_coords_table,
                                         const int*   relative_position_bias_index,
                                         const half*  cpb_mlp_weight1,
                                         const half*  cpb_mlp_bias1,
                                         const half*  cpb_mlp_weight2,
                                         const int    window_size,
                                         const int    cpb_mlp_in_dim,
                                         const int    cpb_mlp_out_dim,
                                         const int    head_num,
                                         cudaStream_t stream);

template void invokeGenRelativePosBiasV2(float*         relative_position_bias,
                                         const float*   relative_coords_table,
                                         const int64_t* relative_position_bias_index,
                                         const float*   cpb_mlp_weight1,
                                         const float*   cpb_mlp_bias1,
                                         const float*   cpb_mlp_weight2,
                                         const int      window_size,
                                         const int      cpb_mlp_in_dim,
                                         const int      cpb_mlp_out_dim,
                                         const int      head_num,
                                         cudaStream_t   stream);

template void invokeGenRelativePosBiasV2(half*          relative_position_bias,
                                         const half*    relative_coords_table,
                                         const int64_t* relative_position_bias_index,
                                         const half*    cpb_mlp_weight1,
                                         const half*    cpb_mlp_bias1,
                                         const half*    cpb_mlp_weight2,
                                         const int      window_size,
                                         const int      cpb_mlp_in_dim,
                                         const int      cpb_mlp_out_dim,
                                         const int      head_num,
                                         cudaStream_t   stream);
}  // namespace turbomind
