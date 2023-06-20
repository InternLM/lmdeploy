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

#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "src/fastertransformer/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/logger.h"

#include "cutlass/numeric_types.h"

using torch::Tensor;
using torch_ext::get_ptr;

namespace ft = fastertransformer;

template<typename T>
void int8_gemm_test(
    const int m, 
    const int n, 
    const int k, 
    const at::ScalarType output_data_type,
    const QuantMode quant_mode,
    const int iters)
{
     const bool per_token_quant = quant_mode == QuantMode::PerTokenChannelQuant
        || quant_mode == QuantMode::PerTokenQuant;
    const bool per_channel_quant = quant_mode == QuantMode::PerTokenChannelQuant
        || quant_mode == QuantMode::PerChannelQuant;
    const int row_scale_size = per_token_quant ? m : 1;
    const int col_scale_size = per_channel_quant ? n : 1;

    const at::ScalarType at_int32 = at::ScalarType::Int;
    const at::ScalarType at_int8  = at::ScalarType::Char;
    const at::ScalarType at_fp16  = at::ScalarType::Half;
    const at::ScalarType at_bf16  = at::ScalarType::BFloat16;
    const at::ScalarType at_fp32  = at::ScalarType::Float;

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;

    torch::manual_seed(0);

    auto x = torch::randint(-128, 128, {m, k}, torch::dtype(at_int32).requires_grad(false));
    auto w = torch::randint(-128, 128, {k, n}, torch::dtype(at_int32).requires_grad(false));

    ft::FT_CHECK(torch::allclose(x, x.to(at_int8).to(at_int32)));
    ft::FT_CHECK(torch::allclose(w, w.to(at_int8).to(at_int32)));

    auto y = torch::matmul(x, w);

    ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, {(size_t)m, (size_t)k}, get_ptr<int32_t>(x)}.saveNpy("x.npy");
    ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, {(size_t)k, (size_t)n}, get_ptr<int32_t>(w)}.saveNpy("w.npy");
    ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, {(size_t)m, (size_t)n}, get_ptr<int32_t>(y)}.saveNpy("y.npy");

    auto x_gpu = x.to(at_int8).to(torch::kCUDA);
    auto w_T_gpu = w.to(at_int8).to(torch::kCUDA).t().contiguous();
    auto w_gpu = w.to(at_int8).to(torch::kCUDA);
    auto y_gpu = torch::zeros({m, n}, torch::dtype(output_data_type).device(torch::kCUDA).requires_grad(false));
    auto y_gpu_int32 = torch::zeros({m, n}, torch::dtype(at_int32).device(torch::kCUDA).requires_grad(false));

    auto alpha_row_cultass = torch::ones({row_scale_size, 1}, torch::dtype(at_fp32).requires_grad(false)) * (1.0 / 100) *
        torch::randint(1, 10, {row_scale_size, 1}, torch::dtype(at_fp32));
    auto alpha_col_cutlass = torch::ones({1, col_scale_size}, torch::dtype(at_fp32).requires_grad(false)) * (1.0 / 100) *
        torch::randint(1, 10, {1, col_scale_size}, torch::dtype(at_fp32));

    auto alpha_row_torch = alpha_row_cultass.expand({m, 1});
    auto alpha_col_torch = alpha_col_cutlass.expand({1, n});

    // std::cout << alpha_row << std::endl;
    auto alpha_row_gpu = alpha_row_cultass.to(torch::kCUDA);
    auto alpha_col_gpu = alpha_col_cutlass.to(torch::kCUDA);

    auto alpha_row_col_scale_gpu = torch::matmul(alpha_row_torch, alpha_col_torch).to(torch::kCUDA);

    ft::CutlassInt8GemmRunner<T> cutlass_runner_half;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // warm_up
    cutlass_runner_half.gemm(get_ptr<int8_t>(x_gpu),
            get_ptr<int8_t>(w_T_gpu),
            quant_mode,
            get_ptr<float>(alpha_col_gpu),
            get_ptr<float>(alpha_row_gpu),
            get_ptr<T>(y_gpu),
            m,
            n,
            k,
            nullptr,
            0,
            stream);

    ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT8, {(size_t)m, (size_t)k}, get_ptr<int8_t>(x_gpu)}.saveNpy("x_gpu.npy");
    ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT8, {(size_t)n, (size_t)k}, get_ptr<int8_t>(w_T_gpu)}.saveNpy("w_T_gpu.npy");
    ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT8, {(size_t)k, (size_t)n}, get_ptr<int8_t>(w_gpu)}.saveNpy("w_gpu.npy");
    ft::Tensor{ft::MEMORY_GPU, ft::TYPE_FP16, {(size_t)m, (size_t)n}, get_ptr<T>(y_gpu)}.saveNpy("y_gpu.npy");
    ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, {(size_t)m, (size_t)n}, get_ptr<int32_t>(y_gpu_int32)}.saveNpy("y_gpu_int32.npy");

    ft::check_cuda_error(cudaStreamSynchronize(stream));
    auto start = high_resolution_clock::now();

    for (int i = 0; i < iters; ++i) {
        cutlass_runner_half.gemm(get_ptr<int8_t>(x_gpu),
            get_ptr<int8_t>(w_T_gpu),
            quant_mode,
            get_ptr<float>(alpha_col_gpu),
            get_ptr<float>(alpha_row_gpu),
            get_ptr<T>(y_gpu),
            m,
            n,
            k,
            nullptr,
            0,
            stream);
    }

    ft::check_cuda_error(cudaStreamSynchronize(stream));
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    if (torch::allclose((y.to(torch::kCUDA).to(at_fp32) * alpha_row_col_scale_gpu.to(torch::kCUDA)).to(output_data_type), y_gpu)) {
        FT_LOG_INFO("SUCCESS " + std::to_string((double(duration.count()) / iters) / 1000) + " ms");
    } else {
        FT_LOG_ERROR("FAILED " + std::to_string((double(duration.count()) / iters) / 1000) + " ms");
        // std::cout << "diff " << (y.to(torch::kCUDA).to(at_fp32) * alpha_row_col_scale_gpu.to(torch::kCUDA)).to(at_fp16) - y_gpu << std::endl;
    }
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        FT_LOG_ERROR("arguments missing, needs m, n, k, data_type(fp16=0, bf16=1), quant_mode (perTensor=0, perToken=1, perChannel=2, perTokenChannel=3), iters.");
        return 0;
    }

    const int m = atoi(argv[1]);
    const int n = atoi(argv[2]);
    const int k = atoi(argv[3]);
    const at::ScalarType output_data_type = atoi(argv[4]) == 0 ?
        at::ScalarType::Half : at::ScalarType::BFloat16;
    const QuantMode quant_mode = static_cast<QuantMode>(atoi(argv[5]));
    if (quant_mode == QuantMode::PerChannelQuant) {
        printf("per channel quant \n");
    }
    const int iters = atoi(argv[6]);

    if (output_data_type == at::ScalarType::Half) {
        int8_gemm_test<half>(m, n, k, output_data_type, quant_mode, iters);
    } else {
#if ENABLE_BF16
        int8_gemm_test<__nv_bfloat16>(m, n, k, output_data_type, quant_mode, iters);
#endif
    }

    return 0;
}
