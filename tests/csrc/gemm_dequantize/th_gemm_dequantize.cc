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

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "src/turbomind/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "src/turbomind/th_op/th_utils.h"
#include "src/turbomind/utils/cuda_bf16_wrapper.h"

#include "cutlass/numeric_types.h"

using torch::Tensor;

namespace torch_ext {

namespace ft = turbomind;

template<typename T, typename WeightType>
Tensor fused_gemm_dq_helper(
    Tensor input_activations, Tensor weight, Tensor scales, const int64_t timing_iterations, float& avg_time)
{
    const at::ScalarType _st    = input_activations.scalar_type();
    const int            m      = input_activations.size(0);
    const int            n      = scales.size(0);
    const int            k      = input_activations.size(1);
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();

    const T*          input_act_ptr = get_ptr<const T>(input_activations);
    const WeightType* weight_ptr    = get_ptr<const WeightType>(weight);
    const T*          scales_ptr    = get_ptr<const T>(scales);

    turbomind::CutlassFpAIntBGemmRunner<T, WeightType> fused_gemm_dq_runner;
    const int ws_bytes = fused_gemm_dq_runner.getWorkspaceSize(m, n, k);

    auto output_tensor = torch::empty({m, n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    auto ws_tensor     = torch::empty({ws_bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

    T*   output_tensor_ptr = get_ptr<T>(output_tensor);
    char* ws_ptr            = get_ptr<char>(ws_tensor);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int64_t iter = 0; iter < timing_iterations; ++iter) {
        fused_gemm_dq_runner.gemm(
            input_act_ptr, weight_ptr, scales_ptr, output_tensor_ptr, m, n, k, ws_ptr, ws_bytes, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float total_time_ms = 0;
    cudaEventElapsedTime(&total_time_ms, start, stop);
    avg_time = total_time_ms / float(timing_iterations);

    return output_tensor;
}

Tensor
_fused_gemm_dq(Tensor input_activations, Tensor weight, Tensor scales, int64_t timing_iterations, float& avg_time)
{
    const at::ScalarType _st = input_activations.scalar_type();
    CHECK_INPUT(scales, _st);

    TORCH_CHECK(input_activations.dim() == 2, "Invalid rank for activations");
    TORCH_CHECK(weight.dim() == 2, "Invalid rank for weight");
    TORCH_CHECK(scales.dim() == 1, "Invalid rank for scales");

    const int m = input_activations.size(0);
    const int n = scales.size(0);
    const int k = input_activations.size(1);

    TORCH_CHECK(input_activations.size(1) == weight.size(0), "dim 1 of act and dim 0 of weight must be equal");

    // We signal int4 by having the last weight dim be half the size of the scales.
    // This is because int4 elements are packed into a single byte.
    torch::ScalarType quant_type = weight.scalar_type();
    if (weight.size(-1) == scales.size(-1) / 2) {
        quant_type = at::ScalarType::QUInt4x2;
    }
    else {
        TORCH_CHECK(weight.size(-1) == scales.size(-1),
                    "Last dim of weight and scales must be equal for int8 "
                    "or last dim of scale must be 2x last dim of weight for int4.");
    }

    Tensor output_tensor;
    switch (_st) {
        case at::ScalarType::Half: {
            if (quant_type == torch::kInt8) {
                output_tensor =
                    fused_gemm_dq_helper<half, uint8_t>(input_activations, weight, scales, timing_iterations, avg_time);
            }
            else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = fused_gemm_dq_helper<half, cutlass::uint4b_t>(
                    input_activations, weight, scales, timing_iterations, avg_time);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            if (quant_type == torch::kInt8) {
                output_tensor = fused_gemm_dq_helper<__nv_bfloat16, uint8_t>(
                    input_activations, weight, scales, timing_iterations, avg_time);
            }
            else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = fused_gemm_dq_helper<__nv_bfloat16, cutlass::uint4b_t>(
                    input_activations, weight, scales, timing_iterations, avg_time);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#endif
        default:
            throw std::runtime_error("Unsupported tensor type. Got " + std::string(at::toString(_st)));
    }
    return output_tensor;
}

Tensor fused_gemm_dq(Tensor input_activations, Tensor weight, Tensor scales)
{
    float dummy = 0.f;
    return _fused_gemm_dq(input_activations, weight, scales, 1, dummy);
}

Tensor
bench_cublas(Tensor input_activations, Tensor weight_dequantized, const int64_t timing_iterations, float& avg_time)
{
    using namespace turbomind;
    const int m = input_activations.size(0);
    const int n = weight_dequantized.size(1);
    const int k = input_activations.size(1);

    const void* input_act_ptr = get_ptr<const void>(input_activations);
    const void* weight_ptr    = get_ptr<const void>(weight_dequantized);

    cublasHandle_t       handle = at::cuda::getCurrentCUDABlasHandle();
    const at::ScalarType _st    = input_activations.scalar_type();

    TORCH_CHECK(input_activations.size(1) == weight_dequantized.size(0),
                "CUBLAS_BENCH: dim 1 of act and dim 0 of weight must be equal");
    CHECK_INPUT(input_activations, _st);
    CHECK_INPUT(weight_dequantized, _st);

    auto  output_tensor     = torch::empty({m, n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    void* output_tensor_ptr = get_ptr<void>(output_tensor);

    TORCH_CHECK(_st == at::ScalarType::Half || _st == at::ScalarType::BFloat16, "Input type must be float or bfloat");
    cudaDataType_t cublasType = _st == at::ScalarType::Half ? CUDA_R_16F : CUDA_R_16BF;

    float alpha = 1.0f;
    float beta  = 0.0f;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    cudaEventRecord(start, stream);
    for (int64_t iter = 0; iter < timing_iterations; ++iter) {
        status = cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              &alpha,
                              weight_ptr,
                              cublasType,
                              n,
                              input_act_ptr,
                              cublasType,
                              k,
                              &beta,
                              output_tensor_ptr,
                              cublasType,
                              n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float total_time_ms = 0;
    cudaEventElapsedTime(&total_time_ms, start, stop);
    avg_time = total_time_ms / float(timing_iterations);
    check_cuda_error(status);
    return output_tensor;
}

std::vector<std::vector<Tensor>> benchmark_against_cublas_fp(Tensor        input_activations,
                                                             Tensor        weight_quantized,
                                                             Tensor        scales,
                                                             Tensor        weight_dequantized,
                                                             const int64_t timing_iterations)
{
    float  cublas_time   = 0.f;
    float  ft_time       = 0.f;
    Tensor cublas_result = bench_cublas(input_activations, weight_dequantized, timing_iterations, cublas_time);
    Tensor ft_result     = _fused_gemm_dq(input_activations, weight_quantized, scales, timing_iterations, ft_time);

    auto timing_tensor =
        torch::empty({2}, torch::dtype(at::ScalarType::Float).device(torch::kCPU).requires_grad(false));
    timing_tensor[0] = cublas_time;
    timing_tensor[1] = ft_time;

    // const int m = input_activations.size(0);
    // const int n = weight_dequantized.size(1);
    // const int k = input_activations.size(1);
    // std::cout << "m, n, k" << m << ", " << n << ", " << k << std::endl;
    // std::cout << "cuBLAS time (ms) " << cublas_time << std::endl;
    // std::cout << "FT time (ms) " << ft_time << std::endl;

    return {{timing_tensor}, {cublas_result, ft_result}};
}

template<typename T, typename WeightType>
Tensor fused_gemm_dq_bias_act_helper(
    Tensor input_activations, Tensor weight, Tensor scales, Tensor bias, ft::ActivationType activation_type)
{
    const at::ScalarType _st    = input_activations.scalar_type();
    const int            m      = input_activations.size(0);
    const int            n      = scales.size(0);
    const int            k      = input_activations.size(1);
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();

    const T*          input_act_ptr = get_ptr<const T>(input_activations);
    const WeightType* weight_ptr    = get_ptr<const WeightType>(weight);
    const T*          scales_ptr    = get_ptr<const T>(scales);
    const T*          bias_ptr      = get_ptr<const T>(bias);

    turbomind::CutlassFpAIntBGemmRunner<T, WeightType> fused_gemm_dq_runner;
    const int ws_bytes = fused_gemm_dq_runner.getWorkspaceSize(m, n, k);

    auto output_tensor = torch::empty({m, n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    auto ws_tensor     = torch::empty({ws_bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

    T*   output_tensor_ptr = get_ptr<T>(output_tensor);
    char* ws_ptr            = get_ptr<char>(ws_tensor);

    fused_gemm_dq_runner.gemm_bias_act(input_act_ptr,
                                       weight_ptr,
                                       scales_ptr,
                                       bias_ptr,
                                       output_tensor_ptr,
                                       m,
                                       n,
                                       k,
                                       activation_type,
                                       ws_ptr,
                                       ws_bytes,
                                       stream);

    return output_tensor;
}

Tensor fused_gemm_dq_bias_act(
    Tensor input_activations, Tensor weight, Tensor scales, Tensor bias, std::string activation_type_str)
{
    const at::ScalarType _st = input_activations.scalar_type();
    CHECK_INPUT(scales, _st);
    CHECK_INPUT(bias, _st);

    TORCH_CHECK(input_activations.dim() == 2, "Invalid rank for activations");
    TORCH_CHECK(weight.dim() == 2, "Invalid rank for weight");
    TORCH_CHECK(scales.dim() == 1, "Invalid rank for scales");
    TORCH_CHECK(bias.dim() == 1, "Invalid rank for bias");

    const int m = input_activations.size(0);
    const int n = scales.size(0);
    const int k = input_activations.size(1);

    TORCH_CHECK(bias.size(0) == n, "Must have 1 bias value for each output column");
    TORCH_CHECK(input_activations.size(1) == weight.size(0), "dim 1 of act and dim 0 of weight must be equal");

    // We signal int4 by having the last weight dim be half the size of the scales.
    // This is because int4 elements are packed into a single byte.
    torch::ScalarType quant_type = weight.scalar_type();
    if (weight.size(-1) == scales.size(-1) / 2) {
        quant_type = at::ScalarType::QUInt4x2;
    }
    else {
        TORCH_CHECK(weight.size(-1) == scales.size(-1),
                    "Last dim of weight and scales must be equal for int8 "
                    "or last dim of scale must be 2x last dim of weight for int4.");
    }

    ft::ActivationType activation_type = ft::ActivationType::InvalidType;
    if (activation_type_str == "identity") {
        activation_type = ft::ActivationType::Identity;
    }
    else {
        activation_type = ft::getActivationType(activation_type_str);
    }

    TORCH_CHECK(!isGatedActivation(activation_type), "Fused gated activations not supported.");

    Tensor output_tensor;
    switch (_st) {
        case at::ScalarType::Half: {
            if (quant_type == torch::kInt8) {
                output_tensor = fused_gemm_dq_bias_act_helper<half, uint8_t>(
                    input_activations, weight, scales, bias, activation_type);
            }
            else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = fused_gemm_dq_bias_act_helper<half, cutlass::uint4b_t>(
                    input_activations, weight, scales, bias, activation_type);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            if (quant_type == torch::kInt8) {
                output_tensor = fused_gemm_dq_bias_act_helper<__nv_bfloat16, uint8_t>(
                    input_activations, weight, scales, bias, activation_type);
            }
            else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = fused_gemm_dq_bias_act_helper<__nv_bfloat16, cutlass::uint4b_t>(
                    input_activations, weight, scales, bias, activation_type);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
#endif
        default:
            throw std::runtime_error("Unsupported tensor type. Got " + std::string(at::toString(_st)));
    }
    return output_tensor;
}

TORCH_LIBRARY(gemm_dq_unit_ops, m)
{
    m.def("fused_gemm_dq", fused_gemm_dq);
    m.def("benchmark_against_cublas_fp", benchmark_against_cublas_fp);
    m.def("fused_gemm_dq_bias_act", fused_gemm_dq_bias_act);
}
}  // namespace torch_ext
