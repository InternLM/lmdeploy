/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "src/turbomind/utils/cuda_bf16_wrapper.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

namespace turbomind {

// clang-format off
template<typename T> struct GeluActivation;
template<typename T> struct ReluActivation;
template<typename T> struct SiluActivation;
template<typename T> struct IdentityActivation;
// clang-format on

template<template<typename T> class Activation, typename T, typename BT>
void invokeGenericActivation(T*           out,
                             const BT*    bias,
                             const T*     gated_weights,
                             const BT*    gated_bias,
                             const int*   ia3_tasks,
                             const T*     ia3_weights,
                             const int    m,
                             const int    n,
                             const int    int8_mode,
                             const float* activation_in,
                             const float* activation_out,
                             const int*   padding_offset,
                             const int    seq_len,
                             cudaStream_t stream);

template<template<typename T> class Activation, typename T, typename BT>
void invokeGenericActivation(T*           out,
                             const BT*    bias,
                             const T*     gated_weights,
                             const BT*    gated_bias,
                             const int*   ia3_tasks,
                             const T*     ia3_weights,
                             const int    m,
                             const int    n,
                             const int    int8_mode,
                             const float* activation_in,
                             const float* activation_out,
                             cudaStream_t stream)
{
    invokeGenericActivation<Activation, T, BT>(out,
                                               bias,
                                               gated_weights,
                                               gated_bias,
                                               ia3_tasks,
                                               ia3_weights,
                                               m,
                                               n,
                                               int8_mode,
                                               activation_in,
                                               activation_out,
                                               (const int*)nullptr,
                                               0,
                                               stream);
}

template<typename T>
void invokeAddBiasGeluV2(T*           out,
                         const T*     bias,
                         const int*   ia3_tasks,
                         const T*     ia3_weights,
                         const int*   padding_offset,
                         const int    seq_len,
                         const int    m,
                         const int    n,
                         cudaStream_t stream);

template<typename T>
void invokeAddBias(T* out, T const* bias, const int m, const int n, cudaStream_t stream)
{
    invokeGenericActivation<IdentityActivation, T, T>(
        out, bias, nullptr, nullptr, nullptr, nullptr, m, n, 0, nullptr, nullptr, stream);
}

template<typename T>
void invokeAddBiasGeluV2(
    T* out, const T* bias, const int* ia3_tasks, const T* ia3_weights, const int m, const int n, cudaStream_t stream)
{
    invokeAddBiasGeluV2(out, bias, ia3_tasks, ia3_weights, nullptr, 0, m, n, stream);
}

template<typename T>
void invokeAddBiasTanh(T* out, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeSigmoid(T* data, const int size, const float scale, cudaStream_t stream);

template<template<typename T> class Activation, typename T, typename BT>
void invokeFusedBiasResidualActivation(
    T* out, const BT* bias, const T* residual, const int m, const int n, cudaStream_t stream);

}  // namespace turbomind
