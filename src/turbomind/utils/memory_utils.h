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

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include <cuda_runtime.h>

namespace turbomind {

template<typename T>
void invokeInPlaceTranspose102(
    T* data, T* workspace, const int dim0, const int dim1, const int dim2, bool copy = true, cudaStream_t stream = 0);

/// Element-wise dtype cast kernel.  Supports fp32 <-> fp16 <-> bf16.
void invokeDtypeCast(
    void* dst, const void* src, size_t count, DataType dst_dtype, DataType src_dtype, cudaStream_t stream = 0);

/// If *tensor* is a trivial float type that differs from *target_dtype*, cast
/// it in-place (allocates a temporary, casts, move-assigns).  Uses
/// Context::stream() internally — no stream parameter needed.
void EnsureFloatDtype(core::Tensor& tensor, DataType target_dtype);

}  // namespace turbomind
