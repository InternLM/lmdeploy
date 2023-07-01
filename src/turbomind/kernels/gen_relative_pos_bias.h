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

#pragma once

#include "src/turbomind/utils/cuda_bf16_wrapper.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace fastertransformer {

enum class PositionEmbeddingType {
    relative,
    absolute,
};

template<typename T, typename Tindex>
void invokeGenRelativePosBias(T*            relative_position_bias,
                              const T*      relative_position_bias_table,
                              const Tindex* relative_position_bias_index,
                              const int     window_size,
                              const int     head_num,
                              cudaStream_t  stream);

template<typename T>
void invokeBuildAlibiSlopes(T* linear_position_bias_slopes, const size_t head_num, cudaStream_t stream);

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
                                cudaStream_t  stream);
}  // namespace fastertransformer
