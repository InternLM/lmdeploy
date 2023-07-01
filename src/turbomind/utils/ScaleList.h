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
#include "stdlib.h"

namespace turbomind {

#define ACTIVATION_AMAX_NUM 72
#define INT8O_GEMM_NUM 8
#define TRT_AMAX_NUM 3
#define SCALE_RESERVE_NUM 21

struct ScaleList {
    // Part 1 -- 72:
    //   First 72 are for activation amaxs. For each activation amax, there are 4 values: amax, amax/127.0f,
    //   amax/127.0f/127.0f, 127.0f/amax -- input_amax 0-3 , Q_aftergemm_amax 4-7, Qbias_amax 8-11, K_aftergemm_amax
    //   12-15, Kbias_amax 16-19, V_aftergemm_amax 20-23, Vbias_amax 24-27, bmm1_amax 28-31, Softmax_amax 32-35,
    //   bmm2_amax 36-39, Proj_aftergemm_scale 40-43, ProjBiasNorm_amax 44-47, FC1_aftergemm_amax 48-51, F1Bias_amax
    //   52-55, FC2_aftergemm_amax 56-59, F2BiasNorm_amax 60-63, reserve 64-71
    // Part 2 -- 9*hidden_dim:
    //   Kernel amaxs, for each kernel amax list, there are output_channel values : query_weight_amax_list,
    //   key_weight_amax_list, value_weight_amax_list, proj_weight_amax_list, FC1_weight_amax_list, FC2_weight_amax_list
    // Part 3 -- 8:
    //   Int8 gemm deQFactor list (8 values): Q_deQ_scale, K_deQ_scale, V_deQ_scale, bmm1_deQ_scale, bmm2_deQ_scale,
    //   FC0_deQ_scale, FC1_deQ_scale, FC2_deQ_scale
    // Part 4 -- 3:
    //   Amax used in trt fused mha kernel (3 values) : QKVbias_amax, Softmax_amax, bmm2_amax
    // Part 5 -- 21: reverse
    const float* d_scale_list_ = nullptr;
    const float* h_scale_list_ = nullptr;
    size_t       size_         = ACTIVATION_AMAX_NUM + 9 * 768 + INT8O_GEMM_NUM + TRT_AMAX_NUM;
    size_t       p2_offset_    = ACTIVATION_AMAX_NUM;
    size_t       p3_offset_    = ACTIVATION_AMAX_NUM + 9 * 768;
    size_t       p4_offset_    = ACTIVATION_AMAX_NUM + 9 * 768 + INT8O_GEMM_NUM;
};

}  // namespace turbomind
