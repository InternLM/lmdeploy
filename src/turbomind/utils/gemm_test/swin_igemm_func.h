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

#include "src/turbomind/utils/cublasAlgoMap.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/gemm_test/encoder_igemm_func.h"
#include <algorithm>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace turbomind {

/* CAUTION : must match cublasLtMatmulTile_t */
// const char* const matmulTileName[] = {
//     "UNDEF",  "8x8",    "8x16",    "16x8",   "8x32",   "16x16",   "32x8",    "8x64",   "16x32",
//     "32x16",  "64x8",   "32x32",   "32x64",  "64x32",  "32x128",  "64x64",   "128x32", "64x128",
//     "128x64", "64x256", "128x128", "256x64", "64x512", "128x256", "256x128", "512x64",
// };

int generate_swin_igemm_config(
    int batch_size, int seq_len, int head_num, int size_per_head, void* buffer, bool isAppend = true);

}  // namespace turbomind
