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
#include <algorithm>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#ifdef __linux__
#include <sys/time.h>
#include <unistd.h>
#endif
#ifdef _MSC_VER
#include "src/turbomind/windows/gettimeofday.h"
#endif
#include <time.h>
#include <vector>

namespace turbomind {

/* CAUTION : must match cublasLtMatmulTile_t */
const char* const matmulTileName[] = {"UNDEF",  "8x8",     "8x16",    "16x8",    "8x32",    "16x16",  "32x8",
                                      "8x64",   "16x32",   "32x16",   "64x8",    "32x32",   "32x64",  "64x32",
                                      "32x128", "64x64",   "128x32",  "64x128",  "128x64",  "64x256", "128x128",
                                      "256x64", "64x512",  "128x256", "256x128", "512x64",  "64x96",  "96*64",
                                      "96x128", "128x160", "160x128", "192x128", "128x192", "128x96", "END"};

int generate_encoder_igemm_config(
    int batch_size, int seq_len, int head_num, int size_per_head, void* buffer, bool isAppend = true);

int printPerfStructure(int m, int n, int k, const customMatmulPerf_t& perf, FILE* fout, int hasPrint);

int printBatchPerfStructure(
    int batchCount, int m, int n, int k, const customMatmulPerf_t& perf, FILE* fout, int hasPrint);

template<typename T, typename scaleT>
int LtIgemmCustomFind(cublasLtHandle_t ltHandle,
                      int              m,
                      int              n,
                      int              k,
                      const scaleT*    alpha, /* host pointer */
                      const int8_t*    A,
                      const int8_t*    B,
                      const scaleT*    beta, /* host pointer */
                      T*               C,
                      void*            workSpace,
                      size_t           workSpaceSize,
                      FILE*            fout);

template<typename T, typename scaleT>
int LtBatchIgemmCustomFind(cublasLtHandle_t ltHandle,
                           int              batchCount,
                           int              m,
                           int              n,
                           int              k,
                           const scaleT*    alpha, /* host pointer */
                           const int8_t*    A,
                           const int8_t*    B,
                           const scaleT*    beta, /* host pointer */
                           T*               C,
                           void*            workSpace,
                           size_t           workSpaceSize,
                           FILE*            fout);

void matInit(int rows, int cols, int8_t* p, int ld);

}  // namespace turbomind
