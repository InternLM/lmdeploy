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

#include "encoder_igemm_func.h"  // TODO(bhsueh) Remove this include
#include "src/turbomind/utils/cublasAlgoMap.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#ifdef ENABLE_BF16
#include <cuda_fp16.h>
#endif
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#include <cuda_profiler_api.h>
#include <map>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

namespace fastertransformer {

// Scale Type Converter
// is_fp16_compute_type is only valid when T = half
template<typename T, bool is_fp16_compute_type = false>
struct ScaleTypeConverter {
    using Type = float;
};

template<>
struct ScaleTypeConverter<half, true> {
    using Type = half;
};

template<typename T, typename scaleT>
int LtHgemmCustomFind(cublasLtHandle_t   ltHandle,
                      int                batch_size,
                      int                seq_len,
                      int                head_num,
                      int                size_per_head,
                      int                m,
                      int                n,
                      int                k,
                      const scaleT*      alpha, /* host pointer */
                      const T*           A,
                      const T*           B,
                      const scaleT*      beta, /* host pointer */
                      T*                 C,
                      void*              workSpace,
                      size_t             workSpaceSize,
                      FILE*              fout,
                      customMatmulPerf_t perfResults[],
                      int                AlgoCombinations,
                      cudaDataType_t     dtype_fp8  = CUDA_R_32F,
                      int                batchCount = 1,
                      int64_t            strideA    = 0,
                      int64_t            strideB    = 0,
                      int64_t            strideD    = 0);

size_t calGemmTestBufSizeInByte(int            batch_size,
                                int            seq_len,
                                int            head_num,
                                int            size_per_head,
                                int            inter_size,
                                int            vocab_size,
                                int            int8_mode,
                                CublasDataType data_type);

size_t calGemmTestBufSizeInByteXlnet(
    int batch_size, int seq_len, int head_num, int size_per_head, int inter_size, int hidden_units, int is_fp16);

int printPerfStructure(int                       batch_size,
                       int                       seq_len,
                       int                       head_num,
                       int                       size_per_head,
                       int                       m,
                       int                       n,
                       int                       k,
                       const customMatmulPerf_t& perf,
                       FILE*                     fout,
                       CublasDataType            data_type,
                       int                       hasPrint,
                       int                       batch_count = 1);

}  // namespace fastertransformer
