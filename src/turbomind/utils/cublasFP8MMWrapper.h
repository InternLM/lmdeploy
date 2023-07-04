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

#include "3rdparty/fp8_qgmma_1x1/fp8_qgmma_1x1_utils.h"
#include "cuda_utils.h"
#include "src/turbomind/utils/cublasAlgoMap.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/cuda_fp8_utils.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <string>

#pragma once

namespace turbomind {

class cublasFP8MMWrapper: public cublasMMWrapper {
public:
    cublasFP8MMWrapper(cublasLtHandle_t cublaslt_handle_,
                       cudaStream_t     stream,
                       cublasAlgoMap*   map,
                       std::mutex*      mu,
                       IAllocator*      allocator);

    cublasFP8MMWrapper(cublasHandle_t   cublas_handle,
                       cublasLtHandle_t cublaslt_handle,
                       cudaStream_t     stream,
                       cublasAlgoMap*   map,
                       std::mutex*      mu,
                       IAllocator*      allocator);

    virtual ~cublasFP8MMWrapper();

    cublasFP8MMWrapper(const cublasFP8MMWrapper& wrapper);

    virtual void cublasVersionCheck() override;

    void Gemm(__nv_bfloat16*       res,
              int                  batchCount,
              int                  m,
              int                  n,
              int                  k,
              int64_t              stridea,
              int64_t              strideb,
              int64_t              stridec,
              const float*         alpha,
              const float*         beta,
              const __nv_fp8_e4m3* input,
              const __nv_fp8_e4m3* kernel,
              const float*         input_scale,
              const float*         kernel_scale);

    void Gemm(__nv_bfloat16*       res,
              int                  batchCount,
              int                  m,
              int                  n,
              int                  k,
              int64_t              stridea,
              int64_t              strideb,
              int64_t              stridec,
              const float*         alpha,
              const float*         beta,
              const __nv_fp8_e4m3* input,
              const __nv_fp8_e4m3* kernel,
              const float*         input_scale,
              const float*         kernel_scale,
              cudaStream_t         stream,
              bool                 fastAccum = true);

    void Gemm(__nv_fp8_e4m3*       res,
              int                  batchCount,
              int                  m,
              int                  n,
              int                  k,
              int64_t              stridea,
              int64_t              strideb,
              int64_t              stridec,
              const float*         alpha,
              const float*         beta,
              const __nv_fp8_e4m3* input,
              const __nv_fp8_e4m3* kernel,
              const float*         input_scale,
              const float*         kernel_scale,
              const float*         output_scale);

    void Gemm(__nv_fp8_e4m3*       res,
              int                  batchCount,
              int                  m,
              int                  n,
              int                  k,
              int64_t              stridea,
              int64_t              strideb,
              int64_t              stridec,
              const float*         alpha,
              const float*         beta,
              const __nv_fp8_e4m3* input,
              const __nv_fp8_e4m3* kernel,
              const float*         input_scale,
              const float*         kernel_scale,
              const float*         output_scale,
              cudaStream_t         stream,
              bool                 fastAccum = true);

    template<bool RELU, bool GELU>
    void Conv1x1Gemm(__nv_fp8_e4m3*       res,
                     int                  m,
                     int                  n,
                     int                  k,
                     const __nv_fp8_e4m3* input,
                     const __nv_fp8_e4m3* kernel,
                     const __nv_bfloat16* bias,
                     const float          input_scale,
                     const float          kernel_scale,
                     const float          output_scale,
                     cudaStream_t         stream);

    template<bool RELU, bool GELU>
    void Gemm_Bias_Act(__nv_bfloat16*       res,
                       int                  batchCount,
                       int                  m,
                       int                  n,
                       int                  k,
                       int64_t              stridea,
                       int64_t              strideb,
                       int64_t              stridec,
                       const float*         alpha,
                       const float*         beta,
                       const __nv_fp8_e4m3* input,
                       const __nv_fp8_e4m3* kernel,
                       const float*         input_scale,
                       const float*         kernel_scale,
                       const __nv_bfloat16* bias,
                       const float*         output_scale,
                       cudaStream_t         stream);

    template<bool RELU, bool GELU>
    void Gemm_Bias_Act(__nv_fp8_e4m3*       res,
                       int                  batchCount,
                       int                  m,
                       int                  n,
                       int                  k,
                       int64_t              stridea,
                       int64_t              strideb,
                       int64_t              stridec,
                       const float*         alpha,
                       const float*         beta,
                       const __nv_fp8_e4m3* input,
                       const __nv_fp8_e4m3* kernel,
                       const float*         input_scale,
                       const float*         kernel_scale,
                       const __nv_bfloat16* bias,
                       const float*         output_scale,
                       cudaStream_t         stream);

private:
    int                         version_major_, version_minor_, version_patch_;
    turbomind::qgmma1x1Launcher qgmmaLauncher;
    void*                       cublas_workspace_qgemm_ = nullptr;
};

}  // namespace turbomind
