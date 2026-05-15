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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cublasLt.h>
#include <cublas_v2.h>
#ifdef SPARSITY_ENABLED
#include <cusparseLt.h>
#endif

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/cuda_bf16_wrapper.h"

namespace turbomind {

/* **************************** debug tools ********************************* */
static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}

static const char* _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

// --- Unified CUDA error checking ---

void ReportCudaError(cudaError_t ec, const char* file, int line);
void ReportCuDrvError(CUresult ec, const char* file, int line);

#define TM_CUDA_CHECK(expr)                                                                                            \
    do {                                                                                                               \
        if (auto _ec = (expr); TM_UNLIKELY(_ec != cudaSuccess)) {                                                      \
            ::turbomind::ReportCudaError(_ec, __FILE__, __LINE__);                                                     \
        }                                                                                                              \
    } while (0)

#define TM_CUDRV_CHECK(expr)                                                                                           \
    do {                                                                                                               \
        if (auto _ec = (expr); TM_UNLIKELY(_ec != CUDA_SUCCESS)) {                                                     \
            ::turbomind::ReportCuDrvError(_ec, __FILE__, __LINE__);                                                    \
        }                                                                                                              \
    } while (0)

template<typename T>
void printMatrix(T* ptr, int m, int k, int stride, bool is_device_ptr);

void printMatrix(unsigned long long* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(int* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(size_t* ptr, int m, int k, int stride, bool is_device_ptr);

template<typename T>
void check_max_val(const T* result, const int size);

template<typename T>
void check_abs_mean_val(const T* result, const int size);

#define PRINT_FUNC_NAME_()                                                                                             \
    do {                                                                                                               \
        std::cout << "[TM][CALL] " << __FUNCTION__ << " " << std::endl;                                                \
    } while (0)

/* ***************************** common utils ****************************** */

int getSMVersion();

int getSMCount();

std::string getDeviceName();

template<class T>
inline T div_up(T a, T n)
{
    return (a + n - 1) / n;
}

int getDevice();

int getDeviceCount();

class CudaDeviceGuard {
public:
    CudaDeviceGuard(int device)
    {
        TM_CUDA_CHECK(cudaGetDevice(&last_device_id_));
        if (device != last_device_id_) {
            TM_CUDA_CHECK(cudaSetDevice(device));
        }
    }

    ~CudaDeviceGuard()
    {
        TM_CUDA_CHECK(cudaSetDevice(last_device_id_));
    }

private:
    int last_device_id_{-1};
};

void trim_default_mempool(int device_id);

/* ************************** end of common utils ************************** */
}  // namespace turbomind
