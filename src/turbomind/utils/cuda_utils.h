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
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/cuda_bf16_wrapper.h"
#include "src/turbomind/utils/logger.h"

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

template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        TM_LOG_ERROR((std::string("CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " " + file + ":"
                      + std::to_string(line))
                         .c_str());
        std::abort();
    }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) check((val), #val, file, line)

void syncAndCheck(const char* const file, int const line);

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__)

#define CUDRVCHECK(expr)                                                                                               \
    if (auto ec = expr; ec != CUDA_SUCCESS) {                                                                          \
        const char* p_str{};                                                                                           \
        cuGetErrorString(ec, &p_str);                                                                                  \
        p_str    = p_str ? p_str : "Unknown error";                                                                    \
        auto msg = fmtstr("[TM][ERROR] CUDA driver error: %s:%d '%s'", __FILE__, __LINE__, p_str);                     \
        throw std::runtime_error(msg.c_str());                                                                         \
    }

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

[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "")
{
    throw std::runtime_error(std::string("[TM][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n");
}

inline void myAssert(bool result, const char* const file, int const line, std::string const& info = "")
{
    if (!result) {
        throwRuntimeError(file, line, info);
    }
}

#define FT_CHECK(val) myAssert(bool(val), __FILE__, __LINE__)
#define FT_CHECK_WITH_INFO(val, info)                                                                                  \
    do {                                                                                                               \
        bool is_valid_val = bool(val);                                                                                 \
        if (!is_valid_val) {                                                                                           \
            turbomind::myAssert(is_valid_val, __FILE__, __LINE__, (info));                                             \
        }                                                                                                              \
    } while (0)

#define FT_THROW(info) throwRuntimeError(__FILE__, __LINE__, info)

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
        check_cuda_error(cudaGetDevice(&last_device_id_));
        if (device != last_device_id_) {
            check_cuda_error(cudaSetDevice(device));
        }
    }

    ~CudaDeviceGuard()
    {
        TM_CHECK_EQ(cudaSetDevice(last_device_id_), cudaSuccess);
    }

private:
    int last_device_id_{-1};
};

void trim_default_mempool(int device_id);

/* ************************** end of common utils ************************** */
}  // namespace turbomind
