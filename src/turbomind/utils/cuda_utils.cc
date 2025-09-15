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

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/macro.h"
#include <driver_types.h>
#include <regex>

namespace turbomind {

void syncAndCheck(const char* const file, int const line)
{
    // When FT_DEBUG_LEVEL=DEBUG, must check error
    static char* level_name = std::getenv("TM_DEBUG_LEVEL");
    if (level_name != nullptr) {
        static std::string level = std::string(level_name);
        if (level == "DEBUG") {
            cudaDeviceSynchronize();
            cudaError_t result = cudaGetLastError();
            if (result) {
                TM_LOG_ERROR((std::string("CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " " + file + ":"
                              + std::to_string(line))
                                 .c_str());
                std::abort();
            }
            TM_LOG_DEBUG(fmtstr("run syncAndCheck at %s:%d", file, line));
        }
    }
}

/* **************************** debug tools ********************************* */

template<typename T>
void printMatrix(T* ptr, int m, int k, int stride, bool is_device_ptr)
{
    T* tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_cuda_error(cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }
    else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        }
        else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%7.3f ", (float)tmp[ii * stride + jj]);
            }
            else {
                printf("%7d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}

template void printMatrix(float* ptr, int m, int k, int stride, bool is_device_ptr);
template void printMatrix(half* ptr, int m, int k, int stride, bool is_device_ptr);
#ifdef ENABLE_BF16
template void printMatrix(__nv_bfloat16* ptr, int m, int k, int stride, bool is_device_ptr);
#endif

void printMatrix(unsigned long long* ptr, int m, int k, int stride, bool is_device_ptr)
{
    typedef unsigned long long T;
    T*                         tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_cuda_error(cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }
    else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        }
        else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%4llu ", tmp[ii * stride + jj]);
            }
            else {
                printf("%4d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}

void printMatrix(int* ptr, int m, int k, int stride, bool is_device_ptr)
{
    typedef int T;
    T*          tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_cuda_error(cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }
    else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        }
        else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%4d ", tmp[ii * stride + jj]);
            }
            else {
                printf("%4d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}

// multiple definitions for msvc
#ifndef _MSC_VER
void printMatrix(size_t* ptr, int m, int k, int stride, bool is_device_ptr)
{
    typedef size_t T;
    T*             tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_cuda_error(cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }
    else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        }
        else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%4ld ", tmp[ii * stride + jj]);
            }
            else {
                printf("%4d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}
#endif

template<typename T>
void check_max_val(const T* result, const int size)
{
    T* tmp = new T[size];
    cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
    float max_val = -100000;
    for (int i = 0; i < size; i++) {
        float val = static_cast<float>(tmp[i]);
        if (val > max_val) {
            max_val = val;
        }
    }
    delete tmp;
    printf("[INFO][CUDA] addr %p max val: %f \n", result, max_val);
}

template void check_max_val(const float* result, const int size);
template void check_max_val(const half* result, const int size);
#ifdef ENABLE_BF16
template void check_max_val(const __nv_bfloat16* result, const int size);
#endif

template<typename T>
void check_abs_mean_val(const T* result, const int size)
{
    T* tmp = new T[size];
    cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += abs(static_cast<float>(tmp[i]));
    }
    delete tmp;
    printf("[INFO][CUDA] addr %p abs mean val: %f \n", result, sum / size);
}

template void check_abs_mean_val(const float* result, const int size);
template void check_abs_mean_val(const half* result, const int size);
#ifdef ENABLE_BF16
template void check_abs_mean_val(const __nv_bfloat16* result, const int size);
#endif

/* ***************************** common utils ****************************** */

int getSMVersion()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    check_cuda_error(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

int getSMCount()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int sm_count{};
    check_cuda_error(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    return sm_count;
}

std::string getDeviceName()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    cudaDeviceProp props;
    check_cuda_error(cudaGetDeviceProperties(&props, device));
    return std::string(props.name);
}

int getDevice()
{
    int current_dev_id = 0;
    check_cuda_error(cudaGetDevice(&current_dev_id));
    return current_dev_id;
}

int getDeviceCount()
{
    int count = 0;
    check_cuda_error(cudaGetDeviceCount(&count));
    return count;
}

void trim_default_mempool(int device_id)
{
    cudaMemPool_t mempool;
    check_cuda_error(cudaDeviceGetDefaultMemPool(&mempool, device_id));
    check_cuda_error(cudaMemPoolTrimTo(mempool, 0));
}

/* ************************** end of common utils ************************** */
}  // namespace turbomind
