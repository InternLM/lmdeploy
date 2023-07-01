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

#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_fp8_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<typename T>
void deviceMalloc(T** ptr, size_t size, bool is_random_initialize = true);

template<typename T>
void deviceMemSetZero(T* ptr, size_t size);

template<typename T>
void deviceFree(T*& ptr);

template<typename T>
void deviceFill(T* devptr, size_t size, T value, cudaStream_t stream = 0);

template<typename T>
void cudaD2Hcpy(T* tgt, const T* src, const size_t size);

template<typename T>
void cudaH2Dcpy(T* tgt, const T* src, const size_t size);

template<typename T>
void cudaD2Dcpy(T* tgt, const T* src, const size_t size);

template<typename T>
void cudaAutoCpy(T* tgt, const T* src, const size_t size, cudaStream_t stream = NULL);

template<typename T>
void cudaRandomUniform(T* buffer, const size_t size);

template<typename T>
int loadWeightFromBin(T*                  ptr,
                      std::vector<size_t> shape,
                      std::string         filename,
                      FtCudaDataType      model_file_type = FtCudaDataType::FP32);

std::vector<float> loadArrayFromBin(std::vector<size_t> shape, std::string filename);

// template<typename T>
// int loadWeightFromBinAndQuantizeForWeightOnly(int8_t*             quantized_weight_ptr,
//                                               T*                  scale_ptr,
//                                               std::vector<size_t> shape,
//                                               std::string         filename,
//                                               FtCudaDataType      model_file_type = FtCudaDataType::FP32);

void invokeCudaD2DcpyHalf2Float(float* dst, half* src, const size_t size, cudaStream_t stream);
void invokeCudaD2DcpyFloat2Half(half* dst, float* src, const size_t size, cudaStream_t stream);
#ifdef ENABLE_FP8
void invokeCudaD2Dcpyfp82Float(float* dst, __nv_fp8_e4m3* src, const size_t size, cudaStream_t stream);
void invokeCudaD2Dcpyfp82Half(half* dst, __nv_fp8_e4m3* src, const size_t size, cudaStream_t stream);
void invokeCudaD2DcpyFloat2fp8(__nv_fp8_e4m3* dst, float* src, const size_t size, cudaStream_t stream);
void invokeCudaD2DcpyHalf2fp8(__nv_fp8_e4m3* dst, half* src, const size_t size, cudaStream_t stream);
void invokeCudaD2DcpyBfloat2fp8(__nv_fp8_e4m3* dst, __nv_bfloat16* src, const size_t size, cudaStream_t stream);
#endif  // ENABLE_FP8
#ifdef ENABLE_BF16
void invokeCudaD2DcpyBfloat2Float(float* dst, __nv_bfloat16* src, const size_t size, cudaStream_t stream);
#endif  // ENABLE_BF16

template<typename T_OUT, typename T_IN>
void invokeCudaCast(T_OUT* dst, T_IN const* const src, const size_t size, cudaStream_t stream);

template<typename T, size_t n_dims>
__inline__ __host__ __device__ size_t dim2flat(const T (&idx)[n_dims], const T (&dims)[n_dims])
{
    size_t flat_idx = 0;
    for (size_t i = 0; i < n_dims; i++) {
        flat_idx += idx[i];
        if (i + 1 < n_dims)
            flat_idx *= dims[i + 1];
    }
    return flat_idx;
}

template<typename T1, size_t n_dims, typename T2>
__inline__ __host__ __device__ void flat2dim(T1 flat_idx, const T2 (&dims)[n_dims], T2 (&idx)[n_dims])
{
    for (int i = n_dims - 1; i >= 0; i--) {
        idx[i] = flat_idx % dims[i];
        flat_idx /= dims[i];
    }
}

template<typename T>
void invokeInPlaceTranspose(T* data, T* workspace, const int dim0, const int dim1);

template<typename T>
void invokeInPlaceTranspose0213(T* data, T* workspace, const int dim0, const int dim1, const int dim2, const int dim3);

template<typename T>
void invokeInPlaceTranspose102(T* data, T* workspace, const int dim0, const int dim1, const int dim2);

template<typename T>
void invokeMultiplyScale(T* tensor, float scale, const size_t size, cudaStream_t stream);

template<typename T>
void invokeDivideScale(T* tensor, float scale, const size_t size, cudaStream_t stream);

template<typename T_IN, typename T_OUT>
void invokeCudaD2DcpyConvert(T_OUT* tgt, const T_IN* src, const size_t size, cudaStream_t stream = 0);

template<typename T_IN, typename T_OUT>
void invokeCudaD2DScaleCpyConvert(
    T_OUT* tgt, const T_IN* src, const float* scale, bool invert_scale, const size_t size, cudaStream_t stream = 0);

inline bool checkIfFileExist(const std::string& file_path)
{
    std::ifstream in(file_path, std::ios::in | std::ios::binary);
    if (in.is_open()) {
        in.close();
        return true;
    }
    return false;
}

template<typename T>
void saveToBinary(const T* ptr, const size_t size, std::string filename);

template<typename T_IN, typename T_fake_type>
void invokeFakeCast(T_IN* input_ptr, const size_t size, cudaStream_t stream);

size_t cuda_datatype_size(FtCudaDataType dt);

template<typename T>
bool invokeCheckRange(T* buffer, const size_t size, T min, T max, bool* d_within_range, cudaStream_t stream);

}  // namespace turbomind
