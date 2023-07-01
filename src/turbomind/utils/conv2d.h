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

#include "cublasLt.h"
#include "cuda_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cudnn.h>

namespace turbomind {

template<typename T>
void conv2d(T*             output,
            const T*       input,
            const T*       kernel,
            const int      batch,
            const int      h,
            const int      w,
            const int      in_channels,
            const int      out_channels,
            const int      kernel_size,
            const int      stride,
            cudnnHandle_t& cudnn_handle)
{
    cudnnDataType_t dataType;
    cudnnDataType_t computeType = CUDNN_DATA_FLOAT;
    float           alpha       = 1.0f;
    float           beta        = 0.0f;
    if (std::is_same<T, half>::value) {
        dataType = CUDNN_DATA_HALF;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        dataType = CUDNN_DATA_BFLOAT16;
    }
#endif
    else {
        dataType = CUDNN_DATA_FLOAT;
    }

    cudnnTensorDescriptor_t      input_descriptor_;
    cudnnTensorDescriptor_t      output_descriptor_;
    cudnnFilterDescriptor_t      kernel_descriptor_;
    cudnnConvolutionDescriptor_t convolution_descriptor_;
    cudnnConvolutionFwdAlgo_t    convolution_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    // cudnnConvolutionFwdAlgo_t convolution_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // cudnnConvolutionFwdAlgo_t convolution_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
    // cudnnConvolutionFwdAlgo_t convolution_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
    // cudnnConvolutionFwdAlgo_t convolution_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
    // cudnnConvolutionFwdAlgo_t convolution_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    // cudnnConvolutionFwdAlgo_t convolution_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    // cudnnConvolutionFwdAlgo_t convolution_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;

    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*dataType=*/dataType,
                                          /*batch_size=*/batch,
                                          /*channels=*/in_channels,
                                          /*image_height=*/h,
                                          /*image_width=*/w));

    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/dataType,
                                          /*batch_size=*/batch,
                                          /*channels=*/out_channels,
                                          /*image_height=*/h / stride,
                                          /*image_width=*/w / stride));

    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_,
                                          /*dataType=*/dataType,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*out_channels=*/out_channels,
                                          /*in_channels=*/in_channels,
                                          /*kernel_height=*/kernel_size,
                                          /*kernel_width=*/kernel_size));

    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_,
                                               /*pad_height=*/0,
                                               /*pad_width=*/0,
                                               /*vertical_stride=*/stride,
                                               /*horizontal_stride=*/stride,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*//*CUDNN_CONVOLUTION,*/ CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/computeType));

    /*checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
                                                   input_descriptor_,
                                                   kernel_descriptor_,
                                                   convolution_descriptor_,
                                                   output_descriptor_,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   0,//memoryLimitInBytes
                                                   &convolution_algorithm_));*/

    checkCUDNN(cudnnConvolutionForward(cudnn_handle,
                                       &alpha,
                                       input_descriptor_,
                                       input,
                                       kernel_descriptor_,
                                       kernel,
                                       convolution_descriptor_,
                                       convolution_algorithm_,
                                       nullptr,
                                       0,
                                       &beta,
                                       output_descriptor_,
                                       output));

    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor_));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor_));
    checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor_));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor_));
}

}  // namespace turbomind
