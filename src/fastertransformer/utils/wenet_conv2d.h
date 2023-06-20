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

namespace fastertransformer {

template<typename T>
void conv2d(T*             output,
            T*             input,
            void*          ws_data,
            const int      index,
            const T*       kernel,
            const T*       bias,
            const int      batch,
            const int      h,
            const int      w,
            const int      in_channels,
            const int      out_channels,
            const int      kernel_size,
            const int      stride,
            cudnnHandle_t& cudnn_handle,
            cudaStream_t   stream)
{
    cudnnDataType_t dataType;
    cudnnDataType_t computeType = CUDNN_DATA_FLOAT;
    const float     alpha1      = 1.0f;
    const float     alpha2      = 0.0f;
    if (std::is_same<T, half>::value) {
        dataType = CUDNN_DATA_HALF;
        // computeType = CUDNN_DATA_HALF;
    }
    else {
        dataType = CUDNN_DATA_FLOAT;
    }

    cudnnTensorDescriptor_t      input_descriptor_;
    cudnnTensorDescriptor_t      output_descriptor_;
    cudnnFilterDescriptor_t      kernel_descriptor_;
    cudnnTensorDescriptor_t      bias_descriptor_;
    cudnnConvolutionDescriptor_t convolution_descriptor_;
    cudnnActivationDescriptor_t  activation_descriptor_;
    cudnnConvolutionFwdAlgo_t    convolution_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor_,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/dataType,
                                          /*batch_size=*/batch,
                                          /*channels=*/in_channels,
                                          /*image_height=*/h,
                                          /*image_width=*/w));

    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor_));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor_,
                                          /*dataType=*/dataType,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*out_channels=*/out_channels,
                                          /*in_channels=*/in_channels,
                                          /*kernel_height=*/kernel_size,
                                          /*kernel_width=*/kernel_size));

    checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor_));
    checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor_,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/dataType,
                                          /*batch_size=*/1,
                                          /*channels=*/out_channels,
                                          /*image_height=*/1,
                                          /*image_width=*/1));

    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor_));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor_,
                                               /*pad_height=*/0,
                                               /*pad_width=*/0,
                                               /*vertical_stride=*/stride,
                                               /*horizontal_stride=*/stride,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*/CUDNN_CROSS_CORRELATION, /*CUDNN_CONVOLUTION,*/
                                               /*computeType=*/computeType));
    checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor_, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor_));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor_,
                                            /*mode=*/CUDNN_ACTIVATION_RELU,
                                            // /*mode=*/CUDNN_ACTIVATION_IDENTITY,
                                            /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                                            /*coef=*/0.0));

    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor_));

    // output
    int out_n;
    int out_c;
    int out_h;
    int out_w;

    // TODO: set the second conv as nhwc in and nchw out
    cudnnTensorFormat_t output_format = index == 0 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        convolution_descriptor_, input_descriptor_, kernel_descriptor_, &out_n, &out_c, &out_h, &out_w));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor_, output_format, dataType, out_n, out_c, out_h, out_w));

    // search algorithm, we use default directly to prevent the overhead to choose bset one
    // int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    // int returnedAlgoCount = -1;
    // cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

    // // Choose the best according to the preference
    // checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle,
    //                                                   input_descriptor_,
    //                                                   kernel_descriptor_,
    //                                                   convolution_descriptor_,
    //                                                   output_descriptor_,
    //                                                   requestedAlgoCount,
    //                                                   &returnedAlgoCount,
    //                                                   results));
    // for (int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex) {
    //     printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n",
    //            cudnnGetErrorString(results[algoIndex].status),
    //            results[algoIndex].algo,
    //            results[algoIndex].time,
    //            (unsigned long long)results[algoIndex].memory);
    // }
    // convolution_algorithm_ = results[0].algo;

    // workspace
    size_t ws_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                       input_descriptor_,
                                                       kernel_descriptor_,
                                                       convolution_descriptor_,
                                                       output_descriptor_,
                                                       convolution_algorithm_,
                                                       &ws_size));
    FT_LOG_DEBUG("Convolution algorithm: %d with workspace size: %d \n", convolution_algorithm_, ws_size);
    FT_CHECK_WITH_INFO(
        ws_size <= (1 << 29),
        "Current workspace used for CuDNN Convolution is fixed as 1 << 29, please increase it in WenetEncoder::allocateBuffer!");
    // void *ws_data;
    // if (ws_size > 0) {
    //     check_cuda_error(cudaMalloc(&ws_data, ws_size));
    // }
    // else{
    //     ws_data = nullptr;
    // }

    sync_check_cuda_error();
    checkCUDNN(cudnnConvolutionBiasActivationForward(cudnn_handle,
                                                     (void*)(&alpha1),
                                                     input_descriptor_,
                                                     input,
                                                     kernel_descriptor_,
                                                     kernel,
                                                     convolution_descriptor_,
                                                     convolution_algorithm_,
                                                     (void*)ws_data,
                                                     ws_size,
                                                     (void*)(&alpha2),
                                                     output_descriptor_,
                                                     output,
                                                     bias_descriptor_,
                                                     bias,
                                                     activation_descriptor_,
                                                     output_descriptor_,
                                                     output));

    sync_check_cuda_error();
    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor_));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor_));
    checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor_));
    checkCUDNN(cudnnDestroyTensorDescriptor(bias_descriptor_));
    checkCUDNN(cudnnDestroyActivationDescriptor(activation_descriptor_));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor_));
}

}  // namespace fastertransformer
