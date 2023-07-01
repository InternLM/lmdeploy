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

#include "FfnINT8Weight.h"
#include "src/turbomind/kernels/activation_int8_kernels.h"
#include "src/turbomind/layers/BaseLayer.h"
#include "src/turbomind/utils/ScaleList.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cublasINT8MMWrapper.h"
#include "src/turbomind/utils/memory_utils.h"
#include <vector>

namespace fastertransformer {

template<typename T>
class GeluFfnLayerINT8;

template<typename T>
class ReluFfnLayerINT8;

template<typename T>
class FfnLayerINT8: public BaseLayer {
private:
    // buffer handling
    size_t max_token_num_ = 0;

    // meta data
    size_t head_num_;
    size_t size_per_head_;

    // calculated data
    size_t hidden_units_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidTokenNum(size_t token_num);

protected:
    size_t inter_size_;
    int    int8_mode_;
    bool   sparse_;

    int*         inter_int_buf_;
    int8_t*      inter_buf_;
    virtual void invokeAddBiasActivation(const int m, const T* bias, ScaleList* scale_list) = 0;

public:
    FfnLayerINT8(size_t           max_batch_size,
                 size_t           max_seq_len,
                 size_t           head_num,
                 size_t           size_per_head,
                 size_t           inter_size,
                 int              int8_mode,
                 cudaStream_t     stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator*      allocator,
                 bool             is_free_buffer_after_forward,
                 bool             sparse = false);

    FfnLayerINT8(FfnLayerINT8<T> const& ffn_layer);

    ~FfnLayerINT8();

    void forward(std::vector<fastertransformer::Tensor>*       output_tensors,
                 const std::vector<fastertransformer::Tensor>* input_tensors,
                 const FfnWeight<T>*                           ffn_weights);

    friend GeluFfnLayerINT8<T>;
    friend ReluFfnLayerINT8<T>;
};

template<typename T>
class GeluFfnLayerINT8: public FfnLayerINT8<T> {
public:
    GeluFfnLayerINT8(size_t           max_batch_size,
                     size_t           max_seq_len,
                     size_t           head_num,
                     size_t           size_per_head,
                     size_t           inter_size,
                     int              int8_mode,
                     cudaStream_t     stream,
                     cublasMMWrapper* cublas_wrapper,
                     IAllocator*      allocator,
                     bool             is_free_buffer_after_forward,
                     bool             sparse = false);

    GeluFfnLayerINT8(GeluFfnLayerINT8<T> const& ffn_layer);

    ~GeluFfnLayerINT8() = default;

private:
    using FfnLayerINT8<T>::inter_int_buf_;
    using FfnLayerINT8<T>::inter_buf_;
    using FfnLayerINT8<T>::inter_size_;
    using FfnLayerINT8<T>::stream_;
    using FfnLayerINT8<T>::int8_mode_;
    using FfnLayerINT8<T>::sparse_;
    using FfnLayerINT8<T>::hidden_units_;
    void invokeAddBiasActivation(const int m, const T* bias, ScaleList* scale_list) override;
};

template<typename T>
class ReluFfnLayerINT8: public FfnLayerINT8<T> {
public:
    ReluFfnLayerINT8(size_t           max_batch_size,
                     size_t           max_seq_len,
                     size_t           head_num,
                     size_t           size_per_head,
                     size_t           inter_size,
                     int              int8_mode,
                     cudaStream_t     stream,
                     cublasMMWrapper* cublas_wrapper,
                     IAllocator*      allocator,
                     bool             is_free_buffer_after_forward);

    ReluFfnLayerINT8(ReluFfnLayerINT8<T> const& ffn_layer);

    ~ReluFfnLayerINT8() = default;

private:
    using FfnLayerINT8<T>::inter_int_buf_;
    using FfnLayerINT8<T>::inter_buf_;
    using FfnLayerINT8<T>::inter_size_;
    using FfnLayerINT8<T>::stream_;
    using FfnLayerINT8<T>::int8_mode_;
    using FfnLayerINT8<T>::hidden_units_;
    void invokeAddBiasActivation(const int m, const T* bias, ScaleList* scale_list) override;
};

}  // namespace fastertransformer
