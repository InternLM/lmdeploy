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

#pragma once

#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/FfnFP8Weight.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <vector>

namespace fastertransformer {

template<typename T1, typename T2>
class FfnFP8Layer: public BaseLayer {
private:
    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(size_t token_num);

protected:
    const int    fp8_mode_;
    T1*          inter_buf_      = nullptr;
    T2*          inter_buf_bf16_ = nullptr;
    size_t       inter_size_;
    virtual void invokeAddBiasActivation(const int    m,
                                         const T2*    bias,
                                         const float* input_scale,
                                         const float* input_scale_2,
                                         const float* input_scale_2_min,
                                         const float* output_scale) = 0;

public:
    FfnFP8Layer(size_t           inter_size,
                int              fp8_mode,
                cudaStream_t     stream,
                cublasMMWrapper* cublas_wrapper,
                IAllocator*      allocator,
                bool             is_free_buffer_after_forward,
                bool             sparse = false);

    FfnFP8Layer(FfnFP8Layer<T1, T2> const& ffn_layer);

    virtual ~FfnFP8Layer();

    virtual void forward(TensorMap* output_tensors, TensorMap* input_tensors, const FfnFP8Weight<T1, T2>* ffn_weights);
    virtual ActivationType getActivationType() = 0;
};

template<typename T1, typename T2>
class GeluFfnFP8Layer: public FfnFP8Layer<T1, T2> {
public:
    GeluFfnFP8Layer(size_t           inter_size,
                    int              fp8_mode_,
                    cudaStream_t     stream,
                    cublasMMWrapper* cublas_wrapper,
                    IAllocator*      allocator,
                    bool             is_free_buffer_after_forward,
                    bool             sparse = false);

    GeluFfnFP8Layer(GeluFfnFP8Layer<T1, T2> const& ffn_layer);

    virtual ~GeluFfnFP8Layer() = default;
    ActivationType getActivationType() override
    {
        return ActivationType::Gelu;
    };

protected:
    using FfnFP8Layer<T1, T2>::stream_;

private:
    using FfnFP8Layer<T1, T2>::inter_buf_;
    using FfnFP8Layer<T1, T2>::inter_size_;
    using FfnFP8Layer<T1, T2>::fp8_mode_;
    using FfnFP8Layer<T1, T2>::inter_buf_bf16_;
    void invokeAddBiasActivation(const int    m,
                                 const T2*    bias,
                                 const float* input_scale,
                                 const float* input_scale_2,
                                 const float* input_scale_2_min,
                                 const float* output_scale) override;
};

template<typename T1, typename T2>
class ReluFfnFP8Layer: public FfnFP8Layer<T1, T2> {
public:
    ReluFfnFP8Layer(size_t           inter_size,
                    int              fp8_mode,
                    cudaStream_t     stream,
                    cublasMMWrapper* cublas_wrapper,
                    IAllocator*      allocator,
                    bool             is_free_buffer_after_forward,
                    bool             sparse = false);

    ReluFfnFP8Layer(ReluFfnFP8Layer<T1, T2> const& ffn_layer);

    virtual ~ReluFfnFP8Layer() = default;
    ActivationType getActivationType() override
    {
        return ActivationType::Relu;
    };

protected:
    using FfnFP8Layer<T1, T2>::stream_;

private:
    using FfnFP8Layer<T1, T2>::inter_buf_;
    using FfnFP8Layer<T1, T2>::inter_size_;
    using FfnFP8Layer<T1, T2>::fp8_mode_;
    using FfnFP8Layer<T1, T2>::inter_buf_bf16_;
    void invokeAddBiasActivation(const int    m,
                                 const T2*    bias,
                                 const float* input_scale,
                                 const float* input_scale_2,
                                 const float* input_scale_2_min,
                                 const float* output_scale) override;
};

}  // namespace fastertransformer
