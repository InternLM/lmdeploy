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

#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "src/turbomind/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "src/turbomind/kernels/matrix_vector_multiplication.h"
#include "src/turbomind/kernels/moe_kernels.h"
#include "src/turbomind/layers/BaseLayer.h"
#include "src/turbomind/layers/FfnWeight.h"
#include "src/turbomind/utils/activation_types.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"
#include <stdint.h>
#include <vector>

namespace turbomind {

template<typename T>
class FfnLayer: public BaseLayer {
private:
    // buffer handling
    size_t max_token_num_ = 0;

    // meta data
    size_t head_num_;       // (martinma): this member is not used in this class. Remove it?
    size_t size_per_head_;  // (martinma): this member is not used in this class. Remove it?
    size_t expert_num_;

    // calculated data
    size_t hidden_units_;

    // gated activation
    bool use_gated_activation_;

    std::shared_ptr<CutlassMoeFCRunner<T, T>>       moe_fc_runner_;
    std::shared_ptr<CutlassMoeFCRunner<T, uint8_t>> moe_int8_weight_only_fc_runner_;

    std::shared_ptr<CutlassFpAIntBGemmRunner<T, uint8_t>> weight_only_int8_fc_runner_;
    std::shared_ptr<CutlassInt8GemmRunner<T>>             int8_fc_runner_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(int moe_k = 0, bool use_moe = false);
    void allocateBuffer(size_t token_num, int moe_k = 0, bool use_moe = false);

protected:
    T*    inter_buf_        = nullptr;
    T*    inter_buf_2_      = nullptr;  // for gated activation
    T*    moe_gates_buf_    = nullptr;
    char* moe_fc_workspace_ = nullptr;

    char*  mixed_gemm_workspace_ = nullptr;
    size_t mixed_gemm_ws_bytes_  = 0;
    char*  int8_gemm_workspace_  = nullptr;
    size_t int8_gemm_ws_bytes_   = 0;

    size_t inter_size_;
    /* used to allocater memory buffers
       different ffn layers (inter_size) will
       reuse the same ffn layer with the max inter size.
       max_inter_size will be passed as inter_size when initializing the ffn layer
    */
    size_t max_inter_size_;

    // int8_mode_ == 0 means we don't use any mechanism related to INT8.
    // int8_mode_ == 1 for weight quantized only gemm for GPT
    // int8_mode_ == 2 for SmoothQuant O3 (per tensor scales)
    int int8_mode_ = 0;

    virtual ActivationType getActivationType() const
    {
        return ActivationType::InvalidType;
    };

    void genericActivation(int          m,
                           const T*     bias1,
                           const T*     bias2,
                           const int*   ia3_tasks,
                           const T*     ia3_weights,
                           const float* activation_in,
                           const float* activation_out,
                           const int*   padding_offset,
                           const int    seq_len);

public:
    FfnLayer(size_t           max_batch_size,
             size_t           max_seq_len,
             size_t           head_num,       // (martinma): redundant parameter?
             size_t           size_per_head,  // (martinma): redundant parameter?
             size_t           expert_num,
             size_t           inter_size,
             cudaStream_t     stream,
             cublasMMWrapper* cublas_wrapper,
             IAllocator*      allocator,
             bool             is_free_buffer_after_forward,
             bool             sparse               = false,
             int              int8_mode            = 0,
             bool             use_gated_activation = false);

    FfnLayer(FfnLayer<T> const& ffn_layer);

    virtual ~FfnLayer();

    void resetInterSize(size_t runtime_inter_size)
    {
        inter_size_ = runtime_inter_size;
    }

    virtual void forward(std::vector<turbomind::Tensor>*       output_tensors,
                         const std::vector<turbomind::Tensor>* input_tensors,
                         const FfnWeight<T>*                           ffn_weights);
    virtual void forward(TensorMap* output_tensors, TensorMap* input_tensors, const FfnWeight<T>* ffn_weights);
};

template<typename T>
class GeluFfnLayer: public FfnLayer<T> {
public:
    GeluFfnLayer(size_t           max_batch_size,
                 size_t           max_seq_len,
                 size_t           head_num,
                 size_t           size_per_head,
                 size_t           expert_num,
                 size_t           inter_size,
                 cudaStream_t     stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator*      allocator,
                 bool             is_free_buffer_after_forward,
                 bool             sparse               = false,
                 int              int8_mode            = 0,
                 bool             use_gated_activation = false);

    GeluFfnLayer(GeluFfnLayer<T> const& ffn_layer);

    virtual ~GeluFfnLayer() = default;

protected:
    using FfnLayer<T>::stream_;
    virtual ActivationType getActivationType() const override
    {
        return ActivationType::Gelu;
    };

private:
    using FfnLayer<T>::inter_buf_;
    using FfnLayer<T>::inter_buf_2_;
    using FfnLayer<T>::inter_size_;
};

template<typename T>
class ReluFfnLayer: public FfnLayer<T> {
public:
    ReluFfnLayer(size_t           max_batch_size,
                 size_t           max_seq_len,
                 size_t           head_num,
                 size_t           size_per_head,
                 size_t           expert_num,
                 size_t           inter_size,
                 cudaStream_t     stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator*      allocator,
                 bool             is_free_buffer_after_forward,
                 bool             sparse               = false,
                 int              int8_mode            = 0,
                 bool             use_gated_activation = false);

    ReluFfnLayer(ReluFfnLayer<T> const& ffn_layer);

    virtual ~ReluFfnLayer() = default;

protected:
    using FfnLayer<T>::stream_;
    virtual ActivationType getActivationType() const override
    {
        return ActivationType::Relu;
    };

private:
    using FfnLayer<T>::inter_buf_;
    using FfnLayer<T>::inter_buf_2_;
    using FfnLayer<T>::inter_size_;
};

template<typename T>
class SiluFfnLayer: public FfnLayer<T> {
public:
    SiluFfnLayer(size_t           max_batch_size,
                 size_t           max_seq_len,
                 size_t           head_num,
                 size_t           size_per_head,
                 size_t           expert_num,
                 size_t           inter_size,
                 cudaStream_t     stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator*      allocator,
                 bool             is_free_buffer_after_forward,
                 bool             sparse               = false,
                 bool             use_gated_activation = false);

    SiluFfnLayer(SiluFfnLayer<T> const& ffn_layer);

    virtual ~SiluFfnLayer() = default;

protected:
    using FfnLayer<T>::stream_;
    virtual ActivationType getActivationType() const override
    {
        return ActivationType::Silu;
    };

private:
    using FfnLayer<T>::inter_buf_;
    using FfnLayer<T>::inter_buf_2_;
    using FfnLayer<T>::inter_size_;
};

}  // namespace turbomind
