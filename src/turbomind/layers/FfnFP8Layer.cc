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

#include "src/turbomind/layers/FfnFP8Layer.h"
#include "src/turbomind/kernels/activation_fp8_kernels.h"
#include "src/turbomind/utils/cublasFP8MMWrapper.h"
#include "src/turbomind/utils/nvtx_utils.h"

namespace turbomind {

template<typename T1, typename T2>
void FfnFP8Layer<T1, T2>::forward(TensorMap*                  output_tensors,
                                  TensorMap*                  input_tensors,
                                  const FfnFP8Weight<T1, T2>* ffn_weights)
{
    // input tensors:
    //      input_hidden_state [token_num, d_model],

    // output tensors:
    //      output_hidden_state [token_num, d_model],

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 1);
    FT_CHECK(output_tensors->size() == 1);

    const int m                  = input_tensors->at("input_hidden_state").shape[0];
    const int d_model            = input_tensors->at("input_hidden_state").shape[1];
    const T1* input_hidden_state = input_tensors->at("input_hidden_state").getPtr<T1>();
    Tensor    output_tensor      = output_tensors->at("output_hidden_state");
    allocateBuffer(m);

#ifdef FUSE_GEMM_ACT
    if (fp8_mode_ == 1) {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
            ->Gemm(inter_buf_bf16_,
                   (int)1,
                   (int)m,
                   (int)inter_size_,
                   (int)d_model,
                   (int64_t)0,
                   (int64_t)0,
                   (int64_t)0,
                   &alpha,
                   &beta,
                   input_hidden_state,
                   ffn_weights->intermediate_weight.kernel,
                   ffn_weights->intermediate_weight.input_scale,
                   ffn_weights->intermediate_weight.per_channel_scale_min,  // identity_scale
                   stream_);
        invokeAddBiasActivation(m,
                                ffn_weights->intermediate_weight.bias,
                                ffn_weights->intermediate_weight.output_scale,
                                ffn_weights->intermediate_weight.scale,
                                ffn_weights->intermediate_weight.per_channel_scale_min,
                                ffn_weights->output_weight.input_scale_inv);
    }
    else if (fp8_mode_ == 2) {
#ifdef USE_QGMMA
        if (getActivationType() == ActivationType::Gelu) {
            PUSH_RANGE("FFN gemm 1 bias gelu");
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                ->Conv1x1Gemm<false, true>(inter_buf_,
                                           m,
                                           inter_size_,
                                           d_model,
                                           input_hidden_state,
                                           ffn_weights->intermediate_weight.kernel,
                                           ffn_weights->intermediate_weight.bias,
                                           *(ffn_weights->intermediate_weight.input_h_scale),   // scale_a,
                                           *(ffn_weights->intermediate_weight.weight_h_scale),  // scale_b,
                                           *(ffn_weights->output_weight.input_h_scale_inv),     // scale_d,
                                           stream_);
            POP_RANGE;
        }
        else if (getActivationType() == ActivationType::Relu) {
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                ->Conv1x1Gemm<true, false>(inter_buf_,
                                           m,
                                           inter_size_,
                                           d_model,
                                           input_hidden_state,
                                           ffn_weights->intermediate_weight.kernel,
                                           ffn_weights->intermediate_weight.bias,
                                           *(ffn_weights->intermediate_weight.input_h_scale),   // scale_a,
                                           *(ffn_weights->intermediate_weight.weight_h_scale),  // scale_b,
                                           *(ffn_weights->output_weight.input_h_scale_inv),     // scale_d,
                                           stream_);
        }
#else  // USE_QGMMA
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        if (getActivationType() == ActivationType::Gelu) {
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
#ifdef FP8_GEMM_OUTPUT_QUANT_DISABLE
                ->Gemm_Bias_Act<false, true>(inter_buf_bf16_,
#else   // FP8_GEMM_OUTPUT_QUANT_DISABLE
                ->Gemm_Bias_Act<false, true>(inter_buf_,
#endif  // FP8_GEMM_OUTPUT_QUANT_DISABLE
                                             (int)1,
                                             (int)m,
                                             (int)inter_size_,
                                             (int)d_model,
                                             (int64_t)0,
                                             (int64_t)0,
                                             (int64_t)0,
                                             &alpha,
                                             &beta,
                                             input_hidden_state,
                                             ffn_weights->intermediate_weight.kernel,
                                             ffn_weights->intermediate_weight.input_scale,
                                             ffn_weights->intermediate_weight.weight_scale,
                                             ffn_weights->intermediate_weight.bias,
                                             ffn_weights->intermediate_weight.output_scale,
                                             stream_);
        }
        else if (getActivationType() == ActivationType::Relu) {
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
#ifdef FP8_GEMM_OUTPUT_QUANT_DISABLE
                ->Gemm_Bias_Act<true, false>(inter_buf_bf16_,
#else   // FP8_GEMM_OUTPUT_QUANT_DISABLE
                ->Gemm_Bias_Act<true, false>(inter_buf_,
#endif  // #ifdef FP8_GEMM_OUTPUT_QUANT_DISABLE
                                             (int)1,
                                             (int)m,
                                             (int)inter_size_,
                                             (int)d_model,
                                             (int64_t)0,
                                             (int64_t)0,
                                             (int64_t)0,
                                             &alpha,
                                             &beta,
                                             input_hidden_state,
                                             ffn_weights->intermediate_weight.kernel,
                                             ffn_weights->intermediate_weight.input_scale,
                                             ffn_weights->intermediate_weight.weight_scale,
                                             ffn_weights->intermediate_weight.bias,
                                             ffn_weights->intermediate_weight.output_scale,
                                             stream_);
        }
#ifdef FP8_GEMM_OUTPUT_QUANT_DISABLE
        invokeQuantizeMatrix<T1, T2, QUANTIZE_MODE::PER_TENSOR>(
            inter_buf_, ffn_weights->output_weight.input_scale_inv, inter_buf_bf16_, m * inter_size_, 1, stream_);
#endif FP8_GEMM_OUTPUT_QUANT_DISABLE
#endif  // USE_QGMMA
    }

#else  // FUSE_GEMM_ACT
    PUSH_RANGE("FFN gemm 1");
#ifdef SPARSITY_ENABLED
    int m_tmp = m;
    if (m_tmp % 8 != 0) {
        m_tmp = (m_tmp / 8 + 1) * 8;
    }
    const int m_padded = m_tmp;
    if (sparse_ && cublas_wrapper_->isUseSparse(1, inter_size_, m, d_model)) {
        FT_CHECK(false);
        // cublas_wrapper_->SpGemm(CUBLAS_OP_N,
        //                         CUBLAS_OP_N,
        //                         inter_size_,
        //                         m_padded,
        //                         d_model,
        //                         ffn_weights->intermediate_weight.sp_kernel,
        //                         input_hidden_state,
        //                         inter_buf_);
    }
    else {
#endif  // SPARSITY_ENABLED
        if (fp8_mode_ == 1) {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                ->Gemm(inter_buf_bf16_,
                       (int)1,
                       (int)m,
                       (int)inter_size_,
                       (int)d_model,
                       (int64_t)0,
                       (int64_t)0,
                       (int64_t)0,
                       &alpha,
                       &beta,
                       input_hidden_state,
                       ffn_weights->intermediate_weight.kernel,
                       ffn_weights->intermediate_weight.input_scale,
                       ffn_weights->intermediate_weight.per_channel_scale_min,  // identity_scale
                       stream_);
        }
        else if (fp8_mode_ == 2) {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                ->Gemm(inter_buf_bf16_,
                       (int)1,
                       (int)m,
                       (int)inter_size_,
                       (int)d_model,
                       (int64_t)0,
                       (int64_t)0,
                       (int64_t)0,
                       &alpha,
                       &beta,
                       input_hidden_state,
                       ffn_weights->intermediate_weight.kernel,
                       ffn_weights->intermediate_weight.input_scale,
                       ffn_weights->intermediate_weight.weight_scale,
                       stream_);
        }
#ifdef SPARSITY_ENABLED
    }
#endif  // SPARSITY_ENABLED
    POP_RANGE;

    PUSH_RANGE("FFN add bias act");
    if (fp8_mode_ == 1) {
        invokeAddBiasActivation(m,
                                ffn_weights->intermediate_weight.bias,
                                ffn_weights->intermediate_weight.output_scale,
                                ffn_weights->intermediate_weight.scale,
                                ffn_weights->intermediate_weight.per_channel_scale_min,
                                ffn_weights->output_weight.input_scale_inv);
    }
    else if (fp8_mode_ == 2) {
        invokeAddBiasActivation(m,
                                ffn_weights->intermediate_weight.bias,
                                ffn_weights->intermediate_weight.output_scale,
                                nullptr,
                                nullptr,
                                ffn_weights->output_weight.input_scale_inv);
    }
    sync_check_cuda_error();
    POP_RANGE;
#endif  // FUSE_GEMM_ACT

    PUSH_RANGE("FFN gemm 2");
#ifdef SPARSITY_ENABLED
    if (sparse_ && cublas_wrapper_->isUseSparse(1, d_model, m, inter_size_)) {
        FT_CHECK(false);
        // cublas_wrapper_->SpGemm(CUBLAS_OP_N,
        //                         CUBLAS_OP_N,
        //                         d_model,
        //                         m_padded,
        //                         inter_size_,
        //                         ffn_weights->output_weight.sp_kernel,
        //                         inter_buf_,
        //                         output_tensor);
    }
    else {
#endif SPARSITY_ENABLED
        if (fp8_mode_ == 1) {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            if (output_tensor.type == TYPE_BF16) {
                reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                    ->Gemm(output_tensor.getPtr<T2>(),
                           (int)1,
                           (int)m,
                           (int)d_model,
                           (int)inter_size_,
                           (int64_t)0,
                           (int64_t)0,
                           (int64_t)0,
                           &alpha,
                           &beta,
                           (const __nv_fp8_e4m3*)inter_buf_,
                           (const __nv_fp8_e4m3*)ffn_weights->output_weight.kernel,
                           ffn_weights->output_weight.input_scale,
                           ffn_weights->identity_scale,
                           stream_);
            }
            else if (output_tensor.type == TYPE_FP8_E4M3) {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                    ->Gemm(output_tensor.getPtr<T1>(),
                           (int)1,
                           (int)m,
                           (int)d_model,
                           (int)inter_size_,
                           (int64_t)0,
                           (int64_t)0,
                           (int64_t)0,
                           &alpha,
                           &beta,
                           (const __nv_fp8_e4m3*)inter_buf_,
                           (const __nv_fp8_e4m3*)ffn_weights->output_weight.kernel,
                           ffn_weights->output_weight.input_scale,
                           ffn_weights->output_weight.per_channel_scale_min,
                           ffn_weights->output_weight.output_scale_inv,
                           stream_);
            }
            else {
                FT_CHECK(false);
            }
        }
        else if (fp8_mode_ == 2) {
            if (output_tensor.type == TYPE_BF16) {
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                    ->Gemm(output_tensor.getPtr<T2>(),
                           (int)1,
                           (int)m,
                           (int)d_model,
                           (int)inter_size_,
                           (int64_t)0,
                           (int64_t)0,
                           (int64_t)0,
                           &alpha,
                           &beta,
                           (const __nv_fp8_e4m3*)inter_buf_,
                           (const __nv_fp8_e4m3*)ffn_weights->output_weight.kernel,
                           ffn_weights->output_weight.input_scale,
                           ffn_weights->output_weight.weight_scale,
                           stream_);
            }
            else if (output_tensor.type == TYPE_FP8_E4M3) {
                // It looks like conv1x1Gemm does not bring better performance for this gemm
                // because the k dimension of this gemm is large
                // #ifdef USE_QGMMA
                //                 reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                //                     ->Conv1x1Gemm<false, false>(output_tensor.getPtr<T1>(),
                //                                                 m,
                //                                                 d_model,
                //                                                 inter_size_,
                //                                                 inter_buf_,
                //                                                 ffn_weights->output_weight.kernel,
                //                                                 ffn_weights->output_weight.bias,
                //                                                 *(ffn_weights->output_weight.input_h_scale),       //
                //                                                 scale_a,
                //                                                 *(ffn_weights->output_weight.weight_h_scale),      //
                //                                                 scale_b,
                //                                                 *(ffn_weights->output_weight.output_h_scale_inv),  //
                //                                                 scale_d, stream_);
                // #else   // USE_QGMMA
                const float alpha = 1.0f;
                const float beta  = 0.0f;
                reinterpret_cast<cublasFP8MMWrapper*>(cublas_wrapper_)
                    ->Gemm(output_tensor.getPtr<T1>(),
                           (int)1,
                           (int)m,
                           (int)d_model,
                           (int)inter_size_,
                           (int64_t)0,
                           (int64_t)0,
                           (int64_t)0,
                           &alpha,
                           &beta,
                           (const __nv_fp8_e4m3*)inter_buf_,
                           (const __nv_fp8_e4m3*)ffn_weights->output_weight.kernel,
                           ffn_weights->output_weight.input_scale,
                           ffn_weights->output_weight.weight_scale,
                           ffn_weights->output_weight.output_scale_inv,
                           stream_);
                // #endif  // USE_QGMMA
            }
            else {
                FT_CHECK(false);
            }
        }
#ifdef SPARSITY_ENABLED
    }
#endif  // SPARSITY_ENABLED
    POP_RANGE;

    sync_check_cuda_error();
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T1, typename T2>
FfnFP8Layer<T1, T2>::FfnFP8Layer(size_t           inter_size,
                                 int              fp8_mode,
                                 cudaStream_t     stream,
                                 cublasMMWrapper* cublas_wrapper,
                                 IAllocator*      allocator,
                                 bool             is_free_buffer_after_forward,
                                 bool             sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    inter_size_(inter_size),
    fp8_mode_(fp8_mode)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T1, typename T2>
FfnFP8Layer<T1, T2>::FfnFP8Layer(FfnFP8Layer<T1, T2> const& ffn_layer):
    BaseLayer(ffn_layer.stream_,
              ffn_layer.cublas_wrapper_,
              ffn_layer.allocator_,
              ffn_layer.is_free_buffer_after_forward_,
              ffn_layer.cuda_device_prop_,
              ffn_layer.sparse_),
    inter_size_(ffn_layer.inter_size_),
    fp8_mode_(ffn_layer.fp8_mode_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T1, typename T2>
FfnFP8Layer<T1, T2>::~FfnFP8Layer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T1, typename T2>
void FfnFP8Layer<T1, T2>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T1, typename T2>
void FfnFP8Layer<T1, T2>::allocateBuffer(size_t token_num)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    inter_buf_          = (T1*)allocator_->reMalloc(inter_buf_, sizeof(T1) * token_num * inter_size_, false);
    inter_buf_bf16_     = (T2*)allocator_->reMalloc(inter_buf_bf16_, sizeof(T2) * token_num * inter_size_, false);
    is_allocate_buffer_ = true;
}

template<typename T1, typename T2>
void FfnFP8Layer<T1, T2>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&inter_buf_));
        allocator_->free((void**)(&inter_buf_bf16_));
        is_allocate_buffer_ = false;
    }
}

template class FfnFP8Layer<__nv_fp8_e4m3, __nv_bfloat16>;

template<typename T1, typename T2>
GeluFfnFP8Layer<T1, T2>::GeluFfnFP8Layer(size_t           inter_size,
                                         int              fp8_mode,
                                         cudaStream_t     stream,
                                         cublasMMWrapper* cublas_wrapper,
                                         IAllocator*      allocator,
                                         bool             is_free_buffer_after_forward,
                                         bool             sparse):
    FfnFP8Layer<T1, T2>(inter_size, fp8_mode, stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse)
{
}

template<typename T1, typename T2>
GeluFfnFP8Layer<T1, T2>::GeluFfnFP8Layer(GeluFfnFP8Layer<T1, T2> const& gelu_ffn_layer):
    FfnFP8Layer<T1, T2>(gelu_ffn_layer)
{
}

template<typename T1, typename T2>
void GeluFfnFP8Layer<T1, T2>::invokeAddBiasActivation(const int    m,
                                                      const T2*    bias,
                                                      const float* input_scale,
                                                      const float* input_scale_2,
                                                      const float* input_scale_2_min,
                                                      const float* output_scale)
{
    FP8ActivationParam<T1, T2> param{inter_buf_bf16_,
                                     inter_buf_,
                                     bias,
                                     input_scale,
                                     input_scale_2,
                                     input_scale_2_min,
                                     output_scale,
                                     (uint32_t)m,
                                     (uint32_t)inter_size_,
                                     stream_};
    invokeFP8AddBiasGelu<T1, T2>(param);
}

template class GeluFfnFP8Layer<__nv_fp8_e4m3, __nv_bfloat16>;

template<typename T1, typename T2>
ReluFfnFP8Layer<T1, T2>::ReluFfnFP8Layer(size_t           inter_size,
                                         int              fp8_mode,
                                         cudaStream_t     stream,
                                         cublasMMWrapper* cublas_wrapper,
                                         IAllocator*      allocator,
                                         bool             is_free_buffer_after_forward,
                                         bool             sparse):
    FfnFP8Layer<T1, T2>(inter_size, fp8_mode, stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse)
{
}

template<typename T1, typename T2>
ReluFfnFP8Layer<T1, T2>::ReluFfnFP8Layer(ReluFfnFP8Layer<T1, T2> const& relu_ffn_layer):
    FfnFP8Layer<T1, T2>(relu_ffn_layer)
{
}

template<typename T1, typename T2>
void ReluFfnFP8Layer<T1, T2>::invokeAddBiasActivation(const int    m,
                                                      const T2*    bias,
                                                      const float* input_scale,
                                                      const float* input_scale_2,
                                                      const float* input_scale_2_min,
                                                      const float* output_scale)
{
    FP8ActivationParam<T1, T2> param{inter_buf_bf16_,
                                     inter_buf_,
                                     bias,
                                     input_scale,
                                     input_scale_2,
                                     input_scale_2_min,
                                     output_scale,
                                     (uint32_t)m,
                                     (uint32_t)inter_size_,
                                     stream_};
    invokeFP8AddBiasRelu<T1, T2>(param);
}

template class ReluFfnFP8Layer<__nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace turbomind
