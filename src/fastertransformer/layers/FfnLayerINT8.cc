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

#include "FfnLayerINT8.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer {

template<typename T>
void FfnLayerINT8<T>::forward(std::vector<fastertransformer::Tensor>*       output_tensors,
                              const std::vector<fastertransformer::Tensor>* input_tensors,
                              const FfnWeight<T>*                           ffn_weights)
{
    // input_tensors: [input (token_num, hidden_dimension)]
    // output_tensors: [output (token_num, hidden_dimension)]
    ScaleList* scale_list = ((const FfnINT8Weight<T>*)ffn_weights)->scale_list_ptr;

    cublasINT8MMWrapper* cublas_wrapper = (cublasINT8MMWrapper*)cublas_wrapper_;

    FT_CHECK(isValidTokenNum(input_tensors->at(0).shape[0]));
    allocateBuffer();

    const int m = static_cast<int>(input_tensors->at(0).shape[0]);
#ifdef SPARSITY_ENABLED
    int m_tmp = m;
    if (m_tmp % 16 != 0) {
        m_tmp = (m_tmp / 16 + 1) * 16;
    }
    const int m_padded = m_tmp;
#endif

    int32_t*      output_tensor = output_tensors->at(0).getPtr<int32_t>();
    const int8_t* input_tensor  = input_tensors->at(0).getPtr<const int8_t>();

    PUSH_RANGE("FFN gemm 1");
    if (int8_mode_ == 1) {
        cublas_wrapper->Gemm(inter_int_buf_,
                             1,
                             m,
                             inter_size_,
                             hidden_units_,
                             0,
                             0,
                             0,
                             input_tensor,
                             (int8_t*)(ffn_weights->intermediate_weight.kernel));
    }
    else if (int8_mode_ == 2 || int8_mode_ == 3) {
#ifdef SPARSITY_ENABLED
        if (sparse_) {
            cublas_wrapper->SpGemm(inter_size_,
                                   m_padded,
                                   hidden_units_,
                                   scale_list->h_scale_list_[scale_list->p3_offset_ + 6],
                                   (int8_t*)(ffn_weights->intermediate_weight.sp_kernel),
                                   input_tensor,
                                   (int8_t*)inter_int_buf_);
        }
        else {
#endif
            cublas_wrapper->Gemm((int8_t*)inter_int_buf_,
                                 1,
                                 m,
                                 inter_size_,
                                 hidden_units_,
                                 0,
                                 0,
                                 0,
                                 scale_list->h_scale_list_[scale_list->p3_offset_ + 6],
                                 input_tensor,
                                 (int8_t*)(ffn_weights->intermediate_weight.kernel));
#ifdef SPARSITY_ENABLED
        }
#endif
    }
    POP_RANGE;

    PUSH_RANGE("add bias act");
    invokeAddBiasActivation(m, ffn_weights->intermediate_weight.bias, scale_list);
    POP_RANGE;
    sync_check_cuda_error();

    PUSH_RANGE("FFN gemm 2");
    if (int8_mode_ == 1) {
        cublas_wrapper->Gemm(output_tensor,
                             1,
                             m,
                             hidden_units_,
                             inter_size_,
                             0,
                             0,
                             0,
                             inter_buf_,
                             (int8_t*)(ffn_weights->output_weight.kernel));
    }
    else if (int8_mode_ == 2 || int8_mode_ == 3) {
#ifdef SPARSITY_ENABLED
        if (sparse_) {
            cublas_wrapper->SpGemm(hidden_units_,
                                   m_padded,
                                   inter_size_,
                                   scale_list->h_scale_list_[scale_list->p3_offset_ + 7],
                                   (int8_t*)(ffn_weights->output_weight.sp_kernel),
                                   inter_buf_,
                                   (int8_t*)output_tensor);
        }
        else {
#endif
            cublas_wrapper->Gemm((int8_t*)output_tensor,
                                 1,
                                 m,
                                 hidden_units_,
                                 inter_size_,
                                 0,
                                 0,
                                 0,
                                 scale_list->h_scale_list_[scale_list->p3_offset_ + 7],
                                 inter_buf_,
                                 (int8_t*)(ffn_weights->output_weight.kernel));
#ifdef SPARSITY_ENABLED
        }
#endif
    }
    POP_RANGE;

    sync_check_cuda_error();
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
FfnLayerINT8<T>::FfnLayerINT8(size_t           max_batch_size,
                              size_t           max_seq_len,
                              size_t           head_num,
                              size_t           size_per_head,
                              size_t           inter_size,
                              int              int8_mode,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             sparse):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_token_num_(max_batch_size * max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    inter_size_(inter_size),
    int8_mode_(int8_mode),
    sparse_(sparse)
{
}

template<typename T>
FfnLayerINT8<T>::FfnLayerINT8(FfnLayerINT8<T> const& ffn_layer):
    BaseLayer(
        ffn_layer.stream_, ffn_layer.cublas_wrapper_, ffn_layer.allocator_, ffn_layer.is_free_buffer_after_forward_),
    max_token_num_(ffn_layer.max_token_num_),
    head_num_(ffn_layer.head_num_),
    size_per_head_(ffn_layer.size_per_head_),
    hidden_units_(ffn_layer.hidden_units_),
    inter_size_(ffn_layer.inter_size_),
    int8_mode_(ffn_layer.int8_mode_),
    sparse_(ffn_layer.sparse_)
{
}

template<typename T>
FfnLayerINT8<T>::~FfnLayerINT8()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void FfnLayerINT8<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        inter_int_buf_ =
            (int32_t*)allocator_->reMalloc(inter_int_buf_, sizeof(int32_t) * max_token_num_ * inter_size_, false);
        inter_buf_ = (int8_t*)allocator_->reMalloc(inter_buf_, sizeof(int8_t) * max_token_num_ * inter_size_, false);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void FfnLayerINT8<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free((void**)(&inter_int_buf_));
        allocator_->free((void**)(&inter_buf_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool FfnLayerINT8<T>::isValidTokenNum(size_t token_num)
{
    if (max_token_num_ == 0) {
        max_token_num_ = token_num;
        return true;
    }
    else {
        return token_num <= max_token_num_;
    }
}

template class FfnLayerINT8<float>;
template class FfnLayerINT8<half>;

template<typename T>
GeluFfnLayerINT8<T>::GeluFfnLayerINT8(size_t           max_batch_size,
                                      size_t           max_seq_len,
                                      size_t           head_num,
                                      size_t           size_per_head,
                                      size_t           inter_size,
                                      int              int8_mode,
                                      cudaStream_t     stream,
                                      cublasMMWrapper* cublas_wrapper,
                                      IAllocator*      allocator,
                                      bool             is_free_buffer_after_forward,
                                      bool             sparse):
    FfnLayerINT8<T>(max_batch_size,
                    max_seq_len,
                    head_num,
                    size_per_head,
                    inter_size,
                    int8_mode,
                    stream,
                    cublas_wrapper,
                    allocator,
                    is_free_buffer_after_forward,
                    sparse)
{
}

template<typename T>
GeluFfnLayerINT8<T>::GeluFfnLayerINT8(GeluFfnLayerINT8<T> const& gelu_ffn_layer): FfnLayerINT8<T>(gelu_ffn_layer)
{
}

template<typename T>
void GeluFfnLayerINT8<T>::invokeAddBiasActivation(const int m, const T* bias, ScaleList* scale_list)
{
    if (int8_mode_ == 1) {
        invokeAddBiasGeluCol32<T>(inter_buf_,
                                  inter_int_buf_,
                                  bias,
                                  m,
                                  inter_size_,
                                  stream_,
                                  &(scale_list->d_scale_list_[scale_list->p2_offset_ + 4 * hidden_units_]),
                                  &(scale_list->d_scale_list_[44 + 2]),
                                  &(scale_list->d_scale_list_[52 + 3]));
    }
    else if (int8_mode_ == 2 || int8_mode_ == 3) {
#ifdef SPARSITY_ENABLED
        if (sparse_) {
            invokeAddBiasGeluRow<T>(inter_buf_,
                                    (const int8_t*)inter_int_buf_,
                                    bias,
                                    m,
                                    inter_size_,
                                    stream_,
                                    &(scale_list->d_scale_list_[48 + 1]),
                                    &(scale_list->d_scale_list_[52 + 3]));
        }
        else {
#endif
            invokeAddBiasGeluCol32<T>(inter_buf_,
                                      (const int8_t*)inter_int_buf_,
                                      bias,
                                      m,
                                      inter_size_,
                                      stream_,
                                      &(scale_list->d_scale_list_[48 + 1]),
                                      &(scale_list->d_scale_list_[52 + 3]));
#ifdef SPARSITY_ENABLED
        }
#endif
    }
}

template class GeluFfnLayerINT8<float>;
template class GeluFfnLayerINT8<half>;

template<typename T>
ReluFfnLayerINT8<T>::ReluFfnLayerINT8(size_t           max_batch_size,
                                      size_t           max_seq_len,
                                      size_t           head_num,
                                      size_t           size_per_head,
                                      size_t           inter_size,
                                      int              int8_mode,
                                      cudaStream_t     stream,
                                      cublasMMWrapper* cublas_wrapper,
                                      IAllocator*      allocator,
                                      bool             is_free_buffer_after_forward):
    FfnLayerINT8<T>(max_batch_size,
                    max_seq_len,
                    head_num,
                    size_per_head,
                    inter_size,
                    int8_mode,
                    stream,
                    cublas_wrapper,
                    allocator,
                    is_free_buffer_after_forward)
{
}

template<typename T>
ReluFfnLayerINT8<T>::ReluFfnLayerINT8(ReluFfnLayerINT8<T> const& relu_ffn_layer): FfnLayerINT8<T>(relu_ffn_layer)
{
}

template<typename T>
void ReluFfnLayerINT8<T>::invokeAddBiasActivation(const int m, const T* bias, ScaleList* scale_list)
{
    // TODO
}

template class ReluFfnLayerINT8<float>;
template class ReluFfnLayerINT8<half>;

}  // namespace fastertransformer
