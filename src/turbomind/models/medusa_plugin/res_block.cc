// Copyright (c) OpenMMLab. All rights reserved.
// Yineng Zhang <me@zhyncs.com>
// Zhiwei Bao <zwbao@foxmail.com>

#include "src/turbomind/models/medusa_plugin/res_block.h"
#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/utils/Tensor.h"

namespace turbomind {

template<typename T>
void ResBlock<T>::forward(TensorMap* output_tensors, const TensorMap* input_tensors, const LlamaDenseWeight<T>& weight)
{
    T*           resblock_output = output_tensors->at("resblock_output").getPtr<T>();
    const T*     resblock_input  = input_tensors->at("resblock_input").getPtr<T>();
    const size_t batch_size      = input_tensors->at("resblock_input").shape[0];

    forward(resblock_output, resblock_input, batch_size, weight);
}

template<typename T>
void ResBlock<T>::forward(T*                         resblock_output,
                          const T*                   resblock_input,
                          size_t                     batch_size,
                          const LlamaDenseWeight<T>& weight)
{
    linear_->forward(resblock_output, resblock_input, batch_size, weight);
    const int tp_num    = tensor_para_.world_size_;
    const int tp_offset = tensor_para_.rank_ * in_size_ / tp_num;

    invokeFusedBiasResidualActivation<SiluActivation>(resblock_output,
                                                      (const T*)weight.bias,     // bias
                                                      (const T*)resblock_input,  // residual
                                                      batch_size,                // m
                                                      in_size_ / tp_num,         // n
                                                      stream_,
                                                      tp_num,
                                                      tp_offset);
}

template class ResBlock<float>;
template class ResBlock<half>;
#ifdef ENABLE_BF16
template class ResBlock<__nv_bfloat16>;
#endif

}  // namespace turbomind
