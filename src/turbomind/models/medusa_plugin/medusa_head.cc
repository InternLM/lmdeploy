// Copyright (c) OpenMMLab. All rights reserved.
// Yineng Zhang <me@zhyncs.com>
// Zhiwei Bao <zwbao@foxmail.com>

#include "src/turbomind/models/medusa_plugin/medusa_head.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cublasMMWrapper.h"

namespace turbomind {

template<typename T>
MedusaHead<T>::MedusaHead(size_t           in_size,
                          size_t           vocab_size,
                          size_t           medusa_num_heads,
                          cudaStream_t     stream,
                          cublasMMWrapper* cublas_wrapper,
                          IAllocator*      allocator,
                          bool             is_free_buffer_after_forward):
    in_size_(in_size),
    vocab_size_(vocab_size),
    medusa_num_heads_(medusa_num_heads),
    stream_(stream),
    cublas_wrapper_(cublas_wrapper),
    allocator_(allocator),
    is_free_buffer_after_forward_(is_free_buffer_after_forward)
{
    resblock_ = std::make_unique<ResBlock<T>>(in_size_, vocab_size, stream_, cublas_wrapper_);
    linear_   = std::make_unique<LlamaLinear<T>>(cublas_wrapper_, stream_);
}

template<typename T>
void MedusaHead<T>::forward(TensorMap*             output_tensors,
                            const TensorMap*       input_tensors,
                            const MedusaWeight<T>& medusa_weight)
{
    const size_t     batch_size             = input_tensors->at("medusa_head_input").shape[0];
    const T*         hidden_states          = input_tensors->at("medusa_head_input").getPtr<T>();
    std::vector<T*>* medusa_head_logits_vec = output_tensors->at("medusa_head_output").getPtr<std::vector<T*>>();
    for (int i = 0; i < medusa_num_heads_; i++) {
        T* medusa_head_logits = (*medusa_head_logits_vec)[i];
        forward(medusa_head_logits, hidden_states, batch_size, medusa_weight, i);
    }
}

template<typename T>
void MedusaHead<T>::forward(T*                     medusa_head_output,
                            const T*               medusa_head_input,
                            size_t                 batch_size,
                            const MedusaWeight<T>& medusa_weight,
                            int                    head_id)
{
    allocate_buffer(batch_size);
    resblock_->forward(resblock_buf_, medusa_head_input, batch_size, medusa_weight.get_resblocks_weights()[head_id][0]);
    linear_->forward(medusa_head_output, resblock_buf_, batch_size, medusa_weight.get_heads_weights()[head_id]);
    free_buffer();
}

template<typename T>
void MedusaHead<T>::allocate_buffer(size_t batch_size)
{
    resblock_buf_        = (T*)allocator_->reMalloc(resblock_buf_, sizeof(T) * batch_size * in_size_, false);
    is_allocated_buffer_ = true;
}

template<typename T>
void MedusaHead<T>::free_buffer()
{
    if (is_free_buffer_after_forward_ && is_allocated_buffer_) {
        allocator_->free((void**)&resblock_buf_);
        is_allocated_buffer_ = false;
    }
}

template struct MedusaHead<float>;
template struct MedusaHead<half>;
#ifdef ENABLE_BF16
template struct MedusaHead<__nv_bfloat16>;
#endif

}  // namespace turbomind
