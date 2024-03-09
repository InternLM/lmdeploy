// Copyright (c) OpenMMLab. All rights reserved.
// Yineng Zhang <me@zhyncs.com>
// Zhiwei Bao <zwbao@foxmail.com>

#include "src/turbomind/models/medusa_plugin/medusa_head.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
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
                          NcclParam        tensor_para,
                          bool             is_free_buffer_after_forward):
    in_size_(in_size),
    vocab_size_(vocab_size),
    medusa_num_heads_(medusa_num_heads),
    stream_(stream),
    cublas_wrapper_(cublas_wrapper),
    allocator_(allocator),
    tensor_para_(tensor_para),
    is_free_buffer_after_forward_(is_free_buffer_after_forward)
{
    resblock_ = std::make_unique<ResBlock<T>>(in_size_, stream_, cublas_wrapper_, tensor_para_);
    linear_   = std::make_unique<LlamaLinear<T>>(cublas_wrapper_, stream_);
}

template<typename T>
void MedusaHead<T>::forward(TensorMap*             output_tensors,
                            const TensorMap*       input_tensors,
                            const MedusaWeight<T>& medusa_weight)
{
    const size_t batch_size        = input_tensors->at("medusa_head_input").shape[0];
    const T*     hidden_states     = input_tensors->at("medusa_head_input").getPtr<T>();
    int*         h_topk_output_ids = output_tensors->at("medusa_head_output").getPtr<int>();

    allocate_buffer(batch_size);
    // TODO parallelize this loop
    for (int i = 0; i < medusa_num_heads_; i++) {
        T* medusa_head_logits = medusa_head_logits_buf_ + i * batch_size * vocab_size_;
        forward(medusa_head_logits, hidden_states, batch_size, medusa_weight, i);
    }

    top_k(h_topk_output_ids, medusa_head_logits_buf_, batch_size * medusa_num_heads_);
}

template<typename T>
void MedusaHead<T>::forward(T*                     medusa_head_output,
                            const T*               medusa_head_input,
                            size_t                 batch_size,
                            const MedusaWeight<T>& medusa_weight,
                            int                    head_id)
{
    // TODO support multi medusa_num_layers
    resblock_->forward(resblock_buf_, medusa_head_input, batch_size, medusa_weight.get_resblocks_weights()[head_id][0]);
    linear_->forward(medusa_head_output, resblock_buf_, batch_size, medusa_weight.get_heads_weights()[head_id]);

    if (tensor_para_.world_size_ > 1) {
        NcclGuard nccl_guard(tensor_para_, stream_);
        ftNcclAllReduceSum(medusa_head_output, medusa_head_output, batch_size * vocab_size_, tensor_para_, stream_);
        sync_check_cuda_error();
    }

    free_buffer();
}

template<typename T>
void MedusaHead<T>::allocate_buffer(size_t batch_size)
{
    resblock_buf_ =
        (T*)allocator_->reMalloc(resblock_buf_, sizeof(T) * batch_size * in_size_ / tensor_para_.world_size_, false);
    medusa_head_logits_buf_ = (T*)allocator_->reMalloc(
        medusa_head_logits_buf_, medusa_num_heads_ * sizeof(T) * batch_size * vocab_size_, false);
    is_allocated_buffer_ = true;
}

template<typename T>
void MedusaHead<T>::free_buffer()
{
    if (is_free_buffer_after_forward_ && is_allocated_buffer_) {
        allocator_->free((void**)&resblock_buf_);
        allocator_->free((void**)&workspace_buf_);
        allocator_->free((void**)&medusa_head_logits_buf_);
        is_allocated_buffer_ = false;
    }
}

template<typename T>
void MedusaHead<T>::top_k(int* h_topk_output_ids, const T* d_input_logits, const size_t batch_size, const int k)
{
    size_t workspace_size_now = 0;
    invokeBatchTopKOnly(nullptr,
                        workspace_size_now,
                        d_input_logits,
                        nullptr,
                        k,
                        nullptr,
                        vocab_size_,
                        nullptr,
                        stream_,
                        batch_size,
                        nullptr);
    workspace_buf_ = (void*)allocator_->reMalloc(workspace_buf_, workspace_size_now, false);
    invokeBatchTopKOnly(workspace_buf_,
                        workspace_size_now,
                        d_input_logits,
                        nullptr,
                        k,
                        nullptr,
                        vocab_size_,
                        nullptr,
                        stream_,
                        batch_size,
                        nullptr);
    int  offset          = (int)(ceil(batch_size * vocab_size_ / 4.)) * 4;
    int  output_size     = (int)(ceil(batch_size * k / 4.)) * 4;
    int* topk_output_ids = (int*)(((T*)workspace_buf_) + offset);
    cudaMemcpy(h_topk_output_ids, topk_output_ids, sizeof(int) * output_size, cudaMemcpyDeviceToHost);
}

template class MedusaHead<float>;
template class MedusaHead<half>;
#ifdef ENABLE_BF16
template class MedusaHead<__nv_bfloat16>;
#endif

}  // namespace turbomind
