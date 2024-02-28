// Copyright (c) OpenMMLab. All rights reserved.
// Yineng Zhang <me@zhyncs.com>
// Zhiwei Bao <zwbao@foxmail.com>

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/medusa_plugin/medusa_weight.h"
#include "src/turbomind/models/medusa_plugin/res_block.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace turbomind {

template<typename T>
class MedusaHead {
public:
    MedusaHead(size_t           in_size,
               size_t           vocab_size,
               size_t           medusa_num_heads,
               cudaStream_t     stream,
               cublasMMWrapper* cublas_wrapper,
               IAllocator*      allocator,
               bool             is_free_buffer_after_forward = false);
    ~MedusaHead()                 = default;
    MedusaHead(const MedusaHead&) = delete;
    MedusaHead& operator=(const MedusaHead&) = delete;

    void forward(TensorMap* output_tensors, const TensorMap* input_tensors, const MedusaWeight<T>& medusa_weight);
    void forward(T*                     medusa_head_output,
                 const T*               medusa_head_input,
                 size_t                 batch_size,
                 const MedusaWeight<T>& medusa_weight,
                 int                    head_id);

private:
    void allocate_buffer(size_t batch_size);
    void free_buffer();

private:
    size_t in_size_;
    size_t vocab_size_;
    size_t medusa_num_heads_;

    std::unique_ptr<ResBlock<T>>    resblock_;
    std::unique_ptr<LlamaLinear<T>> linear_;

    T* resblock_buf_;

    cudaStream_t     stream_;
    cublasMMWrapper* cublas_wrapper_;
    IAllocator*      allocator_;

    bool is_allocated_buffer_          = false;
    bool is_free_buffer_after_forward_ = false;
};
}  // namespace turbomind