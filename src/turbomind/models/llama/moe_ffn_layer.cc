// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/nvtx_utils.h"
#include "src/turbomind/utils/string_utils.h"
#include <cuda_runtime.h>
#include <iomanip>

namespace turbomind {

class Monotonic {
public:
    Monotonic(void* base, size_t alignment = 256): ptr_{base}, alignment_{alignment}
    {
        ptr_ = align(ptr_);
    }

    template<class T>
    void operator()(T** ptr, size_t numel) noexcept
    {
        *ptr = (T*)std::exchange(ptr_, align((T*)ptr_ + numel));
    }

    void* ptr() const noexcept
    {
        return ptr_;
    }

private:
    template<class T>
    void* align(T* p)
    {
        static_assert(sizeof(T*) == sizeof(uintptr_t));
        auto& x = (uintptr_t&)p;
        if (auto remainder = x % alignment_) {
            x += alignment_ - remainder;
        }
        return p;
    }

    void*  ptr_;
    size_t alignment_;
};

template<class T>
void MoeFfnLayer<T>::AllocateBuffer(size_t tokens, size_t padded)
{
    char* base = 0;

    auto allocate = [&](void* base) {
        Monotonic alloc{base};
        alloc(&inout_buf_, tokens * param_.experts_per_token * hidden_dim_);
        // alloc(&inter_buf_, tokens * param_.expert_num * inter_size_ * 2);
        alloc(&logits_, tokens * param_.expert_num);
        alloc(&masks_, param_.expert_num * padded);
        alloc(&f2n_, param_.experts_per_token * tokens);
        alloc(&en2f_, param_.experts_per_token * tokens);
        alloc(&scales_, param_.experts_per_token * tokens);
        return (char*)alloc.ptr() - (char*)base;
    };

    const auto workspace_size = allocate(0);

    workspace_ = (char*)allocator_->reMalloc(workspace_, workspace_size);

    allocate(workspace_);
}

template<class T>
void MoeFfnLayer<T>::FreeBuffer()
{
    allocator_->free((void**)&workspace_);

    allocator_->free((void**)&accum_);
    allocator_->free((void**)&offsets_);

    allocator_->free((void**)&h_offsets_, true);
}

template<class T>
void MoeFfnLayer<T>::gate(float* logits, const T* input, int tokens, const LlamaDenseWeight<T>& weight)
{
    const float alpha = 1.f;
    const float beta  = 0.f;
    cublasGemmEx(cublas_,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 weight.output_dims,
                 tokens,
                 weight.input_dims,
                 &alpha,
                 weight.kernel,
                 CUDA_R_16F,
                 weight.output_dims,
                 input,
                 CUDA_R_16F,
                 hidden_dim_,
                 &beta,
                 logits_,
                 CUDA_R_32F,
                 weight.output_dims,
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

template<class T>
void MoeFfnLayer<T>::forward(T* inout, int tokens, int layer_id, const MoeFfnWeight<T>& moe)
{
    const size_t padded = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;

    AllocateBuffer(tokens, padded);

    gate(logits_, inout, tokens, moe.gate);

    check_cuda_error(cudaMemsetAsync(accum_, 0, sizeof(int) * param_.expert_num * kMoeGateMaxTiles, stream_));

    invokeMoeGate_V2(f2n_,
                     en2f_,
                     offsets_,
                     scales_,
                     masks_,
                     accum_,
                     logits_,
                     tokens,
                     padded,
                     param_.expert_num,
                     param_.experts_per_token,
                     stream_);
    sync_check_cuda_error();

    check_cuda_error(
        cudaMemcpyAsync(h_offsets_, offsets_, sizeof(int) * (param_.expert_num + 1), cudaMemcpyDefault, stream_));

    dispatchMoeGather(inout_buf_, inout, f2n_, tokens, param_.experts_per_token, hidden_dim_, stream_);
    sync_check_cuda_error();

    check_cuda_error(cudaStreamSynchronize(stream_));

    if (h_offsets_[param_.expert_num] != tokens * param_.experts_per_token) {
        FT_CHECK_WITH_INFO(0, fmtstr("%d vs %d", h_offsets_[param_.expert_num], tokens * param_.experts_per_token));
    }

    for (int i = 0; i < param_.expert_num; ++i) {
        const size_t count = h_offsets_[i + 1] - h_offsets_[i];
        if (count == 0) {
            continue;
        }
        auto io = inout_buf_ + h_offsets_[i] * hidden_dim_;

        TensorMap ffn_inputs{{"ffn_input", {MEMORY_GPU, dtype_, {count, hidden_dim_}, io}},
                             {"layer_id", {MEMORY_CPU, TYPE_INT32, {1}, &layer_id}}};
        TensorMap ffn_outputs{{"ffn_output", {MEMORY_GPU, dtype_, {count, hidden_dim_}, io}}};

        expert_ffn_->forward(&ffn_outputs, &ffn_inputs, &moe.experts[i]);
    }

    invokeMoeReduce(inout, inout_buf_, scales_, en2f_, tokens, param_.experts_per_token, hidden_dim_, stream_);
    sync_check_cuda_error();

    if (tensor_para_.world_size_ > 1) {
        ftNcclAllReduceSum(inout, inout, tokens * hidden_dim_, tensor_para_, stream_);
        sync_check_cuda_error();
    }
}

#ifdef ENABLE_FP32
template class MoeFfnLayer<float>;
#endif
template class MoeFfnLayer<half>;
#ifdef ENABLE_BF16
template class MoeFfnLayer<__nv_bfloat16>;
#endif

}  // namespace turbomind
