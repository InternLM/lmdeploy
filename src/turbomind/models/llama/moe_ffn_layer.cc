// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/monotonic.h"
#include "src/turbomind/utils/nvtx_utils.h"
#include "src/turbomind/utils/string_utils.h"
#include <cuda_runtime.h>
#include <iomanip>

namespace turbomind {

template<class T>
void MoeFfnLayer<T>::AllocateBuffer(size_t tokens, size_t padded, size_t expert_num, size_t inter_buf_factor)
{
    char* base = 0;

    auto allocate = [&](void* base) {
        Monotonic alloc{base};
        alloc(&inout_buf_, tokens * param_.experts_per_token * hidden_dim_);
        alloc(&inter_buf_, tokens * param_.experts_per_token * inter_size_ * inter_buf_factor);
        alloc(&logits_, tokens * expert_num);
        alloc(&masks_, expert_num * padded);
        alloc(&f2n_, param_.experts_per_token * tokens);
        alloc(&en2f_, param_.experts_per_token * tokens);
        alloc(&scales_, param_.experts_per_token * tokens);
        alloc(&shared_scales_, tokens);
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
    cublas_->Gemm(CUBLAS_OP_N,
                  CUBLAS_OP_N,
                  weight.output_dims,
                  tokens,
                  weight.input_dims,
                  &alpha,
                  weight.kernel,
                  getCudaDataType<T>(),
                  weight.output_dims,
                  input,
                  getCudaDataType<T>(),
                  hidden_dim_,
                  &beta,
                  logits,
                  CUDA_R_32F,
                  weight.output_dims,
                  CUDA_R_32F,
                  CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

template<class T>
void MoeFfnLayer<T>::forward(T* output, const T* input, int tokens, int layer_id, const MoeFfnWeight<T>& moe)
{
    const size_t padded     = (tokens + kMoeGateVecSize - 1) / kMoeGateVecSize * kMoeGateVecSize;
    const int    expert_num = moe.experts.size();

    FT_CHECK(expert_num);

    const size_t inter_buf_factor = [&] {
        if (param_.method == MoeParam::kNaive) {
            return 0;  // managed by ffn
        }
        else if (moe.block.is_fused_silu) {
            return 1;
        }
        else {
            return 2;
        }
    }();

    AllocateBuffer(tokens, padded, expert_num, inter_buf_factor);

    gate(logits_, input, tokens, moe.gate);
    sync_check_cuda_error();

    // if (tensor_para_.rank_ == 0) {
    //     Compare(logits_, tokens * expert_num, Concat("logit", layer_id), compare_mode, stream_);
    // }

    check_cuda_error(cudaMemsetAsync(accum_, 0, sizeof(int) * expert_num * kMoeGateMaxTiles, stream_));
    check_cuda_error(cudaMemsetAsync(masks_, -1, sizeof(int8_t) * expert_num * padded, stream_));

    // dump_logits(tokens, layer_id);

    bool softmax = true;
    if (param_.topk_method == "group_limited_greedy") {
        invokeMoeSoftmaxMaskTopKGroups(
            logits_, tokens, expert_num, expert_num / param_.n_group, param_.topk_group, stream_);
        sync_check_cuda_error();
        softmax = false;
    }

    /// TODO: fix illegal memory access even if NaN are present in logits
    invokeMoeGate_V2(f2n_,
                     en2f_,
                     offsets_,
                     scales_,
                     masks_,
                     accum_,
                     logits_,
                     tokens,
                     padded,
                     expert_num,
                     param_.experts_per_token,
                     softmax,
                     param_.norm_topk_prob,
                     param_.routed_scale,
                     stream_);
    sync_check_cuda_error();

    if (isTuning()) {
        std::mt19937     g;
        const auto       expert_ids = SampleUniform(tokens, expert_num, param_.experts_per_token, g);
        std::vector<int> cnt(expert_num);
        for (const auto& x : expert_ids) {
            ++cnt[x];
        }
        h_offsets_[0] = 0;
        for (int i = 0; i < expert_num; ++i) {
            h_offsets_[i + 1] = h_offsets_[i] + cnt[i];
        }
        check_cuda_error(
            cudaMemcpyAsync(offsets_, h_offsets_, sizeof(int) * (expert_num + 1), cudaMemcpyDefault, stream_));
    }

    if (param_.method == MoeParam::kNaive) {

        dispatchMoeGather(inout_buf_, input, f2n_, tokens, param_.experts_per_token, hidden_dim_, stream_);
        sync_check_cuda_error();

        check_cuda_error(
            cudaMemcpyAsync(h_offsets_, offsets_, sizeof(int) * (expert_num + 1), cudaMemcpyDefault, stream_));

        check_cuda_error(cudaStreamSynchronize(stream_));

        if (h_offsets_[expert_num] != tokens * param_.experts_per_token) {
            FT_CHECK_WITH_INFO(0, fmtstr("%d vs %d", h_offsets_[expert_num], tokens * param_.experts_per_token));
        }

        for (int i = 0; i < expert_num; ++i) {

            FT_CHECK(moe.experts[i].is_fused_silu == false);

            if (size_t count = h_offsets_[i + 1] - h_offsets_[i]) {
                auto io = inout_buf_ + h_offsets_[i] * hidden_dim_;

                TensorMap ffn_inputs{{"ffn_input", {MEMORY_GPU, dtype_, {count, hidden_dim_}, io}},
                                     {"layer_id", {MEMORY_CPU, TYPE_INT32, {1}, &layer_id}}};
                TensorMap ffn_outputs{{"ffn_output", {MEMORY_GPU, dtype_, {count, hidden_dim_}, io}}};

                expert_ffn_->forward(&ffn_outputs, &ffn_inputs, &moe.experts[i]);
            }
        }
    }
    else {
        context_->update(expert_num, param_.experts_per_token, offsets_);

        auto& block = moe.block;

        linear_->forward_moe(inter_buf_,
                             {input, (int)hidden_dim_},
                             f2n_,
                             offsets_,
                             tokens * param_.experts_per_token,
                             block.fused_gating_intermediate,
                             block.is_fused_silu ? LlamaLinear<T>::kFusedSiluFfn : LlamaLinear<T>::kGemm,
                             context_.get());
        sync_check_cuda_error();
        auto mode = kCmpRead;

        // if (tensor_para_.rank_ == 0) {
        //     Compare(inter_buf_,  //
        //             tokens * param_.experts_per_token * inter_size_ * 2,
        //             "inter_buf",
        //             mode,
        //             stream_);
        // }

        if (!block.is_fused_silu) {
            invokeGenericActivation_v2<SiluActivation>(inter_buf_,
                                                       inter_buf_ + inter_size_,
                                                       inter_size_ * 2,
                                                       tokens * param_.experts_per_token,
                                                       inter_size_,
                                                       stream_);
            sync_check_cuda_error();
        }

        linear_->forward_moe(inout_buf_,
                             {inter_buf_, block.is_fused_silu ? (int)inter_size_ : (int)inter_size_ * 2},
                             nullptr,
                             offsets_,
                             tokens * param_.experts_per_token,
                             block.output,
                             LlamaLinear<T>::kGemm,
                             context_.get());
        sync_check_cuda_error();
        auto mode1 = kCmpRead;

        // if (tensor_para_.rank_ == 0) {
        //     Compare(inter_buf_2_,  //
        //             tokens * param_.experts_per_token * inter_size_,
        //             "inter_buf_2_",
        //             mode1,
        //             stream_);
        //     Compare(inout_buf_,  //
        //             tokens * param_.experts_per_token * hidden_dim_,
        //             "inout_buf",
        //             mode1,
        //             stream_);
        // }
    }

    if (moe.shared_gate.kernel) {
        gate(shared_scales_, input, tokens, moe.shared_gate);
    }
}

template<class T>
void MoeFfnLayer<T>::reduce(T* output, int tokens, float output_scale, int layer_id, const MoeFfnWeight<T>& moe)
{
    invokeMoeReduce(output,
                    inout_buf_,
                    scales_,
                    en2f_,
                    moe.shared_gate.kernel ? shared_scales_ : nullptr,
                    tokens,
                    param_.experts_per_token,
                    hidden_dim_,
                    output_scale,
                    stream_);
    sync_check_cuda_error();

    if (tensor_para_.world_size_ > 1) {
        // std::cout << "moe all reduce " << layer_id << "\n";
        ftNcclAllReduceSum(output, output, tokens * hidden_dim_, tensor_para_, stream_);
        sync_check_cuda_error();
    }
}

template<class T>
void MoeFfnLayer<T>::dump_logits(int token_num, int layer_id, int expert_num)
{
    std::vector<float> logits(token_num * expert_num);
    check_cuda_error(
        cudaMemcpyAsync(logits.data(), logits_, sizeof(float) * logits.size(), cudaMemcpyDefault, stream_));
    check_cuda_error(cudaStreamSynchronize(stream_));

    auto ptr = logits.data();
    std::cout << "layer_id: " << layer_id << std::endl;
    for (int i = 0; i < token_num; ++i) {
        for (int e = 0; e < expert_num; ++e) {
            std::cout << *ptr++ << " ";
        }
        std::cout << std::endl;
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
