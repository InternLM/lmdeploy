// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/nccl_utils.h"
#include <algorithm>

namespace turbomind {

template<class T>
class MoeFfnLayer {
public:
    MoeFfnLayer(ModelParam model, const MoeParam& param, const NcclParam& tp, const Context<T>& ctx):
        inter_size_(param.inter_size / tp.world_size_),
        hidden_dim_(model.hidden_units),
        param_(param),
        dtype_(getTensorType<T>()),
        tensor_para_(tp),
        stream_(ctx.stream),
        cublas_(ctx.cublas_wrapper.get()),
        linear_(ctx.linear.get()),
        allocator_(ctx.allocator.get())
    {
        FT_CHECK(!param.expert_num.empty());
        const int max_expert_num = *std::max_element(param.expert_num.begin(), param.expert_num.end());

        if (param_.method == MoeParam::kFused) {
            context_ = std::make_unique<gemm::MoeGemmContext>(
                max_expert_num, param.experts_per_token, ctx.cuda_device_prop, stream_);
        }
        else {
            expert_ffn_ = std::make_unique<LlamaFfnLayer<T>>(model, tp, ctx);
        }

        h_offsets_ = (int*)allocator_->malloc(sizeof(int) * (max_expert_num + 1), false, true);

        offsets_ = (int*)allocator_->malloc(sizeof(int) * (max_expert_num + 1));
        accum_   = (int*)allocator_->malloc(sizeof(int) * max_expert_num * kMoeGateMaxTiles);
    }

    void AllocateBuffer(size_t tokens, size_t padded, size_t expert_num, size_t inter_buf_factor);

    void FreeBuffer();

    ~MoeFfnLayer()
    {
        FreeBuffer();
    }

    void forward(T* output, const T* input, int tokens, int layer_id, const MoeFfnWeight<T>& moe);

    void reduce(T* output, int tokens, float output_scale, int layer_id, const MoeFfnWeight<T>& moe);

    void gate(float* logits, const T* input, int tokens, const LlamaDenseWeight<T>& weight);

    void dump_logits(int token_num, int layer_id, int expert_num);

private:
    const size_t           inter_size_;
    const size_t           hidden_dim_;
    const MoeParam         param_;
    const DataType         dtype_;
    const NcclParam        tensor_para_;
    cudaStream_t const     stream_;
    cublasMMWrapper* const cublas_;
    LlamaLinear<T>* const  linear_;
    IAllocator* const      allocator_;

    std::unique_ptr<LlamaFfnLayer<T>>     expert_ffn_;
    std::unique_ptr<gemm::MoeGemmContext> context_;

    int* h_offsets_{};

    char* workspace_{};

    T* inout_buf_{};  // [n * e, hidden_dim]
    T* inter_buf_{};  // [n * e, inter_size]

    float* logits_{};
    int*   masks_{};

    int*   f2n_{};
    int*   en2f_{};
    float* scales_{};

    float* shared_scales_{};

    int* accum_{};
    int* offsets_{};
};

}  // namespace turbomind
