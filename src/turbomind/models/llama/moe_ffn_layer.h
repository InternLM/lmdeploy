// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include <algorithm>

namespace turbomind {

class MoeFfnLayer {
public:
    MoeFfnLayer(ModelParam model, const MoeParam& param, size_t tp_size, const Context& ctx):
        inter_size_(param.inter_size / tp_size),
        hidden_dim_(model.hidden_units),
        param_(param),
        stream_(ctx.stream),
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
            expert_ffn_ = std::make_unique<LlamaFfnLayer>(model, ctx);
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

    struct ForwardParam {
        core::Tensor        output;
        core::Tensor        input;
        core::Tensor        temp;
        float               scale;
        int                 layer_id;
        const MoeFfnWeight* weight;
    };

    void Forward(ForwardParam& p);

    void Combine(ForwardParam& p);

private:
    core::Tensor_<float> Gate(const core::Tensor& input, const LlamaDenseWeight& gate);

    void dump_logits(int token_num, int layer_id, int expert_num);

    const int          inter_size_;
    const int          hidden_dim_;
    const MoeParam     param_;
    cudaStream_t const stream_;
    LlamaLinear* const linear_;
    IAllocator* const  allocator_;

    std::unique_ptr<LlamaFfnLayer>        expert_ffn_;
    std::unique_ptr<gemm::MoeGemmContext> context_;

    int* h_offsets_{};

    char* workspace_{};

    int* masks_{};

    int*   f2n_{};
    int*   en2f_{};
    float* scales_{};

    float* shared_scales_{};

    int* accum_{};
    int* offsets_{};
};

}  // namespace turbomind
