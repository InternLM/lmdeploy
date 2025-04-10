// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class MoeFfnLayer {
public:
    MoeFfnLayer(const ModelParam& model, const MoeParam& param, const EngineParam& engine, const Context& ctx);

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

    const int      inter_size_;
    const int      hidden_dim_;
    const MoeParam param_;

    cudaStream_t const stream_;
    LlamaLinear* const linear_;

    std::unique_ptr<LlamaFfnLayer>        expert_ffn_;
    std::unique_ptr<gemm::MoeGemmContext> context_;

    core::Buffer_<int> h_offsets_;

    core::Buffer_<int>   masks_;
    core::Buffer_<int>   f2n_;
    core::Buffer_<int>   en2f_;
    core::Buffer_<float> scales_;
    core::Buffer_<float> shared_scales_;
    core::Buffer_<int>   accum_;
    core::Buffer_<int>   offsets_;
};

}  // namespace turbomind
