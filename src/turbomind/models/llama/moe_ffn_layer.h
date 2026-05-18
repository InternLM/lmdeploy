// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/moe_weight.h"

namespace turbomind {

class MoeFfnLayer {
public:
    MoeFfnLayer(const EngineParam& engine, const Context& ctx);

    struct ForwardParam {
        Tensor           input;
        Tensor           output;
        const MoeWeight* weights;
        float            scale;
        int              max_tokens_per_rank;
        Buffer           ht_buffer;
        int              layer_id;
    };

    void Forward(ForwardParam& p);

    void Combine(ForwardParam& p);

private:
    void Init(ForwardParam& p);

    Tensor_<float> Gate(const Tensor& input, const LinearWeight& gate);

    void SetWarmup(ForwardParam& p);

    void ForwardFused(ForwardParam& p);

    void RouteTP(ForwardParam& p, Tensor_<float>& logits);

    void RouteEP(ForwardParam& p, Tensor_<float>& logits);

    void CombineTP(ForwardParam& p);

    void CombineEP(ForwardParam& p);

    void dump_logits(int token_num, int layer_id, int expert_num);

    const int tp_size_;
    const int ep_size_;
    const int max_token_num_;
    const int ll_max_tokens_per_rank_;

    int& is_warm_up_;

    LlamaLinear& linear_;

    std::unique_ptr<LlamaFfnLayer> expert_ffn_;

    bool initialized_ = false;

    ///////////////////////////////////////////////////////
    /// runtime states
    Buffer_<int> h_offsets_;

    Buffer_<int>   masks_;
    Buffer_<int>   f2n_;
    Buffer_<int>   f2E_;
    Buffer_<int>   en2f_;
    Buffer_<float> scales_;
    Buffer_<int>   accum_;
    Buffer_<int>   offsets_;

    comm::DeviceCommImpl* const             d_comm_;
    Buffer_<float>                          topk_weights_;
    Buffer_<int64_t>                        topk_idx_;
    std::unique_ptr<comm::EpDispatchOutput> dispatch_output_;
    comm::EpMode                            ep_mode_{comm::EpMode::kNull};

    Tensor         input_;
    Tensor         temp_;
    Tensor_<float> shared_scales_;
    ///////////////////////////////////////////////////////
};

}  // namespace turbomind
