// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/comm/cuda_ipc/cuda_ipc_comm.h"
#include "src/turbomind/kernels/gemm/context.h"
#include "src/turbomind/kernels/gemm/moe_a2a_utils.h"
#include "src/turbomind/kernels/gemm/moe_utils_v2.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class MoeFfnLayer {
public:
    MoeFfnLayer(const ModelParam& model, const MoeParam& param, const EngineParam& engine, const Context& ctx);

    struct ForwardParam {
        Tensor              input;
        Tensor              output;
        const MoeFfnWeight* weights;
        float               scale;
        int                 layer_id;
    };

    void Forward(ForwardParam& p);

    void Combine(ForwardParam& p);

private:
    Tensor_<float> Gate(const Tensor& input, const LlamaDenseWeight& gate);

    void dump_logits(int token_num, int layer_id, int expert_num);

    void ForwardNative(ForwardParam& p);

    void ForwardFused(ForwardParam& p);

    void RouteTP(Tensor_<float>& logits, int tokens, int padded, int expert_num, bool softmax, cudaStream_t st);

    void RouteEP(Tensor_<float>& logits, int tokens, int padded, int expert_num, bool softmax, cudaStream_t st);

    void CombineTP(ForwardParam& p);

    void CombineEP(ForwardParam& p);

    const int inter_size_;
    const int hidden_dim_;
    const int tp_size_;
    const int ep_size_;
    const int ep_rank_;

    const MoeParam param_;

    int& is_warm_up_;

    LlamaLinear& linear_;

    std::unique_ptr<LlamaFfnLayer> expert_ffn_;

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

    /// context
    Tensor         input_;
    Tensor         temp_;
    Tensor_<float> shared_scales_;
    ///////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////
    /// EP
    comm::CudaIpcCommImpl* d_comm_{};

    Buffer_<float> topk_scales_;        // (n, ep_size, num_topk)
    Buffer_<int>   topk_experts_;       // (n, num_experts)
    Buffer_<int>   token_idx_in_rank_;  // (ep_size, n + 2), idx..., token_count, expert_count

    Buffer_<int>     symm_meta_;    // (2, ep_size, ep_size)
    Buffer_<uint8_t> symm_hidden_;  // max_tokens, hidden        (n, H)
    Buffer_<float>   symm_scales_;  // num_topk, max_tokens      (e, n)
    Buffer_<int8_t>  symm_masks_;   // local_experts, max_tokens (E, n)

    ZeroCopyItem<int> num_input;  // num of input token
    ZeroCopyItem<int> num_flat;   // num of flatten token

    ///////////////////////////////////////////////////////
};

}  // namespace turbomind
