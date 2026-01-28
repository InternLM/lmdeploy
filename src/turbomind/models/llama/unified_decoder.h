#pragma once

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

namespace turbomind {

class UnifiedDecoder {
public:
    using WeightType = LlamaDecoderLayerWeight;

    UnifiedDecoder(const ModelParam&     model,
                   const EngineParam&    engine,
                   const AttentionParam& attn,
                   const MoeParam&       moe,
                   const Context&        ctx,
                   int                   phases);

    void Run(BatchOp op, int phase, TensorMap& env);

    void Forward(int phase, TensorMap& env, const std::vector<WeightType*>& weights);

private:
    const size_t layer_num_;
    const size_t hidden_units_;

    const int attn_tp_size_;
    const int attn_dp_size_;
    const int attn_dp_rank_;
    const int mlp_tp_size_;

    const int attn_tp_group_;

    const float rmsnorm_eps_;

    comm::DeviceCommImpl* const d_comm_;

    const int tune_layer_num_;

    int& is_warm_up_;

    std::unique_ptr<UnifiedAttentionLayer> attn_layer_;
    std::unique_ptr<LlamaFfnLayer>         ffn_layer_;
    std::unique_ptr<MoeFfnLayer>           moe_ffn_layer_;

    void AllreduceResidualRMSnorm(Tensor&       hidden_states,
                                  Tensor&       residual,
                                  const Tensor& bias,
                                  const Tensor& weight,
                                  int           token_num,
                                  int           t0,
                                  int           t1,
                                  const int*    local_token_nums);
};

}  // namespace turbomind
