#pragma once

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/models/llama/GatedDeltaNetLayer.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

namespace turbomind {

class ModelWeight;
class DecoderLayerWeight;
class CacheRegistry;

class UnifiedDecoder {
public:
    using WeightType = DecoderLayerWeight;

    UnifiedDecoder(CacheRegistry&     registry,
                   const EngineParam& engine,
                   const Context&     ctx,
                   int                phases,
                   const ModelWeight& model_weight);

    void Run(BatchOp op, int phase, TensorMap& env);

    // `selected_hidden_buffer` is the caller's request to write selected
    // hidden states into its own buffer (and receive `pre_final_residual`);
    // empty leaves selection to the decoder. The typed argument is the
    // request — env keys carry no activation.
    void Forward(int                     phase,
                 TensorMap&              env,
                 const std::vector<WeightType*>& weights,
                 const Tensor&           selected_hidden_buffer = {});

    void CommitAcceptedState(int phase, const Buffer_<int>& accept_len)
    {
        if (linear_attn_layer_) {
            linear_attn_layer_->CommitAcceptedState(phase, accept_len);
        }
    }

    size_t SpeculativeStateJournalBytes(int request_count, int verification_positions) const
    {
        return linear_attn_layer_ ?
                   linear_attn_layer_->SpeculativeStateJournalBytes(request_count, verification_positions) :
                   0;
    }

    void SetAttentionForwardMetadata(int phase, const AttentionForwardMetadata& metadata)
    {
        attn_layer_->SetForwardMetadata(phase, metadata);
    }

private:
    const size_t layer_num_;
    const size_t hidden_units_;
    const bool   output_norm_zero_centered_;

    const int attn_tp_size_;
    const int attn_dp_size_;
    const int attn_dp_rank_;
    const int mlp_tp_size_;

    const int attn_tp_group_;
    const int mlp_group_;
    const int node_group_;

    // Per-layer post-FFN reduce group, precomputed in the constructor.
    std::vector<int> ffn_group_;

    // All-valid per-token mask, materialized when no producer supplies `token_mask`.
    Buffer_<bool> all_valid_mask_;

    comm::DeviceCommImpl* const d_comm_;

    const int tune_layer_num_;

    int& is_warm_up_;

    std::unique_ptr<UnifiedAttentionLayer> attn_layer_;
    std::unique_ptr<GatedDeltaNetLayer>    linear_attn_layer_;
    std::unique_ptr<LlamaFfnLayer>         ffn_layer_;
    std::unique_ptr<MoeFfnLayer>           moe_ffn_layer_;

    void AllreduceResidualRMSnorm(Tensor&       hidden_states,
                                  Tensor&       residual,
                                  const Tensor& bias,
                                  const Tensor& weight,
                                  float         eps,
                                  bool          zero_centered,
                                  int           token_num,
                                  int           t0,
                                  int           t1,
                                  const int*    local_token_nums,
                                  int           local_token_nums_count);
};

}  // namespace turbomind
