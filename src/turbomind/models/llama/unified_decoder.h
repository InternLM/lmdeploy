#pragma once

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

class UnifiedDecoder {
private:
    const size_t layer_num_;
    const size_t hidden_units_;

    const int attn_tp_size_;
    const int attn_dp_size_;
    const int attn_dp_rank_;
    const int mlp_tp_size_;

    const int attn_tp_group_;

    const float        rmsnorm_eps_;
    cudaStream_t const stream_;
    IAllocator* const  allocator_;

    comm::DeviceCommImpl* const d_comm_;

    const int tune_layer_num_;
    bool      is_free_buffer_after_forward_{};

    std::unique_ptr<UnifiedAttentionLayer> attn_layer_;
    std::unique_ptr<LlamaFfnLayer>         ffn_layer_;
    std::unique_ptr<MoeFfnLayer>           moe_ffn_layer_;

    using WeightType = LlamaDecoderLayerWeight;

    std::shared_ptr<UnifiedAttentionLayer::ForwardParam> attn_fwd_param_;

    void AllreduceResidualRMSnorm(core::Tensor&       hidden_states,
                                  core::Tensor&       residual,
                                  const core::Buffer& bias,
                                  const core::Buffer& weight,
                                  int                 token_num,
                                  int                 t0,
                                  int                 t1,
                                  const int*          local_token_nums);

public:
    UnifiedDecoder(const ModelParam&     model,
                   const EngineParam&    engine,
                   const AttentionParam& attn,
                   const MoeParam&       moe,
                   const LoraParam&      lora,
                   const Context&        ctx);

    ~UnifiedDecoder();

    void Forward(core::TensorMap& args, const std::vector<WeightType*>& weights);
};

}  // namespace turbomind
