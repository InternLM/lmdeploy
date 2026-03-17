#pragma once

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/GatedDeltaNetWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class GatedDeltaNetLayer {
public:
    struct ForwardParam {
        int                        phase;
        Tensor                     input;
        Tensor                     output;
        const GatedDeltaNetWeight* weights;
        int                        layer_id;
    };

    GatedDeltaNetLayer(const ModelParam&     model,
                       const AttentionParam& attn,
                       const EngineParam&    engine,
                       int                   tp_size,
                       const Context&        ctx,
                       int                   phases);

    ~GatedDeltaNetLayer();

    void Run(BatchOp op, int phase, TensorMap& env);

    void Forward(ForwardParam p);

private:
    void Setup(int phase, TensorMap& env);

    // Model dimensions
    int              hidden_units_;
    int              num_k_heads_;
    int              num_v_heads_;
    int              key_head_dim_;
    int              value_head_dim_;
    int              d_conv_;
    int              key_dim_;            // num_k_heads * key_head_dim
    int              value_dim_;          // num_v_heads * value_head_dim
    int              conv_dim_;           // key_dim * 2 + value_dim
    int              num_linear_layers_;  // count of linear attention layers for state sizing
    std::vector<int> layer_types_;        // model layer types for index mapping

    float    norm_eps_;
    DataType dtype_;
    DataType state_dtype_;  // recurrent state dtype (may differ from dtype_ for float32 state)

    LlamaLinear& linear_;

    // Per-phase batch data (mirrors UnifiedAttentionLayer pattern)
    struct Data {
        std::vector<RequestCache*> rc;          // borrowed batch RequestCache pointers
        std::vector<int>           input_lens;  // snapshot of input_len per request (captured at Setup time)
        int                        batch_size = 0;
        Buffer_<int>               q_offsets;  // cumulative token offsets, device buffer
        std::vector<Tensor>        conv_states;
        std::vector<Tensor>        recurrent_states;
        Buffer_<void*>             conv_state_ptrs;
        Buffer_<void*>             recurrent_state_ptrs;
    };
    std::vector<Data> data_;

    // staging buffers
    Buffer_<void*> conv_state_ptrs_buf_;
    Buffer_<void*> recurrent_state_ptrs_buf_;

    // Queried once at construction; passed to all three kernel launchers.
    int          sm_count_{1};
    Buffer_<int> work_counter_;  // 1-element device int for v3 atomic claiming

    // Dual-stream dispatch: prefill on high-priority aux stream, decode on main
    cudaStream_t aux_stream_{};
    cudaEvent_t  ev_before_{};  // main→aux: prior work done
    cudaEvent_t  ev_after_{};   // aux→main: prefill done
};

}  // namespace turbomind
