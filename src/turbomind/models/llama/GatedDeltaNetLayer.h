#pragma once

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/GatedDeltaNetWeight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

struct Sequence;

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
    struct Data;

    void Setup(int phase, TensorMap& env);
    bool EnsurePrefixCaptureCapacity(Data& d, int capture_count);
    bool CanAllocatePrefixCapture(int capture_count);

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
    bool             enable_linear_prefix_caching_{false};
    int              linear_prefix_cache_interval_tokens_{0};

    float    norm_eps_;
    DataType dtype_;
    DataType state_dtype_;  // recurrent state dtype (may differ from dtype_ for float32 state)
    int&     is_warm_up_;
    size_t   prefix_capture_state_bytes_{0};

    LlamaLinear& linear_;

    // Per-phase batch data (mirrors UnifiedAttentionLayer pattern)
    struct Data {
        std::vector<const Sequence*> seqs;        // borrowed live sequence pointers
        std::vector<int>             input_lens;  // snapshot of input_len per request (captured at Setup time)
        std::vector<int>             history_lens;
        std::vector<int>             capture_counts;
        std::vector<int>             capture_offsets;
        int                          batch_size          = 0;
        int                          total_capture_count = 0;
        Buffer_<int>                 q_offsets;  // cumulative input-token offsets, device buffer
        Buffer_<int>                 k_offsets;  // cumulative key (history+input) offsets, device buffer
        std::vector<Tensor>          conv_states;
        std::vector<Tensor>          recurrent_states;
        Buffer_<void*>               conv_state_ptrs;
        Buffer_<void*>               recurrent_state_ptrs;
        Buffer_<void*>               conv_capture_ptrs;
        Buffer_<void*>               recurrent_capture_ptrs;
        Tensor                       conv_prefix_checkpoints;
        Tensor                       recurrent_prefix_checkpoints;
    };
    std::vector<Data> data_;

    // staging buffers
    Buffer_<void*> conv_state_ptrs_buf_;
    Buffer_<void*> recurrent_state_ptrs_buf_;
    Buffer_<void*> conv_capture_ptrs_buf_;
    Buffer_<void*> recurrent_capture_ptrs_buf_;

    // Queried once at construction; passed to all three kernel launchers.
    int          sm_count_{1};
    Buffer_<int> work_counter_;  // 1-element device int for v3 atomic claiming

    // Dual-stream dispatch: prefill on high-priority aux stream, decode on main
    cudaStream_t aux_stream_{};
    cudaEvent_t  ev_before_{};  // main→aux: prior work done
    cudaEvent_t  ev_after_{};   // aux→main: prefill done

    bool warned_prefix_capture_oom_{false};
    bool warned_prefix_capture_budget_{false};
};

}  // namespace turbomind
