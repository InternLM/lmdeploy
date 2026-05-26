#pragma once

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

struct Sequence;

class GatedDeltaNetLayer {
public:
    struct ForwardParam {
        int                   phase;
        Tensor                input;
        Tensor                output;
        const DeltaNetWeight* weights;
        int                   layer_id;
    };

    GatedDeltaNetLayer(DataType                state_dtype,
                       const DeltaNetWeight&   prototype,
                       const std::vector<int>& layer_types,
                       const EngineParam&      engine,
                       const Context&          ctx,
                       int                     phases);

    ~GatedDeltaNetLayer();

    void Run(BatchOp op, int phase, TensorMap& env);

    void Forward(ForwardParam p);

private:
    struct Data;

    void Setup(int phase, TensorMap& env);
    bool EnsurePrefixCaptureCapacity(Data& d, int capture_count);
    bool CanAllocatePrefixCapture(int capture_count);

    int              tp_size_;
    int              num_k_heads_;
    int              num_v_heads_;
    int              key_head_dim_;
    int              value_head_dim_;
    int              d_conv_;
    int              conv_dim_;
    int              num_linear_layers_;
    std::vector<int> layer_types_;
    bool             enable_linear_prefix_caching_{false};
    int              linear_prefix_cache_interval_tokens_{0};
    DataType         dtype_;
    DataType         state_dtype_;
    int&             is_warm_up_;
    size_t           prefix_capture_state_bytes_{0};

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

    int          sm_count_{1};
    Buffer_<int> work_counter_;

    cudaStream_t aux_stream_{};
    cudaEvent_t  ev_before_{};
    cudaEvent_t  ev_after_{};
    bool         warned_prefix_capture_oom_{false};
    bool         warned_prefix_capture_budget_{false};
};

}  // namespace turbomind
