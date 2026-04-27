#pragma once

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

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
                       const std::vector<int>& layer_types,
                       const EngineParam&      engine,
                       const Context&          ctx,
                       int                     phases);

    ~GatedDeltaNetLayer();

    void Run(BatchOp op, int phase, TensorMap& env);

    void Forward(ForwardParam p);

private:
    void Setup(int phase, TensorMap& env);

    // Config passed at construction
    int              tp_size_;
    int              num_linear_layers_;
    std::vector<int> layer_types_;
    DataType         state_dtype_;

    LlamaLinear& linear_;

    // Per-phase batch data (mirrors UnifiedAttentionLayer pattern)
    struct Data {
        std::vector<RequestCache*> rc;
        std::vector<int>           input_lens;
        int                        batch_size = 0;
        Buffer_<int>               q_offsets;
        Buffer_<int>               k_offsets;
        std::vector<Tensor>        conv_states;
        std::vector<Tensor>        recurrent_states;
        Buffer_<void*>             conv_state_ptrs;
        Buffer_<void*>             recurrent_state_ptrs;
    };
    std::vector<Data> data_;

    // staging buffers
    Buffer_<void*> conv_state_ptrs_buf_;
    Buffer_<void*> recurrent_state_ptrs_buf_;

    int          sm_count_{1};
    Buffer_<int> work_counter_;

    cudaStream_t aux_stream_{};
    cudaEvent_t  ev_before_{};
    cudaEvent_t  ev_after_{};
};

}  // namespace turbomind
