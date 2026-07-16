#pragma once

#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/cache_registry.h"
#include "src/turbomind/kernels/linear_attn/delta_rule.h"
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
        // int                   layer_id;
    };

    GatedDeltaNetLayer(std::vector<DeltaNetWeight*> weights,
                       CacheRegistry&               registry,
                       const EngineParam&           engine,
                       const Context&               context,
                       int                          phases);

    ~GatedDeltaNetLayer();

    void Run(BatchOp op, int phase, TensorMap& env);

    void Forward(ForwardParam p);

private:
    void Setup(int phase, TensorMap& env);

    // Config passed at construction
    const int      tp_size_;
    const DataType recurrent_state_dtype_;

    LlamaLinear& linear_;

    // Per-phase batch data (mirrors UnifiedAttentionLayer pattern)
    struct Data {
        std::vector<int> input_lens;
        int batch_size{};
        std::vector<std::pair<uint8_t*, size_t>> reset_ptrs;
        Buffer_<int> q_offsets;
        Buffer_<int> k_offsets;
        Buffer_<bool> finished;
        Buffer_<void*> conv_state_ptrs;
        Buffer_<void*> recurrent_state_ptrs;
        int decode_count{};
        int prefill_count{};
        std::optional<linear_attn::delta_rule::Plan> recurrent_plan;
        std::optional<linear_attn::delta_rule::Plan> chunked_plan;
        core::Tensor chunked_workspace;
        Buffer_<uint8_t> recurrent_state_tma_descs;
    };
    std::vector<Data> data_;

    int    layer_num_{};         // == weights.size()
    int    rec_base_{};          // composite part id of layer 0's recurrent state (== 1)
    int    layers_per_block_{};  // L_b
    int    heads_per_block_{};   // H_b
    int    num_head_groups_{};   // ceil(num_v_heads / H_b)
    int    num_layer_groups_{};  // ceil(layer_num_ / L_b)
    int    num_blocks_{};        // num_layer_groups_ * num_head_groups_
    size_t block_bytes_{};       // one recurrent block's bytes (one composite part)
    size_t conv_total_bytes_{};  // accumulated conv-state bytes (part 0)

    std::unordered_map<const DeltaNetWeight*, int> layer_index_;  // weight ptr -> GDN-local layer index

    // staging buffers
    Buffer_<void*> conv_state_ptrs_buf_;
    Buffer_<void*> recurrent_state_ptrs_buf_;

    DataType input_dtype_{kNull};
    int      arch_{};
    int      num_k_heads_{};
    int      num_v_heads_{};
    int      head_dim_{};
    int      gate_stride_{};
    linear_attn::delta_rule::GatedDeltaRule delta_rule_;

    int          sm_count_{};
    Buffer_<int> work_counter_;

    cudaStream_t aux_stream_{};
    cudaEvent_t  ev_before_{};
    cudaEvent_t  ev_after_{};
};

}  // namespace turbomind
