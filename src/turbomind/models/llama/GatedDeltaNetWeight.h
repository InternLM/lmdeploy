#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"

namespace turbomind {

struct GatedDeltaNetWeight: public core::Module {

    GatedDeltaNetWeight() = default;

    GatedDeltaNetWeight(int      hidden_dim,
                        int      num_k_heads,
                        int      num_v_heads,
                        int      key_head_dim,
                        int      value_head_dim,
                        int      d_conv,
                        bool     bias,
                        int      tp_size,
                        int      tp_rank,
                        DataType data_type,
                        DataType weight_type,
                        int      group_size);

    void prepare();

    // Projections matching HF: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
    LlamaDenseWeight in_proj_qkv;  // hidden -> key_dim*2 + value_dim
    LlamaDenseWeight in_proj_z;    // hidden -> value_dim (output gate)
    LlamaDenseWeight in_proj_b;    // hidden -> num_v_heads (beta, per-head scalar)
    LlamaDenseWeight in_proj_a;    // hidden -> num_v_heads (alpha/dt, per-head scalar)
    LlamaDenseWeight out_proj;     // value_dim -> hidden

    // Non-dense parameters
    Tensor conv1d;   // depthwise conv weights: (conv_dim, 1, d_conv) flattened
    Tensor A_log;    // log-space decay: (num_v_heads,)
    Tensor dt_bias;  // dt bias: (num_v_heads,)
    Tensor norm;     // RMSNormGated weight: (value_head_dim,)

    int tp_rank_;
    int tp_size_;
};

}  // namespace turbomind
