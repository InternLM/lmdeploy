// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

struct LlamaAttentionParams {
    int   rotary_embedding_dim;
    float rotary_embedding_base;
    int   max_position_embeddings;
    float rope_scaling_factor;
    // bool  use_dynamic_ntk;
    bool use_logn_attn;
};

}  // namespace turbomind
