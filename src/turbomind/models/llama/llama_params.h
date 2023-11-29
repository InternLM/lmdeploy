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

struct EngineParams {
    // batch params
    int max_batch_size;
    int session_len;
    int step_length;

    // cache params
    float cache_max_block_count;
    int   cache_chunk_size;

    // prefill chunking params
    int max_context_token_num;
    int prefill_token_count;
    int prefill_token_tolerance;
    int prefill_max_iterations;
};

}  // namespace turbomind
