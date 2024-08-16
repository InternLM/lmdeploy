// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include <map>
#include <regex>
#include <string>

namespace turbomind {

struct LlamaAttentionParams {
    int         rotary_embedding_dim;
    float       rotary_embedding_base;
    int         max_position_embeddings;
    std::string rope_scaling_type;
    int         original_max_position_embeddings;
    float       rope_scaling_factor;
    float       low_freq_factor;
    float       high_freq_factor;
    bool        use_dynamic_ntk;
    bool        use_logn_attn;
};

struct EngineParams {
    // batch params
    int max_batch_size;
    int session_len;
    int step_length;

    // cache params
    float cache_max_block_count;
    int   cache_chunk_size;
    bool  enable_prefix_caching;

    // chunking params
    int max_prefill_token_num;
    int max_context_token_num;
    int num_tokens_per_iter;
    int max_prefill_iters;
};

struct LoraParams {
    int        r;
    float      scale;
    LoraPolicy policy;
    int        max_wo_r;

    std::map<std::string, std::pair<std::regex, int>>   rank_pattern;
    std::map<std::string, std::pair<std::regex, float>> scale_pattern;
};

}  // namespace turbomind
