// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include <cstddef>
#include <map>
#include <regex>
#include <string>

namespace turbomind {

struct ModelParam {
    size_t head_num;
    size_t head_dim;
    size_t kv_head_num;
    size_t hidden_units;
    size_t layer_num;
    size_t inter_size;
    size_t vocab_size;
    float  norm_eps;
    int    quant_policy;
    //
    int start_id;
    int end_id;
};

struct AttentionParam {
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
    int         cache_block_seq_len;
};

struct EngineParam {
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

struct LoraParam {
    int        r;
    float      scale;
    LoraPolicy policy;
    int        max_wo_r;

    std::map<std::string, std::pair<std::regex, int>>   rank_pattern;
    std::map<std::string, std::pair<std::regex, float>> scale_pattern;
};

}  // namespace turbomind
