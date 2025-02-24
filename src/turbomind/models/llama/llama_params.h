// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstddef>
#include <map>
#include <regex>
#include <string>

#include "src/turbomind/models/llama/weight_type.h"

namespace turbomind {

struct MLAParam {
    size_t q_lora_rank;
    size_t kv_lora_rank;
    size_t qk_rope_dim;
    size_t v_head_dim;
};

struct ModelParam {
    size_t     head_num;
    size_t     head_dim;
    size_t     kv_head_num;
    size_t     hidden_units;
    size_t     layer_num;
    size_t     vocab_size;
    size_t     embedding_size;
    float      norm_eps;
    int        quant_policy;
    bool       attn_bias;
    WeightType weight_type;
    int        group_size;
    MLAParam   mla;
    int        tune_layer_num;

    std::vector<int> inter_size;
};

struct MoeParam {
    enum Method
    {
        kNaive,
        kFused
    } method;

    int   experts_per_token;
    int   inter_size;
    bool  norm_topk_prob;
    bool  shared_gate;
    float routed_scale;

    int         topk_group;
    std::string topk_method;
    int         n_group;

    std::vector<int> expert_num;
};

struct AttentionParam {
    int         rotary_embedding_dim;
    float       rotary_embedding_base;
    int         max_position_embeddings;
    float       softmax_scale;
    std::string rope_scaling_type;
    int         original_max_position_embeddings;
    float       rope_scaling_factor;
    float       low_freq_factor;
    float       high_freq_factor;
    float       attention_factor;
    float       beta_fast;
    float       beta_slow;
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

enum class LoraPolicy : int
{
    kNull,
    kPlora,
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
