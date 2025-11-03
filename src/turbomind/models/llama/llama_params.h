// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstddef>
#include <map>
#include <regex>
#include <string>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/models/llama/llama_rope.h"

namespace turbomind {

struct MLAParam {
    int q_lora_rank;
    int kv_lora_rank;
    int qk_rope_dim;
    int v_head_dim;
};

struct ModelParam {
    size_t   head_num;
    size_t   head_dim;
    size_t   kv_head_num;
    size_t   hidden_units;
    size_t   layer_num;
    size_t   vocab_size;
    size_t   embedding_size;
    float    norm_eps;
    int      quant_policy;
    bool     attn_bias;
    bool     attn_sink;
    bool     mlp_bias;
    DataType data_type;
    DataType weight_type;
    DataType expert_weight_type;
    int      group_size;
    MLAParam mla;
    bool     qk_norm;
    int      tune_layer_num;

    ActivationType act_type;

    std::vector<int> window_size;
    std::vector<int> inter_size;
};

/// TODO: rename all `gate` in the context of MoE router to `router`
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

    bool router_bias;

    int         topk_group;
    std::string topk_method;
    int         n_group;

    std::vector<int> expert_num;
};

struct AttentionParam {
    float softmax_scale;
    int   cache_block_seq_len;
    // logn attention
    bool use_logn_attn;
    int  max_position_embeddings;
    // rotary embedding
    RopeParam rope;
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
    bool  enable_metrics;

    // chunking params
    int max_forward_token_num;
    int max_context_token_num;
    int num_tokens_per_iter;
    int max_prefill_iters;

    // parallel params
    int outer_dp_size;
    int outer_dp_rank;
    int attn_dp_size;
    int attn_dp_rank;
    int attn_tp_size;
    int attn_tp_rank;
    int mlp_tp_size;
    int mlp_tp_rank;

    std::vector<int> devices;
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
