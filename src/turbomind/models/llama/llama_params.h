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

    // Weight types for mixed quantization support.
    // Models like mixed AWQ (e.g. QuantTrio GLM-4.7-Flash) quantize FFN/expert
    // weights to int4 but keep attention weights as fp16. GptOss mxfp4 quantizes
    // only MoE experts to e2m1 while keeping attention and shared experts as fp16.
    //
    //                  weight_type   ffn_weight_type   expert_weight_type
    //  Pure fp16       float16       float16           float16
    //  Full AWQ        int4          int4              int4
    //  Mixed AWQ       float16       int4              int4
    //  GptOss mxfp4    bfloat16      bfloat16          e2m1
    DataType weight_type;           // attention weights
    DataType expert_weight_type;    // MoE routed expert weights
    DataType ffn_weight_type;       // dense FFN / shared expert weights

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
    std::string scoring_func;
    int         router_n_groups;

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
    int attn_cp_size;
    int attn_cp_rank;
    int mlp_tp_size;
    int mlp_tp_rank;

    // multi-node
    int nnodes;
    int node_rank;

    std::vector<int> devices;
};

}  // namespace turbomind
