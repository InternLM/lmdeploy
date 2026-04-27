// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <array>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

namespace turbomind::core {

using MropeSection = std::array<int, 3>;

struct RopeConfig {
#define ROPE_FIELDS(X)                                                                                                 \
    X(int, type, 0)                                                                                                    \
    X(float, base, 10000.f)                                                                                            \
    X(int, dim, 0)                                                                                                     \
    X(float, factor, 1.f)                                                                                              \
    X(int, max_position_embeddings, 0)                                                                                 \
    X(float, yarn_attention_factor, 1.f)                                                                               \
    X(float, yarn_beta_fast, 32.f)                                                                                     \
    X(float, yarn_beta_slow, 1.f)                                                                                      \
    X(float, llama3_low_freq_factor, 1.f)                                                                              \
    X(float, llama3_high_freq_factor, 4.f)                                                                             \
    X(int, llama3_original_max_position_embeddings, 0)                                                                 \
    X(MropeSection, mrope_section, {})

    ROPE_FIELDS(TM_MEMBER)
    TM_FOR_EACH(RopeConfig, ROPE_FIELDS)

#undef ROPE_FIELDS
};

struct AttentionConfig: ModuleConfig {
    AttentionConfig(): ModuleConfig{"AttentionWeight"} {}

#define ATTENTION_FIELDS(X)                                                                                            \
    X(int, hidden_dim)                                                                                                 \
    X(int, head_dim)                                                                                                   \
    X(int, head_num)                                                                                                   \
    X(int, kv_head_num)                                                                                                \
    X(int, kv_lora_rank)                                                                                               \
    X(int, q_lora_rank)                                                                                                \
    X(int, qk_rope_dim)                                                                                                \
    X(int, v_head_dim)                                                                                                 \
    X(bool, has_bias)                                                                                                  \
    X(bool, qk_norm)                                                                                                   \
    X(int, tp_size)                                                                                                    \
    X(int, tp_rank)                                                                                                    \
    X(DataType, data_type)                                                                                             \
    X(int, window_size, -1)                                                                                            \
    X(bool, attn_sink)                                                                                                 \
    X(bool, attn_output_gate)                                                                                          \
    X(RopeConfig, rope, {})                                                                                            \
    X(int, repeat_kv)                                                                                                  \
    X(int, qk_nope_dim)                                                                                                \
    X(float, softmax_scale, 0.f)                                                                                       \
    X(bool, use_logn_attn, false)

    ATTENTION_FIELDS(TM_MEMBER)
    TM_FOR_EACH(AttentionConfig, ATTENTION_FIELDS)

#undef ATTENTION_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

struct RopeKernelParam;
void init_rope_kernel_param(const core::RopeConfig& rope, RopeKernelParam& rope_kernel);

class AttentionWeight: public core::Module {
public:
    const char* type() const override
    {
        return "AttentionWeight";
    }

    AttentionWeight() = default;

    AttentionWeight(const core::AttentionConfig& cfg);

    void prepare() override;

    // --- X-macro field lists ---
#define ATTENTION_WEIGHT_CHILDREN(X)                                                                                   \
    X(LinearWeight, w_qkv)                                                                                             \
    X(LinearWeight, wo)                                                                                                \
    X(LinearWeight, q_proj)                                                                                            \
    X(LinearWeight, q_a_proj)                                                                                          \
    X(LinearWeight, q_b_proj)                                                                                          \
    X(LinearWeight, kv_a_proj)                                                                                         \
    X(NormWeight, q_norm)                                                                                              \
    X(NormWeight, k_norm)                                                                                              \
    X(NormWeight, q_a_layernorm)                                                                                       \
    X(NormWeight, kv_a_layernorm)

#define ATTENTION_WEIGHT_PARAMS(X) X(sinks)

    TM_MODULE_DECLARE(AttentionWeight, ATTENTION_WEIGHT_CHILDREN, ATTENTION_WEIGHT_PARAMS)

    bool is_mla() const
    {
        return kv_lora_rank > 0;
    }

    // --- Config fields (public for runtime access) ---
    int              hidden_dim{};
    int              head_dim{};
    int              head_num{};
    int              kv_head_num{};
    int              kv_lora_rank{};
    int              q_lora_rank{};
    int              qk_rope_dim{};
    int              v_head_dim{};
    bool             bias{};
    bool             qk_norm{};
    int              tp_size{};
    int              tp_rank{};
    DataType         data_type{};
    int              window_size{};
    bool             sink{};
    bool             attn_output_gate{};
    float            softmax_scale{};
    bool             use_logn_attn{};
    core::RopeConfig rope{};
};

}  // namespace turbomind
