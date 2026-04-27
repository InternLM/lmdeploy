// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/attention_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/models/llama/llama_rope.h"

namespace turbomind {

AttentionWeight::AttentionWeight(const core::AttentionConfig& cfg):
    hidden_dim(cfg.hidden_dim),
    head_dim(cfg.head_dim),
    head_num(cfg.head_num),
    kv_head_num(cfg.kv_head_num),
    kv_lora_rank(cfg.kv_lora_rank),
    q_lora_rank(cfg.q_lora_rank),
    qk_rope_dim(cfg.qk_rope_dim),
    v_head_dim(cfg.v_head_dim),
    bias(cfg.has_bias),
    qk_norm(cfg.qk_norm),
    tp_size(cfg.tp_size),
    tp_rank(cfg.tp_rank),
    data_type(cfg.data_type),
    window_size(cfg.window_size),
    sink(cfg.attn_sink),
    attn_output_gate(cfg.attn_output_gate),
    softmax_scale(cfg.softmax_scale),
    use_logn_attn(cfg.use_logn_attn),
    rope(cfg.rope)
{
}

void AttentionWeight::prepare()
{
    Module::prepare();
}

void init_rope_kernel_param(const core::RopeConfig& rope, RopeKernelParam& rope_kernel)
{
    auto rope_type = static_cast<RopeType>(rope.type);

    rope_kernel.type         = rope_type;
    rope_kernel.dim          = rope.dim;
    rope_kernel.scale_factor = -std::log2(rope.base) / rope.dim;
    if (rope_type == RopeType::kDynamic) {
        rope_kernel.inv_factor = 1.f;
    }
    else {
        rope_kernel.inv_factor = (rope.factor != 0.f) ? 1.0 / rope.factor : 1.f;
    }

    if (rope_type == RopeType::kYarn) {
        auto&        dst = rope_kernel.yarn;
        const double PI  = 3.14159265358979323846;

        auto find_correction_dim = [&](float num_rotations) {
            return (rope.dim * std::log(rope.max_position_embeddings / (num_rotations * 2 * PI)))
                   / (2 * std::log(rope.base));
        };

        auto find_correction_range = [&](float low_rot, float high_rot, float& low, float& high) {
            low  = std::floor(find_correction_dim(low_rot));
            high = std::ceil(find_correction_dim(high_rot));
            low  = std::max(low, 0.f);
            high = std::min(high, rope.dim - 1.f);
        };

        float low, high;
        find_correction_range(rope.yarn_beta_fast, rope.yarn_beta_slow, low, high);
        if (low == high) {
            high += 0.001f;
        }
        dst.ramp_inv_factor_div_2   = 1.0 / (high - low) / 2.0;
        dst.ramp_inv_factor_mul_min = 1.0 / (high - low) * low;
        dst.attention_factor        = rope.yarn_attention_factor;
    }
    else if (rope_type == RopeType::kLlama3) {
        auto& dst = rope_kernel.llama3;

        float inv_diff_freq_factor = 1.0 / (rope.llama3_high_freq_factor - rope.llama3_low_freq_factor);
        dst.alpha = rope.llama3_original_max_position_embeddings / (2 * 3.14159265358979323846) * inv_diff_freq_factor;
        dst.beta  = rope.llama3_low_freq_factor * inv_diff_freq_factor;
    }
    else if (rope_type == RopeType::kMrope) {
        auto& dst     = rope_kernel.mrope;
        dst.section.x = rope.mrope_section[0] * 2;
        dst.section.y = rope.mrope_section[1] * 2 + dst.section.x;
        dst.section.z = rope.mrope_section[2] * 2 + dst.section.y;
    }
}

TM_MODULE_REGISTER(AttentionWeight, core::AttentionConfig);

TM_MODULE_METHODS(AttentionWeight, ATTENTION_WEIGHT_CHILDREN, ATTENTION_WEIGHT_PARAMS)

}  // namespace turbomind
