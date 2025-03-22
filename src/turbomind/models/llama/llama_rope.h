// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cmath>
#include <map>
#include <string>

namespace turbomind {

enum class RopeType
{
    kNull,
    kDefault,
    kLinear,
    kDynamic,
    kYarn,
    kLlama3,
};

inline RopeType GetRoPEType(const std::string& type)
{
    std::map<std::string, RopeType> lookup = {{"default", RopeType::kDefault},
                                              {"linear", RopeType::kLinear},
                                              {"dynamic", RopeType::kDynamic},
                                              {"yarn", RopeType::kYarn},
                                              {"llama3", RopeType::kLlama3}};
    return lookup.at(type);
}

struct YarnRopeParam {
    float attention_factor;
    float beta_fast;
    float beta_slow;
};

struct Llama3RopeParam {
    float low_freq_factor;
    float high_freq_factor;
    int   original_max_position_embeddings;
};

struct RopeParam {
    RopeType type;
    // common
    float base;
    int   dim;
    float factor;
    int   max_position_embeddings;
    // unique
    union {
        YarnRopeParam   yarn;
        Llama3RopeParam llama3;
    };
};

struct YarnRopeKernelParam {
    float scale_factor;
    float attention_factor;
    float ramp_inv_factor_div_2;
    float ramp_inv_factor_mul_min;
};

struct Llama3RopeKernelParam {
    float scale_factor;
    float alpha;
    float beta;
};

struct RopeKernelParam {
    RopeType type;

    float* base{};  // for dynamic ntk
    int    dim;
    float  scale_factor;
    float  inv_factor;

    YarnRopeKernelParam   yarn;
    Llama3RopeKernelParam llama3;
};

inline void init_rope_kernel_param(const RopeParam& rope, RopeKernelParam& rope_kernel)
{
    rope_kernel.type         = rope.type;
    rope_kernel.dim          = rope.dim;
    rope_kernel.scale_factor = -std::log2f(rope.base) / rope.dim;
    if (rope.type == RopeType::kDynamic) {
        rope_kernel.inv_factor = 1.f;
    }
    else {
        rope_kernel.inv_factor = (rope.factor != 0.f) ? 1.0 / rope.factor : 1.f;
    }

    if (rope.type == RopeType::kYarn) {
        auto&        src = rope.yarn;
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
        find_correction_range(src.beta_fast, src.beta_slow, low, high);
        // https://github.com/huggingface/transformers/blob/6c3f168b36882f0beebaa9121eafa1928ba29633/src/transformers/modeling_rope_utils.py#L216
        if (low == high) {
            high += 0.001f;
        }
        dst.ramp_inv_factor_div_2   = 1.0 / (high - low) / 2.0;
        dst.ramp_inv_factor_mul_min = 1.0 / (high - low) * low;
        dst.attention_factor        = src.attention_factor;
    }
    else if (rope.type == RopeType::kLlama3) {
        auto& src = rope.llama3;
        auto& dst = rope_kernel.llama3;

        const double PI                   = 3.14159265358979323846;
        float        inv_diff_freq_factor = 1.0 / (src.high_freq_factor - src.low_freq_factor);
        dst.alpha                         = src.original_max_position_embeddings / (2 * PI) * inv_diff_freq_factor;
        dst.beta                          = src.low_freq_factor * inv_diff_freq_factor;
    }
}

}  // namespace turbomind
