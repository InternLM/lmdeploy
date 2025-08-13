// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/models/llama/llama_rope.h"

namespace turbomind {

template<int N>
__device__ void init_default(Array<float, N / 2>& inv_freq, int idx, RopeKernelParam& param)
{
    auto scale_factor = param.scale_factor;
    auto inv_factor   = param.inv_factor;
    PRAGMA_UNROLL
    for (int i = 0; i < N; i += 2) {
        inv_freq[i / 2] = inv_factor * exp2f((idx + i) * scale_factor);
    }
}

template<int N>
__device__ void init_yarn(Array<float, N / 2>& inv_freq, int idx, RopeKernelParam& param)
{
    auto scale_factor            = param.scale_factor;
    auto inv_factor              = param.inv_factor;
    auto ramp_inv_factor_div_2   = param.yarn.ramp_inv_factor_div_2;
    auto ramp_inv_factor_mul_min = param.yarn.ramp_inv_factor_mul_min;

    PRAGMA_UNROLL
    for (int i = 0; i < N; i += 2) {
        auto freq       = exp2f((idx + i) * scale_factor);
        auto alpha      = (idx + i) * ramp_inv_factor_div_2 - ramp_inv_factor_mul_min;
        alpha           = fmaxf(0.f, fminf(1.f, alpha));
        inv_freq[i / 2] = freq - freq * alpha * (1.f - inv_factor);
    }
}

template<int N>
__device__ void init_llama3(Array<float, N / 2>& inv_freq, int idx, RopeKernelParam& param)
{
    auto scale_factor = param.scale_factor;
    auto inv_factor   = param.inv_factor;
    auto alpha        = param.llama3.alpha;
    auto beta         = param.llama3.beta;

    PRAGMA_UNROLL
    for (int i = 0; i < N; i += 2) {
        auto freq       = exp2f((idx + i) * scale_factor);
        auto smooth     = fmaxf(0.f, fminf(1.f, alpha * freq - beta));
        inv_freq[i / 2] = (1 - smooth) * freq * inv_factor + smooth * freq;
    }
}

template<int N>
struct FastRoPE {

    static_assert(N % 2 == 0);

    RopeKernelParam     param_;
    Array<float, N / 2> inv_freq_;
    bool                is_valid_;
    float               attention_scaling_{1.f};
    int                 idx_;

    typedef void (*Func)(Array<float, N / 2>&, int, RopeKernelParam&);
    Func fill_func_;

    __device__ FastRoPE(const RopeKernelParam& param, int batch_idx, std::integral_constant<int, N>): param_(param)
    {

        if (param_.type == RopeType::kDynamic) {
            float base          = param_.base[batch_idx];
            param_.scale_factor = -log2f(base) / param_.dim;
        }
        else if (param_.type == RopeType::kYarn) {
            attention_scaling_ = param_.yarn.attention_factor;
        }
        else if (param_.type == RopeType::kMrope) {
            param_.mrope.position_ids += batch_idx * param_.mrope.stride;
            param_.mrope.position_delta += batch_idx;
            param_.mrope.length += batch_idx;
        }
    }

    __device__ void init(int idx)
    {
        is_valid_ = idx < param_.dim;
        idx_      = idx;
        switch (param_.type) {
            case RopeType::kDefault:
            case RopeType::kLinear:
            case RopeType::kDynamic:
            case RopeType::kMrope:
                init_default<N>(inv_freq_, idx, param_);
                break;
            case RopeType::kYarn:
                init_yarn<N>(inv_freq_, idx, param_);
                break;
            case RopeType::kLlama3:
                init_llama3<N>(inv_freq_, idx, param_);
                break;
        }
    }

    template<typename T>
    __device__ void apply(Array<T, N>& x, float timestep)
    {
        if (param_.type == RopeType::kMrope) {
            return apply_mrope(x, timestep);
        }
        // Most models apply rotary embedding in half precision
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2) {
            float c, s;
            sincosf(timestep * inv_freq_[i / 2], &s, &c);
            s *= attention_scaling_;
            c *= attention_scaling_;
            T tmp0 = (T)c * x[i] - (T)s * x[i + 1];
            T tmp1 = (T)c * x[i + 1] + (T)s * x[i];
            if (is_valid_) {
                x[i]     = tmp0;
                x[i + 1] = tmp1;
            }
        }
    }

    template<typename T>
    __device__ void apply_mrope(Array<T, N>& x, float timestep)
    {
        int  tt, th, tw;
        int3 section = param_.mrope.section;
        if (timestep < *param_.mrope.length) {
            const int* t = param_.mrope.position_ids + 3 * (int)timestep;
            tt           = t[0];
            th           = t[1];
            tw           = t[2];
        }
        else {
            tt = th = tw = (int)timestep + (*param_.mrope.position_delta);
        }

        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2) {
            if (i + idx_ < section.x) {
                timestep = (float)tt;
            }
            else if (i + idx_ < section.y) {
                timestep = (float)th;
            }
            else {
                timestep = (float)tw;
            }
            float c, s;
            sincosf(timestep * inv_freq_[i / 2], &s, &c);
            T tmp0 = (T)c * x[i] - (T)s * x[i + 1];
            T tmp1 = (T)c * x[i + 1] + (T)s * x[i];
            if (is_valid_) {
                x[i]     = tmp0;
                x[i + 1] = tmp1;
            }
        }
    }
};

template<int N>
struct RotaryEmbedding {

    static_assert(N % 2 == 0);

    Array<float, N> cs_;

    bool is_valid_;

    __device__ RotaryEmbedding(float base, int dims, int timestep, int2 offset)
    {
        const int idx = offset.x;
        is_valid_     = idx < dims;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2) {
            const float2 tmp = get_coefficient(idx + i, dims, base, timestep);
            cs_[i]           = tmp.x;
            cs_[i + 1]       = tmp.y;
        }
    }

    // ! depending on the context, this function may generate different result when inlined
    static __device__ __noinline__ float2 get_coefficient(int idx, int dims, float base, int timestep)
    {
        const float inv_freq = timestep / powf(base, idx / (float)dims);
        float2      cs;
        sincosf(inv_freq, &cs.y, &cs.x);
        return cs;
    }

    template<typename T>
    __device__ void apply(Array<T, N>& x)
    {

        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2) {
            auto tmp0 = (T)cs_[i] * x[i] - (T)cs_[i + 1] * x[i + 1];
            auto tmp1 = (T)cs_[i] * x[i + 1] + (T)cs_[i + 1] * x[i];
            if (is_valid_) {
                x[i]     = (T)tmp0;
                x[i + 1] = (T)tmp1;
            }
        }
    }
};
template<class C, class T>
__device__ void ApplyRotaryEmbedding(Array<T, 4>& x, float base, int dims, int ti, int di)
{
    PRAGMA_UNROLL
    for (int d1 = 0; d1 < 2; ++d1) {
        int    d        = d1 * 8 + di;
        float  inv_freq = ti / powf(base, d / (float)dims);
        float2 cs;
        sincosf(inv_freq, &cs.y, &cs.x);
        C x1          = (C)cs.x * (C)x[d1 * 2 + 0] - (C)cs.y * (C)x[d1 * 2 + 1];
        C x2          = (C)cs.x * (C)x[d1 * 2 + 1] + (C)cs.y * (C)x[d1 * 2 + 0];
        x[d1 * 2 + 0] = (T)x1;
        x[d1 * 2 + 1] = (T)x2;
    }
}

template<int N, int C = 8>
struct RoPE {
    Array<float, N> inv_freqs_;

    RoPE() = default;
    __device__ RoPE(float idx, float base, float dims)
    {
        for (int i = 0; i < N; ++i) {
            inv_freqs_[i] = powf(base, idx / dims + (C / dims) * i);
        }
    }

    template<class T>
    __device__ void apply(Array<T, N * 2>& x, float timestep)
    {
        for (int i = 0; i < N; ++i) {
            const float inv_freq = timestep * inv_freqs_[i];
            float2      cs;
            sincosf(inv_freq, &cs.y, &cs.x);
            float tmp0   = cs.x * (float)x[i * 2] - cs.y * (float)x[i * 2 + 1];
            float tmp1   = cs.x * (float)x[i * 2 + 1] + cs.y * (float)x[i * 2];
            x[i * 2]     = (T)tmp0;
            x[i * 2 + 1] = (T)tmp1;
        }
    }
};

struct LogNScaling {

    float scale_;

    __device__ static float get_scale(int seq_len, int max_position_embeddings)
    {
        if (seq_len <= max_position_embeddings) {
            return 1.f;
        }
        else {
            return log2f(seq_len) / log2f(max_position_embeddings);
        }
    }

    __device__ LogNScaling(int seq_len, int max_position_embeddings)
    {
        scale_ = get_scale(seq_len, max_position_embeddings);
    }

    template<typename T, int N>
    __device__ void apply(Array<T, N>& x) const
    {
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            x[i] = (T)((float)x[i] * scale_);
        }
    }
};

}  // namespace turbomind
