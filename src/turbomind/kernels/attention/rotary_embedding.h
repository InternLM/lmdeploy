// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array_ops.h"

namespace turbomind {

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
            float tmp0 = cs_[i] * (float)x[i] - cs_[i + 1] * (float)x[i + 1];
            float tmp1 = cs_[i] * (float)x[i + 1] + cs_[i + 1] * (float)x[i];
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

template<class D, int N>
struct FastRoPE {

    static_assert(N % 2 == 0);

    Array<float, N / 2> inv_freq_;
    bool                is_valid_;

    __device__ FastRoPE(int   idx,
                        D     dims,
                        float base,
                        float ti_scale,
                        float llama3_inv_scaling_factor,
                        float llama3_alpha,
                        float llama3_beta,
                        std::integral_constant<int, N>)
    {
        is_valid_ = idx < dims;
        /// TODO: Take this away from device code
        const float scale_factor = -log2f(base) / dims;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2) {
            inv_freq_[i / 2] = ti_scale * exp2f((idx + i) * scale_factor);
        }
        // clang-format off
        /* The [llama3 rope](https://github.com/huggingface/transformers/blob/5f4ee98a7ade33e1c54fdd6181d04ee7b426b392/src/transformers/modeling_rope_utils.py#L298)
         * used by llama3.1 equals to the following equation, given the precommuted parameters as:
        ```C++
        inv_scaling_factor = 1 / factor;
        inv_diff_freq_factor = 1 / (high_freq_factor - low_freq_factor);
        alpha = old_context_len / (2 * PI) * inv_diff_freq_factor;
        beta = low_freq_factor * inv_diff_freq_factor
        ```
        */
        // clang-format on
        if (llama3_inv_scaling_factor) {
            PRAGMA_UNROLL
            for (int i = 0; i < N; i += 2) {
                auto freq        = inv_freq_[i / 2];
                auto smooth      = fmaxf(0.f, fminf(1.f, llama3_alpha * freq - llama3_beta));
                inv_freq_[i / 2] = (1 - smooth) * freq * llama3_inv_scaling_factor + smooth * freq;
            }
        }
    }

    template<typename T>
    __device__ void apply(Array<T, N>& x, float timestep)
    {
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2) {
            float c, s;
            sincosf(timestep * inv_freq_[i / 2], &s, &c);
            float tmp0 = c * (float)x[i] - s * (float)x[i + 1];
            float tmp1 = c * (float)x[i + 1] + s * (float)x[i];
            if (is_valid_) {
                x[i]     = (T)tmp0;
                x[i + 1] = (T)tmp1;
            }
        }
    }
};

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
