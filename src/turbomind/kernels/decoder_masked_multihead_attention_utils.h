/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "src/turbomind/utils/cuda_bf16_wrapper.h"
#include "src/turbomind/utils/cuda_fp8_utils.h"
#include "src/turbomind/utils/cuda_type_utils.cuh"
#include <stdint.h>

using namespace turbomind;

namespace mmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Float8_ {
    float2 x;
    float2 y;
    float2 z;
    float2 w;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Float4_ {
    float2 x;
    float2 y;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct num_elems;
template<>
struct num_elems<float> {
    static constexpr int value = 1;
};
template<>
struct num_elems<float2> {
    static constexpr int value = 2;
};
template<>
struct num_elems<float4> {
    static constexpr int value = 4;
};
template<>
struct num_elems<Float4_> {
    static constexpr int value = 4;
};
template<>
struct num_elems<Float8_> {
    static constexpr int value = 8;
};

template<>
struct num_elems<uint32_t> {
    static constexpr int value = 2;
};
template<>
struct num_elems<uint2> {
    static constexpr int value = 4;
};
template<>
struct num_elems<uint4> {
    static constexpr int value = 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int N>
struct packed_type;
template<typename T>
struct packed_type<T, 1> {
    using type = T;
};
template<>
struct packed_type<int8_t, 2> {
    using type = int16_t;
};
template<>
struct packed_type<int8_t, 4> {
    using type = int32_t;
};
template<>
struct packed_type<int8_t, 8> {
    using type = int64_t;
};

template<>
struct packed_type<float, 2> {
    using type = float2;
};
template<>
struct packed_type<float, 4> {
    using type = float4;
};
template<>
struct packed_type<float, 8> {
    using type = Float8_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float add(float a, float b)
{
    return a + b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 add(float2 a, float2 b)
{
    float2 c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 add(float4 a, float4 b)
{
    float4 c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    c.z = add(a.z, b.z);
    c.w = add(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint16_t add(uint16_t a, uint16_t b)
{
    uint16_t c;
    asm volatile("add.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t add(uint32_t a, uint32_t b)
{
    uint32_t c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 add(uint2 a, uint2 b)
{
    uint2 c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 add(uint4 a, uint4 b)
{
    uint4 c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    c.z = add(a.z, b.z);
    c.w = add(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint16_t float_to_half(float f)
{
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp;
#if 0 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800  // Is it better?
    float zero = 0.f;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(zero), "f"(f));
#else
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#endif
    return tmp.u16[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t float2_to_half2(float2 f)
{
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
#else
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
#endif
    return tmp.u32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float half_to_float(uint16_t h)
{
    float f;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
    return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 half2_to_float2(uint32_t v)
{
    uint16_t lo, hi;
    asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
    return make_float2(half_to_float(lo), half_to_float(hi));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float add(float a, uint16_t b)
{
    return a + half_to_float(b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 add(uint32_t a, float2 fb)
{
    float2 fa = half2_to_float2(a);
    return add(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ add(uint2 a, Float4_ fb)
{
    Float4_ fc;
    fc.x = add(a.x, fb.x);
    fc.y = add(a.y, fb.y);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ add(uint4 a, Float8_ fb)
{
    Float8_ fc;
    fc.x = add(a.x, fb.x);
    fc.y = add(a.y, fb.y);
    fc.z = add(a.z, fb.z);
    fc.w = add(a.w, fb.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t h0_h0(uint16_t a)
{
    uint32_t b;
    asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));
    return b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float fma(float a, float b, float c)
{
    return a * b + c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(float2 a, float2 b, float2 c)
{
    float2 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(float a, float2 b, float2 c)
{
    float2 d;
    d.x = fma(a, b.x, c.x);
    d.y = fma(a, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float4 a, float4 b, float4 c)
{
    float4 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    d.z = fma(a.z, b.z, c.z);
    d.w = fma(a.w, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float a, float4 b, float4 c)
{
    float4 d;
    d.x = fma(a, b.x, c.x);
    d.y = fma(a, b.y, c.y);
    d.z = fma(a, b.z, c.z);
    d.w = fma(a, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float a, float4 b, Float4_ c)
{
    float4 d;
    d.x = fma(a, b.x, c.x.x);
    d.y = fma(a, b.y, c.x.y);
    d.z = fma(a, b.z, c.y.x);
    d.w = fma(a, b.w, c.y.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(float a, Float4_ b, Float4_ c)
{
    Float4_ d;
    d.x = fma(a, b.x, c.x);
    d.y = fma(a, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(float a, Float8_ b, Float8_ c)
{
    Float8_ d;
    d.x = fma(a, b.x, c.x);
    d.y = fma(a, b.y, c.y);
    d.z = fma(a, b.z, c.z);
    d.w = fma(a, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t d;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c)
{
    return fma(h0_h0(a), b, c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c)
{
    uint2 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c)
{
    uint32_t s = h0_h0(a);
    uint2    d;
    d.x = fma(s, b.x, c.x);
    d.y = fma(s, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c)
{
    uint4 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    d.z = fma(a.z, b.z, c.z);
    d.w = fma(a.w, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c)
{
    uint32_t s = h0_h0(a);
    uint4    d;
    d.x = fma(s, b.x, c.x);
    d.y = fma(s, b.y, c.y);
    d.z = fma(s, b.z, c.z);
    d.w = fma(s, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float fma(uint16_t a, uint16_t b, float fc)
{
    float fa = half_to_float(a);
    float fb = half_to_float(b);
    return fa * fb + fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(uint32_t a, uint32_t b, float2 fc)
{
    float2 fa = half2_to_float2(a);
    float2 fb = half2_to_float2(b);
    return fma(fa, fb, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(uint16_t a, uint32_t b, float2 fc)
{
    return fma(h0_h0(a), b, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(uint2 a, uint2 b, Float4_ fc)
{
    Float4_ fd;
    fd.x = fma(a.x, b.x, fc.x);
    fd.y = fma(a.y, b.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc)
{
    uint32_t s = h0_h0(a);
    Float4_  fd;
    fd.x = fma(s, b.x, fc.x);
    fd.y = fma(s, b.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint4 a, uint4 b, Float8_ fc)
{
    Float8_ fd;
    fd.x = fma(a.x, b.x, fc.x);
    fd.y = fma(a.y, b.y, fc.y);
    fd.z = fma(a.z, b.z, fc.z);
    fd.w = fma(a.w, b.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint16_t a, uint4 b, Float8_ fc)
{
    uint32_t s = h0_h0(a);
    Float8_  fd;
    fd.x = fma(s, b.x, fc.x);
    fd.y = fma(s, b.y, fc.y);
    fd.z = fma(s, b.z, fc.z);
    fd.w = fma(s, b.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b)
{
    return Acc{};  // for compile
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float mul<float, float>(float a, float b)
{
    return a * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(float2 a, float2 b)
{
    float2 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(float a, float2 b)
{
    float2 c;
    c.x = a * b.x;
    c.y = a * b.y;
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 mul(float4 a, float4 b)
{
    float4 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    c.z = a.z * b.z;
    c.w = a.w * b.w;
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 mul(float a, float4 b)
{
    float4 c;
    c.x = a * b.x;
    c.y = a * b.y;
    c.z = a * b.z;
    c.w = a * b.w;
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(float a, Float8_ b)
{
    Float8_ c;
    c.x = mul<float2, float, float2>(a, b.x);
    c.y = mul<float2, float, float2>(a, b.y);
    c.z = mul<float2, float, float2>(a, b.z);
    c.w = mul<float2, float, float2>(a, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint16_t mul(uint16_t a, uint16_t b)
{
    uint16_t c;
    asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint32_t mul(uint32_t a, uint32_t b)
{
    uint32_t c;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint32_t mul(uint16_t a, uint32_t b)
{
    return mul<uint32_t, uint32_t, uint32_t>(h0_h0(a), b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint2 mul(uint2 a, uint2 b)
{
    uint2 c;
    c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
    c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint2 mul(uint16_t a, uint2 b)
{
    uint32_t s = h0_h0(a);
    uint2    c;
    c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
    c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint4 mul(uint4 a, uint4 b)
{
    uint4 c;
    c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
    c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
    c.z = mul<uint32_t, uint32_t, uint32_t>(a.z, b.z);
    c.w = mul<uint32_t, uint32_t, uint32_t>(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint4 mul(uint16_t a, uint4 b)
{
    uint32_t s = h0_h0(a);
    uint4    c;
    c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
    c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
    c.z = mul<uint32_t, uint32_t, uint32_t>(s, b.z);
    c.w = mul<uint32_t, uint32_t, uint32_t>(s, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float mul(uint16_t a, uint16_t b)
{
    float fa = half_to_float(a);
    float fb = half_to_float(b);
    return fa * fb;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float mul(uint16_t a, float b)
{
    return half_to_float(a) * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(uint32_t a, uint32_t b)
{
    float2 fa = half2_to_float2(a);
    float2 fb = half2_to_float2(b);
    return mul<float2, float2, float2>(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(uint16_t a, uint32_t b)
{
    return mul<float2, uint32_t, uint32_t>(h0_h0(a), b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(uint2 a, uint2 b)
{
    Float4_ fc;
    fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(uint16_t a, uint2 b)
{
    uint32_t s = h0_h0(a);
    Float4_  fc;
    fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(uint4 a, uint4 b)
{
    Float8_ fc;
    fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
    fc.z = mul<float2, uint32_t, uint32_t>(a.z, b.z);
    fc.w = mul<float2, uint32_t, uint32_t>(a.w, b.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(uint16_t a, uint4 b)
{
    uint32_t s = h0_h0(a);
    Float8_  fc;
    fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
    fc.z = mul<float2, uint32_t, uint32_t>(s, b.z);
    fc.w = mul<float2, uint32_t, uint32_t>(s, b.w);
    return fc;
}

inline __device__ float sum(float v)
{
    return v;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float2 v)
{
    return v.x + v.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float4 v)
{
    return v.x + v.y + v.z + v.w;
}

inline __device__ float sum(uint16_t v)
{
    return half_to_float(v);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint32_t v)
{
    float2 tmp = half2_to_float2(v);
    return tmp.x + tmp.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint2 v)
{
    uint32_t c = add(v.x, v.y);
    return sum(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint4 v)
{
#if 1
    uint32_t c = add(v.x, v.y);
    c          = add(c, v.z);
    c          = add(c, v.w);
#else
    uint32_t c = add(v.x, v.y);
    uint32_t d = add(v.z, v.w);
    c          = add(c, d);
#endif
    return sum(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(Float4_ v)
{
    return v.x.x + v.x.y + v.y.x + v.y.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(Float8_ v)
{
    return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x + v.w.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ float dot(T a, T b)
{
    return sum(mul<T, T, T>(a, b));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename A, typename T>
inline __device__ float dot(T a, T b)
{
    return sum(mul<A, T, T>(a, b));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void zero(uint16_t& dst)
{
    dst = uint16_t(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ void zero(T& dst)
{
    constexpr int WORDS = sizeof(T) / 4;
    union {
        T        raw;
        uint32_t words[WORDS];
    } tmp;
#pragma unroll
    for (int ii = 0; ii < WORDS; ++ii) {
        tmp.words[ii] = 0u;
    }
    dst = tmp.raw;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float logn_attn_get_scaling(float seq_len, int max_position_embeddings)
{
    if (seq_len <= max_position_embeddings) {
        return 1.f;
    }
    return log2f(seq_len) / log2f(max_position_embeddings);
}

inline __device__ float
rotary_embedding_get_base(float seq_len, int max_position_embeddings, float rot_embed_dim, float base)
{
    if (seq_len < max_position_embeddings) {
        return base;
    }
    float ntk_alpha = max(exp2f(ceilf(log2f(seq_len / max_position_embeddings) + 1.f)) - 1.f, 1.f);
    base *= powf(ntk_alpha, rot_embed_dim / (rot_embed_dim - 2.f));
    return base;
}

// inline __device__ float
// rotary_embedding_get_base(float seq_len, int max_position_embeddings, float rot_embed_dim, float base)
// {
//     constexpr float scaling_factor = 1.f;
//     if (scaling_factor * seq_len < max_position_embeddings) {
//         return base;
//     }
//     base *= powf((scaling_factor * seq_len / max_position_embeddings) - (scaling_factor - 1.f),
//                  rot_embed_dim / (rot_embed_dim - 2.f));
//     return base;
// }

inline __device__ float2 rotary_embedding_coefficient(int zid, int rot_embed_dim, float base, float t_step)
{
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 rotary_embedding_transform(const float2 v, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * v.x - coef.y * v.y;
    rot_v.y = coef.x * v.y + coef.y * v.x;
    return rot_v;
}

inline __device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef)
{
    float2 fv     = half2_to_float2(v);
    float2 rot_fv = rotary_embedding_transform(fv, coef);
    return float2_to_half2(rot_fv);
}

#ifdef ENABLE_BF16
inline __device__ __nv_bfloat162 rotary_embedding_transform(const __nv_bfloat162 v, const float2 coef)
{
    float2 fv     = bf1622float2(v);
    float2 rot_fv = rotary_embedding_transform(fv, coef);
    return __floats2bfloat162_rn(rot_fv.x, rot_fv.y);
}
#endif

inline __device__ void apply_rotary_embedding(float& q, int zid, int rot_embed_dim, int t_step)
{
    return;
}

inline __device__ void apply_rotary_embedding(float& q, float& k, int zid, int rot_embed_dim, int t_step)
{
    return;
}

inline __device__ void apply_rotary_embedding(float2& q, int tid, int rot_embed_dim, float base, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, base, t_step);
    q               = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(float2& q, float2& k, int tid, int rot_embed_dim, float base, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, base, t_step);
    q               = rotary_embedding_transform(q, coef);
    k               = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(float4& q, int tid, int rot_embed_dim, float base, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }

    Float4_&   q_    = *reinterpret_cast<Float4_*>(&q);
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, base, t_step);
    q_.x             = rotary_embedding_transform(q_.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, base, t_step);
    q_.y             = rotary_embedding_transform(q_.y, coef1);
}

inline __device__ void apply_rotary_embedding(float4& q, float4& k, int tid, int rot_embed_dim, float base, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }

    Float4_&   q_    = *reinterpret_cast<Float4_*>(&q);
    Float4_&   k_    = *reinterpret_cast<Float4_*>(&k);
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, base, t_step);
    q_.x             = rotary_embedding_transform(q_.x, coef0);
    k_.x             = rotary_embedding_transform(k_.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, base, t_step);
    q_.y             = rotary_embedding_transform(q_.y, coef1);
    k_.y             = rotary_embedding_transform(k_.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint32_t& q, int tid, int rot_embed_dim, float base, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, base, t_step);
    q               = rotary_embedding_transform(q, coef);
}

inline __device__ void
apply_rotary_embedding(uint32_t& q, uint32_t& k, int tid, int rot_embed_dim, float base, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, base, t_step);
    q               = rotary_embedding_transform(q, coef);
    k               = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(uint2& q, int tid, int rot_embed_dim, float base, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, base, t_step);
    q.x              = rotary_embedding_transform(q.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, base, t_step);
    q.y              = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint2& q, uint2& k, int tid, int rot_embed_dim, float base, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, base, t_step);
    q.x              = rotary_embedding_transform(q.x, coef0);
    k.x              = rotary_embedding_transform(k.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, base, t_step);
    q.y              = rotary_embedding_transform(q.y, coef1);
    k.y              = rotary_embedding_transform(k.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint4& q, int tid, int rot_embed_dim, float base, int t_step)
{
    if (8 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, base, t_step);
    q.x              = rotary_embedding_transform(q.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, base, t_step);
    q.y              = rotary_embedding_transform(q.y, coef1);
    const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, base, t_step);
    q.z              = rotary_embedding_transform(q.z, coef2);
    const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, base, t_step);
    q.w              = rotary_embedding_transform(q.w, coef3);
}

inline __device__ void apply_rotary_embedding(uint4& q, uint4& k, int tid, int rot_embed_dim, float base, int t_step)
{
    if (8 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, base, t_step);
    q.x              = rotary_embedding_transform(q.x, coef0);
    k.x              = rotary_embedding_transform(k.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, base, t_step);
    q.y              = rotary_embedding_transform(q.y, coef1);
    k.y              = rotary_embedding_transform(k.y, coef1);
    const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, base, t_step);
    q.z              = rotary_embedding_transform(q.z, coef2);
    k.z              = rotary_embedding_transform(k.z, coef2);
    const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, base, t_step);
    q.w              = rotary_embedding_transform(q.w, coef3);
    k.w              = rotary_embedding_transform(k.w, coef3);
}

}  // namespace mmha
