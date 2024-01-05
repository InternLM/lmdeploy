// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm_s_f16/common.h"
#include "src/turbomind/utils/cuda_bf16_wrapper.h"
#include <cfloat>
#include <limits>

namespace turbomind {

namespace ops {

template<typename T>
struct plus {
    __device__ T operator()(T a, T b)
    {
        return a + b;
    }
};

template<typename T>
struct minus {
    __device__ T operator()(T a, T b)
    {
        return a - b;
    }
};

template<typename T>
struct multiplies {
    __device__ T operator()(T a, T b)
    {
        return a * b;
    }
};

template<typename T, int N, typename Op>
inline __device__ Array<T, N> binary_op_vv(const Array<T, N>& a, const Array<T, N>& b, Op op)
{
    Array<T, N> c;
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        c[i] = op(a[i], b[i]);
    }
    return c;
}

template<typename T, int N, typename Op>
inline __device__ Array<T, N> binary_op_sv(const T& a, const Array<T, N>& b, Op op)
{
    Array<T, N> c;
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        c[i] = op(a, b[i]);
    }
    return c;
}

template<typename T, int N, typename Op>
inline __device__ Array<T, N> binary_op_vs(const Array<T, N>& a, const T& b, Op op)
{
    Array<T, N> c;
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        c[i] = op(a[i], b);
    }
    return c;
}

template<typename T, int N>
inline __device__ Array<T, N> operator+(const Array<T, N>& a, const Array<T, N>& b)
{
    return binary_op_vv(a, b, plus<T>{});
}

template<typename T, int N>
inline __device__ Array<T, N> operator*(const Array<T, N>& a, const Array<T, N>& b)
{
    return binary_op_vv(a, b, multiplies<T>{});
}

template<typename T, int N>
inline __device__ Array<T, N> operator*(const Array<T, N>& a, const T& b)
{
    return binary_op_vs(a, b, multiplies<T>{});
}

}  // namespace ops

template<typename To, typename From, int N>
inline __device__ Array<To, N> cast(const Array<From, N>& src)
{
    Array<To, N> dst;
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        dst[i] = (To)src[i];
    }
    return dst;
}

template<int N>
struct RotaryEmbedding {

    static_assert(N % 2 == 0);

    Array<float, N> cs_;
    float           scale_;

    __device__ RotaryEmbedding(float base, int dims, int timestep, int2 offset, float scaling_factor)
    {
        scale_ = scaling_factor == 0.f ? 1.f : scaling_factor;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2) {
            const float2 tmp = get_coefficient(offset.x + i, dims, base, timestep);
            cs_[i]           = tmp.x;
            cs_[i + 1]       = tmp.y;
        }
    }

    __device__ inline float2 get_coefficient(int idx, int dims, float base, int timestep)
    {
        const float inv_freq = timestep / (powf(base, idx / (float)dims) * scale_);
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
            x[i]       = (T)tmp0;
            x[i + 1]   = (T)tmp1;
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

template<typename T, int N>
inline __device__ void Store(T* dst, const Array<T, N>& src)
{
    static_assert(sizeof(Array<T, N>) <= sizeof(uint4));

    if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
        *(uint4*)dst = (const uint4&)src;
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
        *(uint2*)dst = (const uint2&)src;
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint1)) {
        *(uint1*)dst = (const uint1&)src;
    }
    else {
        static_assert(!std::is_same_v<T, T>);
    }
}

template<typename T, int N>
inline __device__ void Ldg(Array<T, N>& dst, const T* src)
{
    static_assert(sizeof(Array<T, N>) <= sizeof(uint4));

    if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
        (uint4&)dst = __ldg((const uint4*)src);
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
        (uint2&)dst = __ldg((const uint2*)src);
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint)) {
        (uint&)dst = __ldg((const uint*)src);
    }
    else {
        static_assert(!std::is_same_v<T, T>);
    }
}

template<typename T, int N>
inline __device__ void Lds(Array<T, N>& dst, const T* src)
{
    static_assert(sizeof(Array<T, N>) <= sizeof(uint4));

    if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
        (uint4&)dst = *(const uint4*)src;
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
        (uint2&)dst = *(const uint2*)src;
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint)) {
        (uint1&)dst = *(const uint1*)src;
    }
    else {
        static_assert(!std::is_same_v<T, T>);
    }
}

template<typename Accum, typename Compute, int kThreadGroupSize, typename Tq, typename Tk, int N, int V>
inline __device__ Accum qk_dot(const Array<Tq, N> (&q)[V], const Array<Tk, N> (&k)[V])
{
    Accum accum{};

    PRAGMA_UNROLL
    for (int vi = 0; vi < V; ++vi) {
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            accum += Accum(Compute(q[vi][i]) * Compute(k[vi][i]));
        }
    }

    PRAGMA_UNROLL
    for (int mask = kThreadGroupSize / 2; mask >= 1; mask /= 2) {
        accum += __shfl_xor_sync((uint32_t)-1, accum, mask);
    }

    return accum;
}

template<typename Accum, typename Compute, int kThreadGroupSize, typename Tq, typename Tk, int N>
inline __device__ Accum qk_dot(const Array<Tq, N>& q, const Array<Tk, N>& k)
{
    Accum accum{};

    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        accum += Accum(Compute(q[i]) * Compute(k[i]));
    }

    PRAGMA_UNROLL
    for (int mask = kThreadGroupSize / 2; mask >= 1; mask /= 2) {
        accum += __shfl_xor_sync((uint32_t)-1, accum, mask);
    }

    return accum;
}

template<typename ComputeType, typename Tp, typename Tv, typename To, int N, int M>
inline __device__ void fma_pv(Tp pr, const Array<Tv, N> (&v)[M], Array<To, N> (&o)[M])
{
    PRAGMA_UNROLL
    for (int m = 0; m < M; ++m) {
        PRAGMA_UNROLL
        for (int n = 0; n < N; ++n) {
            o[m][n] += To(ComputeType(v[m][n]) * ComputeType(pr));
        }
    }
}

template<typename ThreadMap, typename T, int N>
inline __device__ Array<T, N> qk_max(Array<T, N> val, T* smem_red, int warp_id, int lane_id)
{
    constexpr int kWarpCount = ThreadMap::kWarpCount;

    // warp maximum
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        PRAGMA_UNROLL
        for (int mask = WARP_SIZE / 2; mask >= ThreadMap::kWarpThreadC; mask /= 2) {
            val[i] = fmaxf(val[i], __shfl_xor_sync((uint32_t)-1, val[i], mask));
        }
        if (lane_id == 0) {
            smem_red[i * kWarpCount + warp_id] = val[i];
        }
    }

    __syncthreads();

    // block maximum
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        val[i] = lane_id < kWarpCount ? smem_red[i * kWarpCount + lane_id] : -FLT_MAX;
        PRAGMA_UNROLL
        for (int mask = kWarpCount >> 1; mask >= 1; mask >>= 1) {
            val[i] = fmaxf(val[i], __shfl_xor_sync((uint32_t)-1, val[i], mask));
        }
        // braodcast to all threads
        val[i] = __shfl_sync((uint32_t)-1, val[i], 0);
    }

    return val;
}

template<int kWarpCount, typename T, int N>
inline __device__ Array<T, N> blockSum(Array<T, N> val, T* smem_red, int warp_id, int lane_id)
{
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        PRAGMA_UNROLL
        for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
            val[i] += __shfl_xor_sync((uint32_t)-1, val[i], mask);
        }
        if (lane_id == 0) {
            smem_red[i * kWarpCount + warp_id] = val[i];
        }
    }

    __syncthreads();

    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        val[i] = lane_id < kWarpCount ? smem_red[i * kWarpCount + lane_id] : T{};
        PRAGMA_UNROLL
        for (int mask = kWarpCount >> 1; mask >= 1; mask >>= 1) {
            val[i] += __shfl_xor_sync((uint32_t)-1, val[i], mask);
        }
        val[i] = __shfl_sync((uint32_t)-1, val[i], 0);
    }

    return val;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// generic case for floating point -> floating point / integer -> integer conversion
template<typename Ti, typename To, typename = void>
struct ConvertKvCache {
    __device__ __host__ ConvertKvCache(float, float) {}
    template<int N>
    inline __device__ auto operator()(const Array<Ti, N>& vi) const -> Array<To, N>
    {
        Array<To, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            vo[i] = (To)vi[i];
        }
        return vo;
    }
};

// generic case for converting to same type, bypass
template<typename T>
struct ConvertKvCache<T, T> {
    __device__ __host__ ConvertKvCache(float, float) {}
    template<int N>
    inline __device__ auto operator()(const Array<T, N>& v) const -> Array<T, N>
    {
        return v;
    }
};

template<typename Ti>
struct ConvertKvCache<Ti, int8_t> {

    float scale_;
    float zero_;

    __device__ __host__ ConvertKvCache(float scale, float zero): scale_(scale), zero_(zero) {}

    inline __device__ uint8_t round(float x) const
    {
        uint32_t y;
        asm("cvt.rni.sat.u8.f32 %0, %1;\n" : "=r"(y) : "f"(x));
        return y;
    }

    template<int N>
    inline __device__ auto operator()(const Array<Ti, N>& vi) const -> Array<int8_t, N>
    {
        Array<int8_t, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            // convert to unsigned int by offsetting +128
            (uint8_t&)vo[i] = round(((float)vi[i] - zero_) / scale_ + 128.f);
        }
        return vo;
    }
};

inline __device__ Array<float, 4> fast_i2f_f32_s8(const Array<int8_t, 4>& x)
{
    union {
        Array<float, 4>    f32x4;
        Array<uint32_t, 4> u32x4;
    };

    auto& i8s = (const uint32_t&)x;

    // 00000000111111112222222233333333
    // 01234567012345670123456701234567
    // SEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMM
    // 0????????_______XXXXXXXX________
    // (1 + x / 2^15) * 2^(e - 127) -> e - 127 == 15 -> e = 142
    //                                       7 6 5 4
    static constexpr uint32_t f32_magic = 0x47000000;  // 2^15 = 32768
    static constexpr uint32_t m0        = 0x7604;
    static constexpr uint32_t m1        = 0x7614;
    static constexpr uint32_t m2        = 0x7624;
    static constexpr uint32_t m3        = 0x7634;

    asm("prmt.b32 %0,%1,%2,%3;\n" : "=r"(u32x4[0]) : "r"(i8s), "n"(f32_magic), "n"(m0));
    asm("prmt.b32 %0,%1,%2,%3;\n" : "=r"(u32x4[1]) : "r"(i8s), "n"(f32_magic), "n"(m1));
    asm("prmt.b32 %0,%1,%2,%3;\n" : "=r"(u32x4[2]) : "r"(i8s), "n"(f32_magic), "n"(m2));
    asm("prmt.b32 %0,%1,%2,%3;\n" : "=r"(u32x4[3]) : "r"(i8s), "n"(f32_magic), "n"(m3));

    if (0) {  // fused with dequantization
        PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
            f32x4[i] -= 32896.f;  // 32768 + 128
        }
    }

    return f32x4;
}

template<>
struct ConvertKvCache<int8_t, float> {

    float scale_;
    float zero_;

    __device__ __host__ ConvertKvCache(float scale, float zero): scale_(scale), zero_(zero)
    {
        zero_ = zero_ - 32896.f * scale_;
    }

    template<int N>
    inline __device__ auto operator()(const Array<int8_t, N>& vi) const -> Array<float, N>
    {
        Array<float, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 4) {
            auto& vec = (Array<float, 4>&)vo[i];
            vec       = fast_i2f_f32_s8((const Array<int8_t, 4>&)vi[i]);
            PRAGMA_UNROLL
            for (int j = 0; j < 4; ++j) {
                vec[j] = vec[j] * scale_ + zero_;
                // vec[j] = vec[j] * scale_ + (zero_ - 32896.f * scale_);
            }
        }
        return vo;
    }
};

template<>
struct ConvertKvCache<int8_t, half> {

    float scale_;
    float zero_;

    __device__ __host__ ConvertKvCache(float scale, float zero): scale_(scale), zero_(zero)
    {
        zero_ = zero_ - 32896.f * scale_;
    }

    template<int N>
    inline __device__ auto operator()(const Array<int8_t, N>& vi) const -> Array<half, N>
    {
        Array<half, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 4) {
            auto& vec = (Array<half, 4>&)vo[i];
            auto  tmp = fast_i2f_f32_s8((const Array<int8_t, 4>&)vi[i]);
            PRAGMA_UNROLL
            for (int j = 0; j < 4; ++j) {
                vec[j] = half(tmp[j] * scale_ + zero_);
                // vec[j] = half(tmp[j] * scale_ + (zero_ - 32896.f * scale_));
            }
        }
        return vo;
    }
};

#ifdef ENABLE_BF16
template<>
struct ConvertKvCache<int8_t, __nv_bfloat16> {

    float scale_;
    float zero_;

    __device__ __host__ ConvertKvCache(float scale, float zero): scale_(scale), zero_(zero)
    {
        zero_ = zero_ - 32896.f * scale_;
    }

    template<int N>
    inline __device__ auto operator()(const Array<int8_t, N>& vi) const -> Array<__nv_bfloat16, N>
    {
        Array<__nv_bfloat16, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 4) {
            auto& vec = (Array<__nv_bfloat16, 4>&)vo[i];
            auto  tmp = fast_i2f_f32_s8((const Array<int8_t, 4>&)vi[i]);
            PRAGMA_UNROLL
            for (int j = 0; j < 4; ++j) {
                vec[j] = __nv_bfloat16(tmp[j] * scale_ + zero_);
                // vec[j] = half(tmp[j] * scale_ + (zero_ - 32896.f * scale_));
            }
        }
        return vo;
    }
};
#endif  // ENABLE_BF16
}  // namespace turbomind
