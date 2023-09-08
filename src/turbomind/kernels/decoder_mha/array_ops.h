#pragma once

#include "../gemm_s_f16/common.h"
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
        return a + b;
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

template<int N>
struct RotaryEmbedding {

    static_assert(N % 2 == 0);

    Array<float, N> inv_freqs_;

    __device__ RotaryEmbedding(float base, int dims, int timestep, int2 offset)
    {
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2) {
            const float2 tmp  = rotary_embedding_coefficient(offset.x + i, dims, base, timestep);
            inv_freqs_[i]     = tmp.x;
            inv_freqs_[i + 1] = tmp.y;
        }
    }

    inline __device__ float2 rotary_embedding_coefficient(int idx, int dims, float base, int timestep)
    {
        const float inv_freq = timestep / powf(base, idx / (float)dims);
        return {cos(inv_freq), sin(inv_freq)};
    }

    template<typename T>
    __device__ void apply(Array<T, N>& x)
    {
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2) {
            float tmp0 = inv_freqs_[i] * (float)x[i] - inv_freqs_[i + 1] * (float)x[i + 1];
            float tmp1 = inv_freqs_[i] * (float)x[i + 1] + inv_freqs_[i + 1] * (float)x[i];
            x[i]       = (T)tmp0;
            x[i + 1]   = (T)tmp1;
        }
    }
};

template<typename VecQk, typename ThreadMap>
struct LogNScaling {
    __device__ void apply(VecQk& x)
    {
        PRAGMA_UNROLL
        for (int i = 0; i < VecQk::kSize; ++i) {
            // TODO:
        }
    }
};

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

template<typename Accum, typename Compute, int kThreadGroupSize, typename T, int N, int V>
inline __device__ Accum qk_dot(const Array<T, N> (&q)[V], const Array<T, N> (&k)[V])
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

template<typename Accum, typename Compute, int kThreadGroupSize, typename T, int N>
inline __device__ Accum qk_dot(const Array<T, N>& q, const Array<T, N>& k)
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

}  // namespace turbomind