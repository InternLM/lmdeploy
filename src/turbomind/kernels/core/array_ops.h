// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/common.h"
#include <cassert>
#include <type_traits>

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

template<typename T, int N>
inline __device__ Array<T, N> operator+(const Array<T, N>& a, const T& b)
{
    return binary_op_vs(a, b, plus<T>{});
}

template<typename T, int N>
inline __device__ Array<T, N> operator-(const Array<T, N>& a, const T& b)
{
    return binary_op_vs(a, b, minus<T>{});
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

template<class T, int N>
inline __device__ void fill(Array<T, N>& x, T val)
{
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        x[i] = val;
    }
}

template<class T, int M, int N>
inline __device__ void fill(Array<T, N> (&x)[M], T val)
{
    PRAGMA_UNROLL
    for (int i = 0; i < M; ++i) {
        fill(x[i], val);
    }
}

template<class T, int N>
inline __device__ void clear(Array<T, N>& x)
{
    fill(x, T(0));
}

template<class T, int M, int N>
inline __device__ void clear(Array<T, N> (&x)[M])
{
    PRAGMA_UNROLL
    for (int i = 0; i < M; ++i) {
        clear(x[i]);
    }
}

template<class T, int M1, int M0, int N>
inline __device__ void clear(Array<T, N> (&x)[M1][M0])
{
    PRAGMA_UNROLL
    for (int m1 = 0; m1 < M1; ++m1) {
        PRAGMA_UNROLL
        for (int m0 = 0; m0 < M0; ++m0) {
            clear(x[m1][m0]);
        }
    }
}

template<class T, int N>
inline __device__ void copy(const Array<T, N>& src, Array<T, N>& dst)
{
    dst = src;
}

template<class T, int M, int N>
inline __device__ void copy(const Array<T, N> (&src)[M], Array<T, N> (&dst)[M])
{
    PRAGMA_UNROLL
    for (int m = 0; m < M; ++m) {
        dst[m] = src[m];
    }
}

template<typename T, int N>
inline __device__ void Store(T* __restrict__ dst, const Array<T, N>& src)
{
    if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
        *(uint4*)dst = (const uint4&)src;
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
        *(uint2*)dst = (const uint2&)src;
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint1)) {
        *(uint1*)dst = (const uint1&)src;
    }
    else if constexpr (sizeof(Array<T, N>) % sizeof(uint4) == 0) {  //  uncoalesced
        static_assert(bitsof<T> % 8 == 0, "raw pointer arithmetic of sub-byte types");
        constexpr int M = sizeof(Array<T, N>) / sizeof(uint4);
        PRAGMA_UNROLL
        for (int i = 0; i < M; ++i) {
            *((uint4*)dst + i) = *((uint4*)&src + i);
        }
    }
    else {
        static_assert(!std::is_same_v<T, T>);
    }
}

template<typename T, int N>
inline __device__ void Stcs(T* __restrict__ dst, const Array<T, N>& src)
{
    static_assert(sizeof(Array<T, N>) <= sizeof(uint4));

    if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
        __stcs((uint4*)dst, (const uint4&)src);
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
        __stcs((uint2*)dst, (const uint2&)src);
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint1)) {
        __stcs((uint*)dst, (const uint&)src);
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
inline __device__ void Load(Array<T, N>& dst, const T* src)
{
    if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
        (uint4&)dst = *(const uint4*)src;
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
        (uint2&)dst = *(const uint2*)src;
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint)) {
        (uint1&)dst = *(const uint1*)src;
    }
    else if constexpr (sizeof(Array<T, N>) % sizeof(uint4) == 0) {  //  uncoalesced
        static_assert(bitsof<T> % 8 == 0, "raw pointer arithmetic of sub-byte types");
        constexpr int M = sizeof(Array<T, N>) / sizeof(uint4);
        PRAGMA_UNROLL
        for (int i = 0; i < M; ++i) {
            *((uint4*)&dst + i) = *((uint4*)src + i);
        }
    }
    else {
        static_assert(!std::is_same_v<T, T>);
    }
}

template<typename T, int N>
inline __device__ void Lds(Array<T, N>& dst, const T* src)
{
    Load(dst, src);
}

template<typename T, int N>
inline __device__ void LdShared(Array<T, N>& dst, uint32_t uintptr)
{
    static_assert(sizeof(Array<T, N>) <= sizeof(uint4));
    if constexpr (sizeof(Array<T, N>) == sizeof(uint4)) {
        uint4& p = (uint4&)dst;
        // clang-format off
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n" : "=r"(p.x), "=r"(p.y), "=r"(p.z), "=r"(p.w) : "r"(uintptr));
        // clang-format on
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint2)) {
        uint2& p = (uint2&)dst;
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];\n" : "=r"(p.x), "=r"(p.y) : "r"(uintptr));
    }
    else if constexpr (sizeof(Array<T, N>) == sizeof(uint)) {
        uint& p = (uint&)dst;
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(p) : "r"(uintptr));
    }
    else {
        static_assert(!std::is_same_v<T, T>);
    }
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

template<class T, int N>
__device__ void CpAsync(T* dst, const Array<T, N>* __restrict__ src)
{
    const int     smem_int_ptr = cast_smem_ptr_to_uint(dst);
    constexpr int cp_size      = sizeof(Array<T, N>);
#if TURBOMIND_ARCH_SM80
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(smem_int_ptr), "l"(src), "n"(cp_size));
#else
    assert(TURBOMIND_ARCH_SM80);
#endif
}

__inline__ __device__ uint transpose_m8n8_b16_warp_shuffle(uint value)
{
    const int lane_id  = threadIdx.x % WARP_SIZE;
    int       src_lane = lane_id / 8 + lane_id % 4 * 8;
    uint      u0       = __shfl_sync(0xffffffff, value, src_lane);
    uint      u1       = __shfl_sync(0xffffffff, value, src_lane + 4);
    short2    r;

    if (lane_id % 8 < 4) {
        r.x = ((short2&)u0).x;
        r.y = ((short2&)u1).x;
    }
    else {
        r.x = ((short2&)u0).y;
        r.y = ((short2&)u1).y;
    }
    return (uint&)r;
}

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 8)
__inline__ __device__ uint transpose_m8n8_b16_movmatrix(uint a)
{
#if TURBOMIND_ARCH_SM75
    uint d;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a));
    return d;
#else
    assert(TURBOMIND_ARCH_SM75);
    return 0;
#endif
}
#endif

__inline__ __device__ uint32_t transpose_m8n8_b16(uint32_t a)
{
#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 8)
    return transpose_m8n8_b16_movmatrix(a);
#else
    return transpose_m8n8_b16_warp_shuffle(a);
#endif
}

__inline__ __device__ Array<uint32_t, 2> transpose_m8n8_b32(const Array<uint32_t, 2>& x)
{
    uint32_t lo = __byte_perm(x[0], x[1], 0x5410);
    uint32_t hi = __byte_perm(x[0], x[1], 0x7632);

    lo = transpose_m8n8_b16(lo);
    hi = transpose_m8n8_b16(hi);

    Array<uint32_t, 2> y;
    y[0] = __byte_perm(lo, hi, 0x5410);
    y[1] = __byte_perm(lo, hi, 0x7632);

    return y;
}

}  // namespace turbomind
