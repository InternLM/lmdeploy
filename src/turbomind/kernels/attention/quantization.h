#pragma once

#include "src/turbomind/kernels/attention/array_ops.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace turbomind {

template<class T>
__device__ T Infinity()
{
    if constexpr (std::is_same_v<T, half>) {
        return __ushort_as_half((unsigned short)0x7C00U);
    }

    if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __ushort_as_bfloat16((unsigned short)0x7F80U);
    }

    if constexpr (std::is_same_v<T, float>) {
        return __int_as_float(0x7f800000U);
    }

    return T{};
}

template<class T>
__device__ constexpr T Max(T a, T b)
{
    if constexpr (sizeof(T) == 2) {
        return __hmax(a, b);
    }

    if constexpr (std::is_same_v<T, float>) {
        return fmaxf(a, b);
    }

    if constexpr (std::is_same_v<T, int>) {
        return max(a, b);
    }

    return T{};
}

template<class T>
__device__ constexpr T Min(T a, T b)
{
    if constexpr (sizeof(T) == 2) {
        return __hmin(a, b);
    }

    if constexpr (std::is_same_v<T, float>) {
        return fminf(a, b);
    }

    if constexpr (std::is_same_v<T, int>) {
        return min(a, b);
    }

    return T{};
}

template<bool norm = true>
inline __device__ Array<half, 4> cvt_f16x4_u8(const Array<uint8_t, 4>& src)
{
    static constexpr uint32_t f16_magic = 0x64000000;
    // 01234567 01234567
    // SEEEEEMM MMMMMMMM
    //      1MM XXXXXXXX
    // (1 + x/2^10) * 2^(e-15) -> e-15=10 -> e=25=16+8+1 -> 01100100b -> 0x64
    Array<uint32_t, 2> dst;
    dst[0] = __byte_perm((uint32_t&)src, f16_magic, 0x7170);
    dst[1] = __byte_perm((uint32_t&)src, f16_magic, 0x7372);
    if constexpr (norm) {
        for (int i = 0; i < 4; ++i) {
            ((Array<half, 4>&)dst)[i] -= __ushort_as_half(0x6400U);
        }
    }
    return (Array<half, 4>&)dst;
}

inline __device__ Array<nv_bfloat16, 4> cvt_bf16x4_u8(const Array<uint8_t, 4>& src)
{
    // 01234567 01234567 01234567 01234567
    // SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM
    //      1MM          XXXXXXXX
    // (1 + x/2^15) * 2^(e-127) -> e-127=15 -> e=142 -> 01000111 -> 0x47
    static constexpr uint32_t f32_magic = 0x47000000;  // 32768

    Array<uint32_t, 4> tmp;
    tmp[0] = __byte_perm((uint32_t&)src, f32_magic, 0x7604);
    tmp[1] = __byte_perm((uint32_t&)src, f32_magic, 0x7614);
    tmp[2] = __byte_perm((uint32_t&)src, f32_magic, 0x7624);
    tmp[3] = __byte_perm((uint32_t&)src, f32_magic, 0x7634);

    auto& vec = (Array<float, 4>&)tmp;

    Array<nv_bfloat16, 4> dst;
    PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
        dst[i] = __float2bfloat16(vec[i] - 32768.f);
    }
    return dst;
}

template<class T>
inline __device__ T round(float x)
{
    uint32_t y{};
    if constexpr (std::is_same_v<T, uint8_t>) {
        asm("cvt.rni.sat.u8.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    }
    if constexpr (std::is_same_v<T, uint16_t>) {
        asm("cvt.rni.sat.u16.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    }
    if constexpr (std::is_same_v<T, uint32_t>) {
        asm("cvt.rni.sat.u32.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    }
    return y;
}

template<class T>
inline __device__ T round(half x)
{
    uint32_t y{};
    if constexpr (std::is_same_v<T, uint8_t>) {
        asm("cvt.rni.sat.u8.f16 %0, %1;\n" : "=r"(y) : "h"((uint16_t&)x));
    }
    if constexpr (std::is_same_v<T, uint16_t>) {
        asm("cvt.rni.sat.u16.f16 %0, %1;\n" : "=r"(y) : "h"((uint16_t&)x));
    }
    if constexpr (std::is_same_v<T, uint32_t>) {
        asm("cvt.rni.sat.u32.f16 %0, %1;\n" : "=r"(y) : "h"((uint16_t&)x));
    }
    return y;
}

template<class T, class B>
inline __device__ T quant(float x, B n_bits)
{
    auto y = round<T>(x);
    if constexpr (n_bits < sizeof(T) * 8) {
        return min(y, T((1 << n_bits) - 1));
    }
    else {
        return y;
    }
}

template<int WarpThreadC, class T, int C>
__device__ inline void warp_minmax(Array<T, 2>& stats, const Array<T, C>& x)
{
    PRAGMA_UNROLL
    for (int i = 0; i < C; ++i) {
        stats[0] = Min(stats[0], x[i]);
        stats[1] = Max(stats[1], x[i]);
    }
    if constexpr (sizeof(T) == 2) {
        PRAGMA_UNROLL
        for (int mask = WarpThreadC / 2; mask > 0; mask /= 2) {
            Array<T, 2> tmp;
            (uint32_t&)tmp = __shfl_xor_sync(uint32_t(-1), (uint32_t&)stats, mask);
            stats[0]       = Min(stats[0], tmp[0]);
            stats[1]       = Max(stats[1], tmp[1]);
        }
    }
    else {
        PRAGMA_UNROLL
        for (int mask = WarpThreadC / 2; mask > 0; mask /= 2) {
            stats[0] = Min(stats[0], __shfl_xor_sync(uint32_t(-1), stats[0], mask));
            stats[1] = Max(stats[1], __shfl_xor_sync(uint32_t(-1), stats[1], mask));
        }
    }
}

template<int WarpThreadC, class P, class T, class B, int N, int C, int S>
__device__ void warp_stats(Array<P, 2> (&param)[S], const Array<T, N> (&x)[S][C], B n_bits)
{
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        Array<T, 2> stats{Infinity<T>(), -Infinity<T>()};
        PRAGMA_UNROLL
        for (int c = 0; c < C; ++c) {
            warp_minmax<WarpThreadC>(stats, x[s][c]);
        }
        const float inv_q_max = fdividef(1.f, float((1 << n_bits) - 1));
        const float scale     = ((float)stats[1] - (float)stats[0]) * inv_q_max;
        param[s][0]           = (P)scale;
        param[s][1]           = (P)stats[0];
    }
}

template<class Q, class T, class P, class B, int N, int C, int S>
__device__ void
quantize(Array<Q, N> (&dst)[S][C], const Array<T, N> (&src)[S][C], const Array<P, 2> (&params)[S], B n_bits)
{
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        P inv_scale = (P)fdividef(1.f, (float)params[s][0]);
        P zero      = params[s][1];
        PRAGMA_UNROLL
        for (int c = 0; c < C; ++c) {
            PRAGMA_UNROLL
            for (int i = 0; i < N; ++i) {
                const auto v = ((P)src[s][c][i] - zero) * inv_scale;
                dst[s][c][i] = quant<Q>(v, n_bits);
            }
        }
    }
}

template<class P, int S>
__device__ void fuse_magic(Array<P, 2> (&params)[S])
{
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        params[s][1] = params[s][1] - half(1024) * params[s][0];
    }
}

template<class T, class Q, class P, class B, int N, int C, int S>
inline __device__ void
dequantize(Array<T, N> (&dst)[S][C], const Array<Q, N> (&src)[S][C], const Array<P, 2> (&params)[S], B n_bits)
{
    static_assert(N % 4 == 0);
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        auto scale = params[s][0];
        auto zero  = params[s][1];  // - half(1024.f) * params[s][0];
        PRAGMA_UNROLL
        for (int c = 0; c < C; ++c) {
            if constexpr (1) {
                PRAGMA_UNROLL
                for (int i = 0; i < N; i += 4) {
                    using namespace ops;
                    (Array<T, 4>&)dst[s][c][i] =
                        cast<T>(cast<P>(cvt_f16x4_u8<true>((Array<Q, 4>&)src[s][c][i])) * scale + zero);
                }
            }
            else {
                using signed_t = std::make_signed_t<Q>;
                for (int i = 0; i < N; ++i) {
                    dst[s][c][i] = T(P((signed_t)src[s][c][i] - (signed_t)quant<Q>(-zero / scale, n_bits)) * scale);
                }
            }
        }
    }
}

template<int D, int S>
inline __device__ void
dequantize_K(Array<half, D> (&dst)[S], const Array<uint8_t, D> (&src)[S], const Array<half, 2> (&param)[S])
{
    static_assert(D % 4 == 0);
    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        PRAGMA_UNROLL
        for (int d = 0; d < D; d += 4) {
            (Array<half, 4>&)dst[s][d] = cvt_f16x4_u8<false>((Array<uint8_t, 4>&)src[s][d]);
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        using namespace ops;
        dst[s] = dst[s] * param[s][0] + (param[s][1] - half(1024.f) * param[s][0]);
    }
}

template<int S, int D>
__device__ void
dequantize_V(Array<half, S> (&dst)[D], const Array<uint8_t, S> (&src)[D], const Array<half, 2> (&param)[S])
{
    static_assert(S % 4 == 0);
    PRAGMA_UNROLL
    for (int d = 0; d < D; ++d) {
        PRAGMA_UNROLL
        for (int s = 0; s < S; s += 4) {
            (Array<half, 4>&)dst[d][s] = cvt_f16x4_u8((Array<uint8_t, 4>&)src[d][s]);
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
        PRAGMA_UNROLL
        for (int d = 0; d < D; ++d) {
            using namespace ops;
            dst[d][s] = dst[d][s] * param[s][0] + param[s][1];
        }
    }
}

//        0         1
//    0123 4567 89ab cdef
// -> 0189 23ab 45cd 67ef

template<int N>
__device__ void permute_K(Array<uint8_t, N>& x)
{
    static_assert(N % 8 == 0);
    PRAGMA_UNROLL
    for (int i = 0; i < N; i += 8) {
        auto u = (Array<uint32_t, 2>&)x[i];

        Array<uint32_t, 2> v;

        v[0] = __shfl_xor_sync(uint32_t(-1), u[0], 1);
        v[1] = __shfl_xor_sync(uint32_t(-1), u[1], 1);

        Array<uint32_t, 2> w;

        if (threadIdx.x % 2 == 0) {
            w[0] = __byte_perm(u[0], v[0], 0x5410);
            w[1] = __byte_perm(u[0], v[0], 0x7632);
        }
        else {
            w[0] = __byte_perm(v[1], u[1], 0x5410);
            w[1] = __byte_perm(v[1], u[1], 0x7632);
        }

        (Array<uint32_t, 2>&)x[i] = w;
    }
}

// From:
//  (0,0)(0,1)(0,2)(0,3)(0,4)(0,5)(0,6)(0,7)(0,8)(0,9)(0,a)(0,b)(0,c)(0,d)(0,e)(0,f)
//  (1,0)(1,1)(1,2)(1,3)(1,4)(1,5)(1,6)(1,7)(1,8)(1,9)(1,a)(1,b)(1,c)(1,d)(1,e)(1,f)
// To:
//  (0,0)(1,0)(0,1)(1,1)(0,2)(1,2)(0,3)(1,3)(0,4)(1,4)(0,5)(1,5)(0,6)(1,6)(0,7)(1,7)
//  (8,0)(9,0)(8,1)(9,1)(8,2)(9,2)(8,3)(9,3)(8,4)(9,4)(8,5)(9,5)(8,6)(9,6)(8,7)(9,7)
//  (2,0)(3,0)(2,1)(3,1)(2,2)(3,2)(2,3)(3,3)(2,4)(3,4)(2,5)(3,5)(2,6)(3,6)(2,7)(3,7)
//  (a,0)(b,0)(a,1)(b,1)(a,2)(b,2)(a,3)(b,3)(a,4)(b,4)(a,5)(b,5)(a,6)(b,6)(a,7)(b,7)
//   4    5
//   c    d
//   6    7
//   e    f

// for (int i = 0; i < 16; ++i) {
//     for (int j = 0; j < 16; ++j) {
//         int ii = i % 2 * 8 + i % 8 / 2 * 2 + j % 2;
//         int jj = i / 8 * 8 + j / 2;
//         printf("%x%x ", ii, jj);
//     }
//     printf("\n");
// }

template<class Map>
__device__ void permute_V(Array<uint8_t, Map::kAccessC> (&x)[Map::kIterS][Map::kIterC])
{
    // __shared__ __align__(16) uint8_t tmp[Map::kDimS][Map::kDimC];

    __shared__ __align__(16) uint8_t tmp[Map::kDimS / 16][Map::kDimC / 16][16][16];

    const int  warp_id = threadIdx.x / WARP_SIZE;
    const int  lane_id = threadIdx.x % WARP_SIZE;
    const int2 offset  = Map::get_offset(warp_id, lane_id);

    constexpr int N = Map::kAccessC;
    static_assert(N == 8);

    PRAGMA_UNROLL
    for (int s = 0; s < Map::kIterS; ++s) {
        PRAGMA_UNROLL
        for (int c = 0; c < Map::kIterC; ++c) {
            const int si = offset.y + s * Map::kDeltaS;
            const int ci = offset.x + c * Map::kDeltaC;
            //
            (Array<uint8_t, N>&)tmp[si / 16][ci / 16][si % 16][ci % 16] = x[s][c];
        }
    }

    __syncthreads();

    PRAGMA_UNROLL
    for (int s = 0; s < Map::kIterS; ++s) {
        const int si = offset.y + s * Map::kDeltaS;
        PRAGMA_UNROLL
        for (int c = 0; c < Map::kIterC; ++c) {
            PRAGMA_UNROLL
            for (int i = 0; i < N; ++i) {
                const int ci = offset.x + c * Map::kDeltaC + i;
                const int ss = si % 16;
                const int cc = ci % 16;
                const int sj = ss % 2 * 8 + ss % 8 / 2 * 2 + cc % 2;
                const int cj = ss / 8 * 8 + cc / 2;
                x[s][c][i]   = tmp[si / 16][ci / 16][sj][cj];
            }
        }
    }
}

}  // namespace turbomind
