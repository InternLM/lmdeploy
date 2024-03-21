#pragma once

#include "src/turbomind/kernels/attention/array_ops.h"
#include "src/turbomind/kernels/attention/data_type.h"
#include "src/turbomind/kernels/attention/smem_layout.h"
#include "src/turbomind/kernels/gemm_s_f16/common.h"

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace turbomind {

template<class T>
__device__ T Infinity()
{
    if constexpr (std::is_same_v<T, half>) {
        return __ushort_as_half((unsigned short)0x7C00U);
    }

#if __CUDA_ARCH__ >= 800
    if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __ushort_as_bfloat16((unsigned short)0x7F80U);
    }
#endif

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

template<bool norm = true>
inline __device__ Array<half, 4> cvt_f16x2x2_u8_trans(const Array<uint8_t, 4>& src)
{
    static constexpr uint32_t f16_magic = 0x64000000;
    // 01234567 01234567
    // SEEEEEMM MMMMMMMM
    //      1MM XXXXXXXX
    // (1 + x/2^10) * 2^(e-15) -> e-15=10 -> e=25=16+8+1 -> 01100100b -> 0x64
    Array<uint32_t, 2> dst;
    dst[0] = __byte_perm((uint32_t&)src, f16_magic, 0x7270);
    dst[1] = __byte_perm((uint32_t&)src, f16_magic, 0x7371);
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

inline __device__ Array<float, 4> cvt_f32x4_u8(const Array<uint8_t, 4>& src)
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
    PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
        vec[i] -= 32768.f;
    }
    return vec;
}

template<class T>
inline __device__ T round(float x)
{
    uint32_t y{};
    if constexpr (std::is_same_v<T, uint8_t>) {
        asm("cvt.rni.sat.u8.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        asm("cvt.rni.sat.u16.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        asm("cvt.rni.sat.u32.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    }
    else {
        static_assert(!std::is_same_v<T, T>, "not implemented");
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
    else if constexpr (std::is_same_v<T, uint16_t>) {
        asm("cvt.rni.sat.u16.f16 %0, %1;\n" : "=r"(y) : "h"((uint16_t&)x));
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        asm("cvt.rni.sat.u32.f16 %0, %1;\n" : "=r"(y) : "h"((uint16_t&)x));
    }
    else {
        static_assert(!std::is_same_v<T, T>, "not implemented");
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

        param[s][1] = (P)(rintf((float)stats[0] / scale) * scale);
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

//  floating point -> u8
template<class T>
struct ConvertKvCache<T, uint8_t> {
    T          inv_scale_;
    T          zero_;
    __device__ ConvertKvCache(T scale, T zero): zero_{zero}
    {
        // NVCC complains if we put this in the member init list
        inv_scale_ = (T)fdividef(1.f, (float)scale);
    }

    template<int N>
    __device__ auto operator()(const Array<T, N>& vi) const
    {
        Array<uint8_t, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            vo[i] = quant<uint8_t>((vi[i] - zero_) * inv_scale_, std::integral_constant<int, 8>{});
        }
        return vo;
    }
};

template<class T>
struct ConvertKvCache<T, uint4_t> {
    T          inv_scale_;
    T          zero_;
    __device__ ConvertKvCache(T scale, T zero): zero_{zero}
    {
        // NVCC complains if we put this in the member init list
        inv_scale_ = (T)fdividef(1.f, (float)scale);
    }

    static __device__ Array<uint4_t, 8> pack(const Array<uint8_t, 8>& vi)
    {
        Array<uint32_t, 2> ui = (Array<uint32_t, 2>&)vi;

        ui[0] |= (ui[0] >> 12);
        ui[1] |= (ui[1] >> 12);

        //  7 6 5 4 3 2 1 0
        // _7_67564_3_23120
        uint32_t uo = __byte_perm(ui[0], ui[1], 0x5140);

        return (Array<uint4_t, 8>&)uo;
    }

    /// TODO: try cvt.pack.sat.u4
    template<int N>
    __device__ auto operator()(const Array<T, N>& vi) const
    {
        static_assert(N % 8 == 0);
        Array<uint8_t, N> tmp;
        PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            tmp[i] = quant<uint8_t>((vi[i] - zero_) * inv_scale_, std::integral_constant<int, 4>{});
        }
        Array<uint4_t, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 8) {
            (Array<uint4_t, 8>&)vo[i] = pack((Array<uint8_t, 8>&)tmp[i]);
        }
        return vo;
    }
};

template<>
struct ConvertKvCache<uint4_t, half> {
    // half scale_;
    // half zero_;

    // __device__ ConvertKvCache(half scale, half zero): scale_{scale}, zero_{zero} {}

    half scale_;
    // Array<half, 2> zero_;
    half zero_;

    __device__ ConvertKvCache(half scale, half zero)
    {
        scale_ = scale;
        // zero_[0] = zero - scale * __ushort_as_half(0x6400);
        // zero_[1] = zero - scale * __ushort_as_half(0x5400);

        // zero_ = zero - scale * __ushort_as_half(0x5400);
        zero_ = zero;
    }

    static __device__ Array<half, 8> cvt_f16x8_u4(const Array<uint4_t, 8>& vi)
    {
        Array<half, 8>            result;
        uint32_t*                 h           = reinterpret_cast<uint32_t*>(&result);
        uint32_t const&           i4s         = reinterpret_cast<uint32_t const&>(vi);
        static constexpr uint32_t immLut      = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t BOT_MASK    = 0x000f000f;
        static constexpr uint32_t TOP_MASK    = 0x00f000f0;
        static constexpr uint32_t MAGIC_NUM_0 = 0x64006400;  // `1024`
        static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;  // `64`
        const uint32_t            top_i4s     = i4s >> 8;
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[0]) : "r"(i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_0), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[1]) : "r"(i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[2]) : "r"(top_i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_0), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[3]) : "r"(top_i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        // asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(MAGIC_NUM_0));
        // asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(MAGIC_NUM_1));
        // asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(MAGIC_NUM_0));
        // asm("sub.f16x2 %0, %1, %2;\n" : "=r"(h[3]) : "r"(h[3]), "r"(MAGIC_NUM_1));
        return result;
    }

    static __device__ Array<half, 8> cvt_f16x8_u4_biased(const Array<uint4_t, 8>& vi)
    {
        Array<half, 8>            result;
        uint32_t*                 h           = reinterpret_cast<uint32_t*>(&result);
        uint32_t const&           i4s         = reinterpret_cast<uint32_t const&>(vi);
        static constexpr uint32_t immLut      = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t BOT_MASK    = 0x000f000f;
        static constexpr uint32_t TOP_MASK    = 0x00f000f0;
        static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;        // `64`
        static constexpr uint32_t MAGIC_NUM_2 = MAGIC_NUM_1 >> 4;  // `64` >> 4
        const uint32_t            top_i4s     = i4s >> 8;
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[0]) : "r"(i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[1]) : "r"(i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[2]) : "r"(top_i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[3]) : "r"(top_i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        h[0] <<= 4;
        h[2] <<= 4;
        return result;
    }

    template<int N>
    __device__ auto operator()(const Array<uint4_t, N>& vi) const
    {
        Array<half, N> vo;
        PRAGMA_UNROLL
        for (int i = 0; i < N; i += 8) {
            auto& v = (Array<half, 8>&)vo[i];
            v       = cvt_f16x8_u4_biased((Array<uint4_t, 8>&)vi[i]);
            // v[0]    = v[0] * scale_ + zero_[0];
            // v[1]    = v[1] * scale_ + zero_[0];
            // v[2]    = v[2] * scale_ + zero_[1];
            // v[3]    = v[3] * scale_ + zero_[1];
            // v[4]    = v[4] * scale_ + zero_[0];
            // v[5]    = v[5] * scale_ + zero_[0];
            // v[6]    = v[6] * scale_ + zero_[1];
            // v[7]    = v[7] * scale_ + zero_[1];
            {
                using namespace ops;
                v = v * scale_ + zero_;
            }
        }
        return vo;
    }
};

template<>
struct ConvertKvCache<uint4_t, float> {

#if 1
    ConvertKvCache<uint4_t, half> impl_;

    __device__ ConvertKvCache(float scale, float zero): impl_{scale, zero} {}

    template<int N>
    __device__ auto operator()(const Array<uint4_t, N>& vi) const
    {
        return cast<float>(impl_(vi));
    }
#else
    static __device__ Array<half, 8> cvt_f16x8_u4_biased(const Array<uint4_t, 8>& vi)
    {
        Array<half, 8>            result;
        uint32_t*                 h           = reinterpret_cast<uint32_t*>(&result);
        uint32_t const&           i4s         = reinterpret_cast<uint32_t const&>(vi);
        static constexpr uint32_t immLut      = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t BOT_MASK    = 0x000f000f;
        static constexpr uint32_t TOP_MASK    = 0x00f000f0;
        static constexpr uint32_t MAGIC_NUM_1 = 0x54005400;        // `64`
        static constexpr uint32_t MAGIC_NUM_2 = MAGIC_NUM_1 >> 4;  // `64` >> 4
        const uint32_t            top_i4s     = i4s >> 8;
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[0]) : "r"(i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[1]) : "r"(i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[2]) : "r"(top_i4s), "n"(BOT_MASK), "n"(MAGIC_NUM_2), "n"(immLut));
        asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[3]) : "r"(top_i4s), "n"(TOP_MASK), "n"(MAGIC_NUM_1), "n"(immLut));
        h[0] <<= 4;
        h[2] <<= 4;
        return result;
    }
    float      scale_;
    float      zero_;
    __device__ ConvertKvCache(float scale, float zero)
    {
        scale_ = scale;
        zero_  = zero - scale * 64.f;
    }
    template<int N>
    __device__ auto operator()(const Array<uint4_t, N>& vi) const
    {
        auto vo = cast<float>(cvt_f16x8_u4_biased(vi));
        using namespace ops;
        return vo * scale_ + zero_;
    }
#endif
};

// u8 -> f32/f16/bf16
template<class T>
struct ConvertKvCache<uint8_t, T> {
    T          scale_;
    T          zero_;
    __device__ ConvertKvCache(T scale, T zero): scale_{scale}, zero_{zero} {}

    template<int N>
    __device__ auto operator()(const Array<uint8_t, N>& vi) const
    {
        Array<T, N> vo;
        PRAGMA_UNROLL
        for (int n = 0; n < N; n += 4) {
            auto& ui = (const Array<uint8_t, 4>&)vi[n];
            auto& uo = (Array<T, 4>&)vo[n];

            if constexpr (std::is_same_v<T, half>) {
                uo = cvt_f16x4_u8<true>(ui);
            }
            else if constexpr (std::is_same_v<T, float>) {
                uo = cvt_f32x4_u8(ui);
            }
            else if constexpr (std::is_same_v<T, nv_bfloat16>) {
            }

            PRAGMA_UNROLL
            for (int c = 0; c < 4; ++c) {
                uo[c] = uo[c] * scale_ + zero_;
            }
        }
        return vo;
    }
};

template<class Q, class T>
inline __device__ void StoreQuantParam(T* dst, Array<T, 2> src)
{
    Store(dst, src);
}

template<>
inline __device__ void StoreQuantParam<uint4_t, half>(half* dst, Array<half, 2> src)
{
    src[1] = src[1] - src[0] * __ushort_as_half(0x5400);
    Store(dst, src);
}

#if 0
template<int K_K, int K_M, class Map, class T, class Tk, int ITER_S, int ITER_C>
__device__ void QuantizeK(const Array<T, 8> (&data)[ITER_S][ITER_C],
                          const Array<T, 2> (&param)[ITER_S],
                          Array<T, 8>       (&frag_K)[K_K][K_M],
                          Array<Tk, 8>      (&qdata)[ITER_S][ITER_C])
{
    __shared__ T data_buf[Map::kDimS * Map::kDimC];
    __shared__ T param_buf[Map::kDimS];

    SmemAccessor<T, SmemLayoutV2<Map::kDimS, Map::kDimC, Map::kDimS, Map::kDimC, Identity>> smem_K{data_buf};
    SmemAccessor<T, SmemLayoutV2<Map::kDimS, 2, Map::kDimS, 2, Identity>>                   smem_Q{param_buf};

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int2 offset = Map::get_offset(warp_id, lane_id);

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        Store(&smem_param(s * Map::kIterS + offset.y, 0), param[s]);
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            Store(&smem_data(s * Map::kDeltaS + offset.y, c * Map::kDeltaC + offset.x), data[s][c]);
        }
    }

    __syncthreads();

    static_assert(Map::kWarpCount == 4);

    // Load FP fragments
    const int offset_s = lane_id % 16 * 1 + warp_id * 16;
    const int offset_c = lane_id / 16 * 8;
    PRAGMA_UNROLL
    for (int k = 0; k < K_K; ++k) {
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            const int s = m * 16 + offset_s;
            const int c = k * 16 + offset_c;
            ldsm_x4((Array<uint32_t, 4>&)frag_K[k][m], cast_smem_ptr_to_uint(&smem_K(s, c)));
        }
    }

    // Quantize the fragments
    Array<Tk, 8> data_K[K_K][K_M];
    PRAGMA_UNROLL
    for (int k = 0; k < K_K; ++k) {
        PRAGMA_UNROLL
        for (int m = 0; m < K_M; ++m) {
            quantize(data_K[k][m], frag_K[k][m], param);
        }
    }

    // Rearrange for LDS.128

    // num fragments per LDS.128 processes
    constexpr int kFragsPerLds = 16 / sizeof(Array<Tk, 8>);

    constexpr int kWarpAccess = WARP_SIZE * kFragsPerLds * 8;

    constexpr int kDeltaC = kWarpAccess % Map::kDimC;
    constexpr int kDeltaS = kWarpAccess / Map::kDimC;

    static_assert((kDeltaC == 0) ^ (kDeltaS == 0));

    SmemAccessor<Tk, SmemLayoutV2<Map::kDimS, Map::kDimC, Map::kDimS, Map::kDimC, Identity>> smem_O{data_buf};

    const int warp_offset_s = Map::get_offset(warp_id, 0).y;
    PRAGMA_UNROLL
    for (int i = 0, s =0, c = 0; i < K_K * K_M; i += kFragsPerLds, s += kDeltaS, c += kDeltaC) {
        const int  m = i % K_M;
        const int  k = i / K_M;
        const int cc = c % Map::kDimC;
        const int ss = c / Map::kDimC + s;

        // smem_O(&warp_offset_s + n /, )
    }
}
#endif

}  // namespace turbomind
