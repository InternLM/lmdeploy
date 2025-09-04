#pragma once

#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/core/common.h"

namespace turbomind {

template<int E, int M>
struct FloatingPoint {
    static constexpr unsigned exponent_bits = E;
    static constexpr unsigned mantissa_bits = M;
    static constexpr unsigned exponent_bias = ((1 << exponent_bits) - 1) / 2;

    static constexpr unsigned bits = 1 + exponent_bits + mantissa_bits;

    static constexpr unsigned exponent_mask = (1 << exponent_bits) - 1;
    static constexpr unsigned mantissa_mask = (1 << mantissa_bits) - 1;

    // clang-format off
    // For `reinterpret_cast` is not constexpr yet
    static constexpr float exp2(unsigned e) { float x = 1; for (; e > 0; --e) { x *= 2; } return x; }
    // clang-format on

    static constexpr float max_normal =
        ((1U << (mantissa_bits + 1U)) - 1U) * exp2(exponent_bias + 1) / exp2(mantissa_bits);
    static constexpr float min_normal   = 1 / exp2(exponent_bias - 1);
    static constexpr float max_denormal = mantissa_mask / exp2(exponent_bias - 1 + mantissa_bits);
    static constexpr float min_denormal = 1 / exp2(exponent_bias - 1 + mantissa_bits);

    // Modified from `__nv_cvt_double_to_fp8` in <cuda_fp8.hpp>
    template<class R>
    __device__ static unsigned from_f32(float x, R rbits)
    {
        constexpr bool stochastic = std::is_same_v<R, unsigned>;

        // 1/2 LSB of the target format, positioned in single precision mantissa
        constexpr int half_ulp = 1U << (23U - mantissa_bits - 1U);

        auto absx = fabsf(x);

        unsigned xbits = __float_as_uint(x);

        unsigned sign     = (xbits >> 31U) << (bits - 1);
        unsigned exp      = ((xbits >> 23U) & 0xFFU) - 127U + exponent_bias;
        unsigned mantissa = (xbits >> (23U - mantissa_bits)) & mantissa_mask;

        unsigned res;

        if (absx <= min_denormal / 2.) {  // underflow
            res = 0;
        }
        else if (absx > max_normal) {  // overflow
            res = (exponent_mask << mantissa_bits) | mantissa_mask;
        }
        else if (absx >= min_normal) {  // normal
            res = (exp << mantissa_bits) | mantissa;

            unsigned round_mask = (half_ulp << 1U) - 1U;
            // rounded-off bits
            unsigned round = xbits & round_mask;
            if constexpr (stochastic) {
                // stochastic rounding (.rs) adjustment
                if (round + (rbits & round_mask) > round_mask) {
                    res += 1U;
                }
            }
            else {
                // round-to-nearest-even (.rn) adjustment
                if ((round > half_ulp) || ((round == half_ulp) && (mantissa & 1U))) {
                    res += 1U;
                }
            }
        }
        else {  // denormal
            unsigned shift = 1U - exp;
            // add implicit leading bit
            mantissa |= 1U << mantissa_bits;
            // additional round-off due to denormalization
            res = mantissa >> shift;

            unsigned round_mask = (half_ulp << (shift + 1U)) - 1U;
            // rounded-off bits, including implicit leading bit
            unsigned round = (xbits | (1U << 23U)) & round_mask;
            if constexpr (stochastic) {
                // stochastic rounding (.rs) adjustment
                if (round + (rbits & round_mask) > round_mask) {
                    res += 1U;
                }
            }
            else {
                // round-to-nearest-even (.rn) adjustment
                if ((round > (half_ulp << shift)) || ((round == (half_ulp << shift)) && (res & 1U))) {
                    res += 1U;
                }
            }
        }

        res |= sign;  // preserve sign

        return res;
    }

    __device__ static float to_f32(unsigned x)
    {
        unsigned u = (x >> (bits - 1U)) << 31U;
        u |= (x & ((1U << (bits - 1U)) - 1U)) << (23U - mantissa_bits);

        unsigned e = (127U - exponent_bias + 127U) << 23U;

        float res;
        /// ! force non-FTZ multiplication
        asm("mul.f32 %0, %1, %2;" : "=f"(res) : "r"(u), "r"(e));

        return res;
    }
};

static_assert(FloatingPoint<2, 1>::max_normal == 6);
static_assert(FloatingPoint<2, 1>::min_normal == 1);
static_assert(FloatingPoint<2, 1>::max_denormal == .5);
static_assert(FloatingPoint<2, 1>::min_denormal == .5);

static_assert(FloatingPoint<3, 2>::max_normal == 28.0);
static_assert(FloatingPoint<3, 2>::min_normal == 0.25);
static_assert(FloatingPoint<3, 2>::max_denormal == 0.1875);
static_assert(FloatingPoint<3, 2>::min_denormal == 0.0625);

static_assert(FloatingPoint<2, 3>::max_normal == 7.5);
static_assert(FloatingPoint<2, 3>::min_normal == 1.0);
static_assert(FloatingPoint<2, 3>::max_denormal == 0.875);
static_assert(FloatingPoint<2, 3>::min_denormal == 0.125);

// FloatingPoint<4, 3>::max_normal;
// FloatingPoint<4, 3>::min_normal;
// FloatingPoint<4, 3>::max_denormal;
// FloatingPoint<4, 3>::min_denormal;

// FloatingPoint<5, 2>::max_normal;
// FloatingPoint<5, 2>::min_normal;
// FloatingPoint<5, 2>::max_denormal;
// FloatingPoint<5, 2>::min_denormal;

#if 0
__device__ int cvt_rn_sat_e2m1_f32(float x)
{
    // 0000  0.0
    // 0001  0.5
    // 0010  1.0
    // 0011  1.5
    // 0100  2.0
    // 0101  3.0
    // 0110  4.0
    // 0111  6.0

    float z = fabs(x);
    //   0.25  0.75   1.25  1.75  2.5   3.5    5.0
    // 0.0   0.5   1.0   1.5   2.0   3.0   4.0   6.0
    // 0000  0001  0010  0011  0100  0101  0110  0111
    //   *           *           *           *
    auto f = [](float z) {
        if (z <= .25f) {
            return 0;
        }
        else if (z < .75f) {
            return 1;  // 0.5
        }
        else if (z <= 1.25f) {
            return 2;  // 1.0
        }
        else if (z < 1.75f) {
            return 3;  // 1.5
        }
        else if (z <= 2.5) {
            return 4;  // 2.0
        }
        else if (z < 3.5f) {
            return 5;  // 3.0
        }
        else if (z <= 5.f) {
            return 6;  // 4.0
        }
        else {
            return 7;  // 6.0
        }
    };

    return f(z) | ((__float_as_uint(x) >> 31) << 3);
}
#endif

}  // namespace turbomind
