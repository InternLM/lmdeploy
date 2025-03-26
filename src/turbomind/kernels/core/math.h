// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include <cassert>
#include <cstdint>
#include <type_traits>

namespace turbomind {

template<class T>
TM_HOST_DEVICE constexpr T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}

template<class T>
TM_HOST_DEVICE constexpr T cdiv(T a, T b)
{
    return (a + b - 1) / b;
}

template<class T>
TM_HOST_DEVICE constexpr T round_up(T a, T b)
{
    return (a + b - 1) / b * b;
}

template<class T>
TM_HOST_DEVICE constexpr T log2(T x)
{
    T n = 0;
    while (x != 1) {
        x /= 2;
        ++n;
    }
    return n;
}

// static_assert(log2(65536) == 16);
// static_assert(log2(32) == 5);
// static_assert(log2(1) == 0);

template<class T>
TM_HOST_DEVICE constexpr T lowbit(T x)
{
    const std::make_signed_t<T> s = x;
    return static_cast<T>(s & -s);
}

// https://arxiv.org/abs/1902.01961
template<class T>
struct FastDivMod {
};

template<>
struct FastDivMod<uint16_t> {
    uint32_t c_;  // cdiv(2^32,d) = (2^32+d-1)/d = (2^32-1)/d+1
    uint32_t d_;

    TM_HOST_DEVICE constexpr FastDivMod(uint16_t d): c_{0xFFFFFFFF / d + 1}, d_{d} {}

    template<class T>
    TM_HOST_DEVICE friend constexpr uint16_t operator/(T a, FastDivMod b)
    {
        return (a * (uint64_t)b.c_) >> 32;
    }

    template<class T>
    TM_HOST_DEVICE friend constexpr uint16_t operator%(T a, FastDivMod b)
    {
        uint64_t lowbits = (a * (uint64_t)b.c_) & 0xFFFFFFFF;
        return (lowbits * b.d_) >> 32;
    }

    TM_HOST_DEVICE constexpr operator uint16_t() const noexcept
    {
        return d_;
    }
};

static_assert(32 / FastDivMod<uint16_t>{5} == 6);
static_assert(32 % FastDivMod<uint16_t>{5} == 2);

}  // namespace turbomind
