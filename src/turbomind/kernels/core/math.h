// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include <cassert>

namespace turbomind {

template<class T>
TM_HOST_DEVICE constexpr T ceil_div(T a, T b)
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

}  // namespace turbomind
