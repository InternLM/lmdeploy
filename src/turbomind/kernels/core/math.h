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

}  // namespace turbomind