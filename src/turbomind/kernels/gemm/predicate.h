// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <type_traits>

namespace turbomind::gemm {

template<int S, int C, bool AlignedS, bool AlignedC>
struct Predicate {

    static constexpr int kSizeC = AlignedC ? 1 : C;

    static_assert(S * kSizeC <= 32);

    static constexpr bool is_active = true;

    uint32_t pred_{};

    __device__ int operator()(int s, int c) const
    {
        return (pred_ & (1 << (s * kSizeC + c))) != 0;
    }

    __device__ void set(int s, int c)
    {
        pred_ |= (1 << (s * kSizeC + c));
    }

    __device__ void clear()
    {
        pred_ = 0;
    }
};

template<int S, int C>
struct Predicate<S, C, true, true> {

    static constexpr bool is_active = false;

    __device__ constexpr std::integral_constant<int, 1> operator()(int, int) const
    {
        return {};
    }

    __device__ void set(int, int) {}

    __device__ void clear()
    {
        // pred_ = 0;
    }
};

}  // namespace turbomind::gemm
