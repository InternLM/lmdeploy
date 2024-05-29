// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

template<Order order>
__host__ __device__ constexpr int2 mk2cs(int m, int k)
{
    if constexpr (order == Order::kRowMajor) {
        return {k, m};
    }
    else {
        return {m, k};
    }
}

template<Order order>
__host__ __device__ constexpr int2 cs2mk(int c, int s)
{
    if constexpr (order == Order::kRowMajor) {
        return {s, c};
    }
    else {
        return {c, s};
    }
}

template<Order order>
__host__ __device__ constexpr int2 kn2cs(int k, int n)
{
    if constexpr (order == Order::kColMajor) {
        return {k, n};
    }
    else {
        return {n, k};
    }
}

template<Order order>
__host__ __device__ constexpr int2 cs2kn(int c, int s)
{
    if constexpr (order == Order::kColMajor) {
        return {c, s};
    }
    else {
        return {s, c};
    }
}

template<class Index>
__host__ __device__ constexpr Index cs2idx(int2 cs, Index ld)
{
    return ld * cs.y + cs.x;
}

template<Pack pack>
struct Packing {};

template<>
struct Packing<Pack::kNone> {
    __host__ __device__ static constexpr int2 apply(int2 cs)
    {
        return cs;
    }
};

template<>
struct Packing<Pack::kHMMA_16816_A> {
    __host__ __device__ static constexpr int2 apply(int2 cs)
    {
        return {cs.x * 16, cs.y / 16};
    }
};

template<>
struct Packing<Pack::kHMMA_16816_B>: Packing<Pack::kHMMA_16816_A> {};

}  // namespace turbomind::gemm