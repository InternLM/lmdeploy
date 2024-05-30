// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

__host__ __device__ constexpr Order transpose(Order order)
{
    return order == Order::kColMajor ? Order::kRowMajor : Order::kColMajor;
}

__host__ __device__ constexpr MatrixLayout transpose(MatrixLayout x)
{
    auto tmp = x.cols;  // `std::swap` is not constexpr
    x.cols   = x.rows;
    x.rows   = tmp;
    x.order  = transpose(x.order);
    return x;
}

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
__host__ __device__ constexpr int2 _kn2cs(int k, int n)
{
    if constexpr (order == Order::kColMajor) {
        return {k, n};
    }
    else {
        return {n, k};
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
struct Packing<0> {
    __host__ __device__ static constexpr int2 apply(int2 cs)
    {
        return cs;
    }
};

template<>
struct Packing<HMMA_16816 | OPERAND_A | 1> {
    __host__ __device__ static constexpr int2 apply(int2 cs)
    {
        return {cs.x * 16, cs.y / 16};
    }
};

template<>
struct Packing<HMMA_16816 | OPERAND_B | 1>: Packing<HMMA_16816 | OPERAND_A | 1> {};

}  // namespace turbomind::gemm