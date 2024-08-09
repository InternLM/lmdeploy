// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/simt.h"
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
__host__ __device__ constexpr int2 mk2cs(int2 mk)
{
    return mk2cs<order>(mk.x, mk.y);
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

template<MMA_Tag mma, Op_Tag op, int num, Order order>
struct PackingImpl {
    __host__ __device__ static constexpr int2 apply(int2 mk)
    {
        return mk;
    }
};

template<Pack pack, Order order>
struct Packing_v2: PackingImpl<get_mma_tag(pack), get_operand_tag(pack), get_pack_num(pack), order> {
};

/// TODO: move packing utility to arch/smem_copy_xxx

template<int num>
struct PackingImpl<HMMA_16816, OPERAND_A, num, kRowMajor> {
    __host__ __device__ static constexpr int2 apply(int2 mk)
    {
        return {mk.x / 16 / num, mk.y * 16 * num};
    }
};

template<int num>
struct PackingImpl<HMMA_16816, OPERAND_A, num, kColMajor> {
    __host__ __device__ static constexpr int2 apply(int2 mk)
    {
        return {mk.x * 16, mk.y / 16};
    }
};

template<int num, Order order>
struct PackingImpl<HMMA_16816, OPERAND_B, num, order>: PackingImpl<HMMA_16816, OPERAND_A, num, order> {
};

template<int num>
struct PackingImpl<HMMA_SIMT, OPERAND_A, num, kRowMajor> {
    __host__ __device__ static constexpr int2 apply(int2 mk)
    {
        return {mk.x / (simt::OP_M * num), mk.y * simt::OP_M * num};
    }
};

template<int num>
struct PackingImpl<HMMA_SIMT, OPERAND_B, num, kRowMajor> {
    __host__ __device__ static constexpr int2 apply(int2 mk)
    {
        return {mk.x / (simt::OP_N * num), mk.y * simt::OP_N * num};
    }
};

template<int num>
struct PackingImpl<HMMA_884, OPERAND_B, num, kRowMajor> {
    __host__ __device__ static constexpr int2 apply(int2 mk)
    {
        // return {mk.x / (16 * num), mk.y * 16 * num};
        return {mk.x / (32 * num), mk.y * 32 * num};
    }
};

}  // namespace turbomind::gemm
