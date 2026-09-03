// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>

#include <cuda_bf16.h>

#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"

namespace turbomind::gemm::detail {

template<class Format>
struct Sm90MixedDequant;

struct alignas(16) E2m1Bf16ByteTable {
    uint32_t low0_3;
    uint32_t low4_7;
    uint32_t high0_3;
    uint32_t high4_7;
};

static_assert(sizeof(E2m1Bf16ByteTable) == 16);

// Inject a signed unbiased power-of-two exponent into the complete positive
// BF16 E2M1 table, then split its entries into the low- and high-byte tables
// consumed by PRMT. Entry zero is never adjusted, so it remains exact zero.
__device__ __forceinline__ E2m1Bf16ByteTable make_e2m1_bf16_byte_table(int exponent)
{
    constexpr uint32_t kValues0_1 = 0x3f000000u;  // {0, 0.5}
    constexpr uint32_t kValues2_3 = 0x3fc03f80u;  // {1, 1.5}
    constexpr uint32_t kValues4_5 = 0x40404000u;  // {2, 3}
    constexpr uint32_t kValues6_7 = 0x40c04080u;  // {4, 6}

    const uint32_t delta      = static_cast<uint16_t>(exponent * 128);
    const uint32_t delta_pair = delta * 0x00010001u;
    const uint32_t values0_1  = kValues0_1 + delta * 0x00010000u;
    const uint32_t values2_3  = kValues2_3 + delta_pair;
    const uint32_t values4_5  = kValues4_5 + delta_pair;
    const uint32_t values6_7  = kValues6_7 + delta_pair;

    return {
        prmt(values0_1, values2_3, 0x6420u),
        prmt(values4_5, values6_7, 0x6420u),
        prmt(values0_1, values2_3, 0x7531u),
        prmt(values4_5, values6_7, 0x7531u),
    };
}

// Each packed half contains one four-entry magnitude selector group. Sign bits
// are cross-packed so each BF16x2 pair needs at most one mask and shift.
template<int Pair>
__device__ __forceinline__ void
e2m1_prmt_bf16x4(
    uint32_t packed, const E2m1Bf16ByteTable& table, uint32_t& first, uint32_t& second)
{
    static_assert(Pair == 0 || Pair == 1);
    const uint32_t selector = (packed >> (16 * Pair)) & 0x7777u;
    const uint32_t low      = prmt(table.low0_3, table.low4_7, selector);
    const uint32_t high     = prmt(table.high0_3, table.high4_7, selector);

    if constexpr (Pair == 0) {
        first  = prmt(low, high, 0x5140u) | ((packed & 0x00080008u) << 12);
        second = prmt(low, high, 0x7362u) | ((packed & 0x08000800u) << 4);
    }
    else {
        first  = prmt(low, high, 0x5140u) | ((packed & 0x00800080u) << 8);
        second = prmt(low, high, 0x7362u) | (packed & 0x80008000u);
    }
}

__device__ __forceinline__ void
e2m1_prmt_unpack_scaled(uint32_t packed, uint16_t exponent_pair, nv_bfloat16* out)
{
    auto* h = reinterpret_cast<uint32_t*>(out);
    {
        const auto table = make_e2m1_bf16_byte_table(static_cast<int8_t>(exponent_pair & 0xffu));
        e2m1_prmt_bf16x4<0>(packed, table, h[0], h[2]);
    }
    {
        const auto table = make_e2m1_bf16_byte_table(static_cast<int8_t>(exponent_pair >> 8));
        e2m1_prmt_bf16x4<1>(packed, table, h[1], h[3]);
    }
}

}  // namespace turbomind::gemm::detail
