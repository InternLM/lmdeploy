// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/sm90_mixed_dequant.h"

namespace turbomind::gemm {

namespace detail {

inline constexpr int kNvFp4E2m1TableCount = 128;

// Construct one complete E2M1 -> BF16 lookup table for an unsigned E4M3
// NVFP4 block scale. This runs once per scale code at CTA startup; the K-loop
// only loads the two indexed tables and performs PRMT decoding.
__device__ __forceinline__ E2m1Bf16ByteTable make_nvfp4_e2m1_bf16_byte_table(uint8_t e4m3)
{
    constexpr uint32_t kValues0_1 = 0x3f000000u;  // {0, 0.5}
    constexpr uint32_t kValues2_3 = 0x3fc03f80u;  // {1, 1.5}
    constexpr uint32_t kValues4_5 = 0x40404000u;  // {2, 3}
    constexpr uint32_t kValues6_7 = 0x40c04080u;  // {4, 6}

    const uint16_t e4m3_pair = uint16_t(e4m3) | (uint16_t(e4m3) << 8);
    uint32_t       fp16_pair;
    uint16_t       scale;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(fp16_pair) : "h"(e4m3_pair));
    asm("cvt.rn.bf16.f16 %0, %1;" : "=h"(scale) : "h"(static_cast<uint16_t>(fp16_pair)));
    const uint32_t scale_pair = uint32_t(scale) | (uint32_t(scale) << 16);

    uint32_t values0_1;
    uint32_t values2_3;
    uint32_t values4_5;
    uint32_t values6_7;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(values0_1) : "r"(kValues0_1), "r"(scale_pair));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(values2_3) : "r"(kValues2_3), "r"(scale_pair));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(values4_5) : "r"(kValues4_5), "r"(scale_pair));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(values6_7) : "r"(kValues6_7), "r"(scale_pair));

    return {
        prmt(values0_1, values2_3, 0x6420u),
        prmt(values4_5, values6_7, 0x6420u),
        prmt(values0_1, values2_3, 0x7531u),
        prmt(values4_5, values6_7, 0x7531u),
    };
}

template<>
struct Sm90MixedDequant<Sm90NvFp4Format> {
    static constexpr int kWordsPerThreadKBlock = 1;

    struct SharedStorage {
        cute::array_aligned<E2m1Bf16ByteTable, kNvFp4E2m1TableCount, 16> tables;
    };

    __device__ static void init(SharedStorage& storage)
    {
        if (threadIdx.x < kNvFp4E2m1TableCount) {
            storage.tables[threadIdx.x] = make_nvfp4_e2m1_bf16_byte_table(static_cast<uint8_t>(threadIdx.x));
        }
    }

    __device__ static int packed_lane(int local_tid)
    {
        return local_tid;
    }

    template<int RestM>
    struct Registers {
        uint16_t scale_pair[RestM]{};
    };

    template<int RestM, int AtomM, int TileOut>
    __device__ static void
    load(Registers<RestM>& regs, const uint8_t* q, int segment_base, int segment_stride, int group, int local_tid)
    {
        static_assert(TileOut == RestM * AtomM * 64);
        const int pair = local_tid / 4;

        CUTE_UNROLL
        for (int rest_m = 0; rest_m < RestM; ++rest_m) {
            const int   segment     = segment_base + rest_m * segment_stride;
            const auto* fragment    = q + group * TileOut + segment * kSm90MixedFragmentN;
            regs.scale_pair[rest_m] = reinterpret_cast<const uint16_t*>(fragment)[pair];
        }
    }

    template<int RestM>
    __device__ static void dequant(const uint32_t*         packed,
                                   const Registers<RestM>& regs,
                                   int                     rest_m,
                                   int /*local_tid*/,
                                   const SharedStorage& storage,
                                   nv_bfloat16*         out)
    {
        const uint16_t scales   = regs.scale_pair[rest_m];
        const auto     table_lo = storage.tables[static_cast<uint8_t>(scales)];
        const auto     table_hi = storage.tables[static_cast<uint8_t>(scales >> 8)];
        auto*          h        = reinterpret_cast<uint32_t*>(out);
        e2m1_prmt_bf16x4<0>(packed[0], table_lo, h[0], h[2]);
        e2m1_prmt_bf16x4<1>(packed[0], table_hi, h[1], h[3]);
    }
};

}  // namespace detail

}  // namespace turbomind::gemm
