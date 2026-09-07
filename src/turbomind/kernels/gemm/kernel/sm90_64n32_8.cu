
#include <cuda.h>

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel/config.h"
#include "src/turbomind/kernels/gemm/gemm_universal_sm90_fp8_wa.h"
#include "src/turbomind/kernels/gemm/gemm_universal_sm90_v3.h"
#include "src/turbomind/kernels/gemm/kernel/e4m3.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

namespace {
void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    TM_CHECK_EQ(linear.weight.dtype(), kFloat8_e4m3);
    TM_CHECK_EQ(linear.scales.dtype(), kFloat);

    Tensor weight{{linear.weight.shape(1), linear.weight.shape(0)}, kFloat8_e4m3, kDEVICE};
    invokeTransposeAxis01(static_cast<uint8_t*>(weight.raw_data()),
                          static_cast<uint8_t*>(linear.weight.raw_data()),
                          linear.weight.shape(0),
                          linear.weight.shape(1),
                          1,
                          stream);
    linear.weight = std::move(weight);
    linear.k_desc = MatrixLayout{kFloat8_e4m3,
                                 kColMajor,
                                 (int)linear.weight.shape(1),
                                 (int)linear.weight.shape(0),
                                 (int)linear.weight.stride(0),
                                 0,
                                 0,
                                 nullptr,
                                 nullptr};

    Tensor scales{{linear.scales.shape(1), linear.scales.shape(0)}, kFloat, kDEVICE};
    invokeTransposeAxis01(static_cast<float*>(scales.raw_data()),
                          static_cast<float*>(linear.scales.raw_data()),
                          linear.scales.shape(0),
                          linear.scales.shape(1),
                          1,
                          stream);
    linear.scales = std::move(scales);
    linear.q_desc = MatrixLayout{kFloat,
                                 kColMajor,
                                 (int)linear.scales.shape(1),
                                 (int)linear.scales.shape(0),
                                 (int)linear.scales.stride(0),
                                 0,
                                 0,
                                 nullptr,
                                 nullptr};
}

const Family w8a8{28,
                  300,
                  DataFormat{kFloat8_e4m3, {128, 1}, kFloat},
                  kBfloat16,
                  1,
                  1,
                  1,
                  1,
                  false,
                  true,
                  supports_e4m3<kFloat, 128>,
                  pack,
                  128,
                  DataFormat{kFloat8_e4m3, {128, 1}, kFloat},
                  true,
                  fp8_output_spec};

struct ConfigV3 {
    template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, int MulticastA = 1, int MulticastB = 1, int MaxOpN = 128, int EpiStages = 1>
    using Type = KernelImplSm90<GemmUniversalSm90_v3<Config_, Stages, Raster, Mode, Silu, MulticastA, MulticastB, MaxOpN, EpiStages>>;
};

struct ConfigWA {
    template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, int MulticastA = 1, int MulticastB = 1>
    using Type = KernelImplSm90<GemmUniversalSm90_Fp8Wa<Config_, Stages, Raster, Mode, Silu, MulticastA, MulticastB>>;
};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, int MulticastA = 1, int MulticastB = 1, int MaxOpN = 128, int EpiStages = 1> class K>
void register_v3(Collector& c)
{
    using config::Config;
    using config::Registers;
    using config::Shape;

    {
        using _128x192_2x1 = Config<Shape<128, 192>, Shape<2, 1>, Registers<40, 232>>;
        using _128x256_2x1 = Config<Shape<128, 256>, Shape<2, 1>, Registers<40, 232>>;
        using _64x256_1x1 = Config<Shape<64, 256>, Shape<1, 1>, Registers<40, 216>>;

        // Catalog pruned per full-suite scans tmp/sm90_fp8wa_scan1 + tmp/sm90_v3_scan1
        // (2026-07-26, H200, TP/EP 1/2/4/8, swizzle 0-3; both FP8 types exercise the same
        // kernel pool, refs combined). `refs: N` = dispatch records (tuned selections);
        // `// unused` entries had zero refs and are kept visible for re-enabling.
        // --- Dense (v3 act-as-A), row raster ---
        add<K<_128x192_2x1, 4, kRowMajor, Striding::kFlat, false, 1, 1, 192, 2>>(c);  // refs: 2319
        add<K<_128x192_2x1, 4, kRowMajor, Striding::kFlat, false, 2, 1, 192, 2>>(c);  // refs: 1020
        add<K<_128x192_2x1, 4, kRowMajor, Striding::kFlat, false, 1, 2, 192, 2>>(c);  // refs: 1813
        add<K<_128x256_2x1, 4, kRowMajor, Striding::kFlat, true>>(c);  // refs: 4648
        add<K<_128x256_2x1, 4, kRowMajor, Striding::kFlat, true, 2>>(c);  // refs: 86
        add<K<_128x256_2x1, 4, kRowMajor, Striding::kFlat, true, 1, 2>>(c);  // refs: 1331
        add<K<_64x256_1x1, 4, kRowMajor, Striding::kFlat, true, 1, 1, 128, 2>>(c);  // refs: 2133
        add<K<_64x256_1x1, 4, kRowMajor, Striding::kFlat, true, 2, 1, 128, 2>>(c);  // refs: 76
        add<K<_64x256_1x1, 4, kRowMajor, Striding::kFlat, true, 1, 2, 128, 2>>(c);  // refs: 323

        // --- Dense grouped (kBlocked), col raster ---
        add<K<_128x192_2x1, 4, kColMajor, Striding::kBlocked, false, 1, 1, 192, 2>>(c);  // refs: 1247
        add<K<_128x192_2x1, 4, kColMajor, Striding::kBlocked, false, 2, 1, 192, 2>>(c);  // refs: 777
        add<K<_128x192_2x1, 4, kColMajor, Striding::kBlocked, false, 1, 2, 192, 2>>(c);  // refs: 99

    }
    {
        // --- Indexed (gate/up), col raster: N192 non-fused; N256 supports both epilogues ---
        using _128x192_2x1 = Config<Shape<128, 192>, Shape<2, 1>, Registers<88, 208>>;
        using _128x256_2x1 = Config<Shape<128, 256>, Shape<2, 1>, Registers<88, 208>>;
        using _64x256_1x1 = Config<Shape<64, 256>, Shape<1, 1>, Registers<88, 216>>;

        add<K<_128x192_2x1, 4, kColMajor, Striding::kIndexed, false, 1, 1, 192, 2>>(c);  // refs: 173
        add<K<_128x192_2x1, 4, kColMajor, Striding::kIndexed, false, 2, 1, 192, 2>>(c);  // refs: 99
        add<K<_128x192_2x1, 4, kColMajor, Striding::kIndexed, false, 1, 2, 192, 2>>(c);  // refs: 37
        add<K<_128x256_2x1, 4, kColMajor, Striding::kIndexed, true>>(c);  // refs: 6903
        add<K<_128x256_2x1, 4, kColMajor, Striding::kIndexed, true, 2>>(c);  // refs: 49
        add<K<_128x256_2x1, 4, kColMajor, Striding::kIndexed, true, 1, 2>>(c);  // refs: 196
        add<K<_64x256_1x1, 4, kColMajor, Striding::kIndexed, true, 1, 1, 128, 2>>(c);  // refs: 3824
        add<K<_64x256_1x1, 4, kColMajor, Striding::kIndexed, true, 2, 1, 128, 2>>(c);  // refs: 67
        add<K<_64x256_1x1, 4, kColMajor, Striding::kIndexed, true, 1, 2, 128, 2>>(c);  // refs: 16
    }
}

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, int MulticastA = 1, int MulticastB = 1> class K>
void register_wa(Collector& c)
{
    using config::Config;
    using config::Registers;
    using config::Shape;

    {
        using _8x128_1x1 = Config<Shape<8, 128>, Shape<1, 1>, Registers<40, 168>>;
        using _16x128_1x1 = Config<Shape<16, 128>, Shape<1, 1>, Registers<40, 168>>;
        using _32x128_1x1 = Config<Shape<32, 128>, Shape<1, 1>, Registers<40, 168>>;
        using _64x128_1x1 = Config<Shape<64, 128>, Shape<1, 1>, Registers<40, 208>>;
        using _8x256_1x1 = Config<Shape<8, 256>, Shape<1, 1>, Registers<40, 168>>;
        using _16x256_1x1 = Config<Shape<16, 256>, Shape<1, 1>, Registers<40, 168>>;
        using _32x256_1x1 = Config<Shape<32, 256>, Shape<1, 1>, Registers<40, 168>>;
        using _64x256_1x1 = Config<Shape<64, 256>, Shape<1, 1>, Registers<40, 208>>;

        // --- Weight-as-A FP8, dense (kFlat), row raster ---
        // OUT=128: plain bf16 epilogue. TILE_M<64: no (2,1) (TMA 128B alignment).
        add<K<_8x128_1x1, 4, kRowMajor, Striding::kFlat>>(c);  // refs: 5602
        add<K<_8x128_1x1, 4, kRowMajor, Striding::kFlat, false, 1, 2>>(c);  // refs: 266
        add<K<_16x128_1x1, 4, kRowMajor, Striding::kFlat>>(c);  // refs: 1745
        add<K<_16x128_1x1, 4, kRowMajor, Striding::kFlat, false, 1, 2>>(c);  // refs: 66
        add<K<_32x128_1x1, 4, kRowMajor, Striding::kFlat>>(c);  // refs: 1516
        add<K<_32x128_1x1, 4, kRowMajor, Striding::kFlat, false, 1, 2>>(c);  // refs: 145
        add<K<_64x128_1x1, 4, kRowMajor, Striding::kFlat>>(c);  // refs: 1403
        add<K<_64x128_1x1, 4, kRowMajor, Striding::kFlat, false, 2>>(c);  // refs: 46
        add<K<_64x128_1x1, 4, kRowMajor, Striding::kFlat, false, 1, 2>>(c);  // refs: 211
        // OUT=256: plain bf16 and fused SiLU->FP8 epilogues.
        add<K<_8x256_1x1, 3, kRowMajor, Striding::kFlat, true>>(c);  // refs: 1556
        add<K<_8x256_1x1, 3, kRowMajor, Striding::kFlat, true, 1, 2>>(c);  // refs: 111
        add<K<_16x256_1x1, 3, kRowMajor, Striding::kFlat, true>>(c);  // refs: 329
        add<K<_16x256_1x1, 3, kRowMajor, Striding::kFlat, true, 1, 2>>(c);  // refs: 25
        add<K<_32x256_1x1, 3, kRowMajor, Striding::kFlat, true>>(c);  // refs: 221
        // add<K<_32x256_1x1, 3, kRowMajor, Striding::kFlat, true, 1, 2>>(c);  // unused
        add<K<_64x256_1x1, 3, kRowMajor, Striding::kFlat, true>>(c);  // refs: 13
        // add<K<_64x256_1x1, 3, kRowMajor, Striding::kFlat, true, 2>>(c);  // unused
        // add<K<_64x256_1x1, 3, kRowMajor, Striding::kFlat, true, 1, 2>>(c);  // unused

        // --- Weight-as-A FP8, blocked, col raster ---
        add<K<_8x128_1x1, 4, kColMajor, Striding::kBlocked>>(c);  // refs: 37
        // add<K<_8x128_1x1, 4, kColMajor, Striding::kBlocked, false, 2>>(c);  // unused
        // add<K<_8x128_1x1, 4, kColMajor, Striding::kBlocked, false, 1, 2>>(c);  // unused
        add<K<_16x128_1x1, 4, kColMajor, Striding::kBlocked>>(c);  // refs: 36
        // add<K<_16x128_1x1, 4, kColMajor, Striding::kBlocked, false, 2>>(c);  // unused
        // add<K<_16x128_1x1, 4, kColMajor, Striding::kBlocked, false, 1, 2>>(c);  // unused
        add<K<_32x128_1x1, 4, kColMajor, Striding::kBlocked>>(c);  // refs: 63
        // add<K<_32x128_1x1, 4, kColMajor, Striding::kBlocked, false, 2>>(c);  // unused
        // add<K<_32x128_1x1, 4, kColMajor, Striding::kBlocked, false, 1, 2>>(c);  // unused
        add<K<_64x128_1x1, 4, kColMajor, Striding::kBlocked>>(c);  // refs: 10
        // add<K<_64x128_1x1, 4, kColMajor, Striding::kBlocked, false, 2>>(c);  // unused
        // add<K<_64x128_1x1, 4, kColMajor, Striding::kBlocked, false, 1, 2>>(c);  // unused
        add<K<_8x256_1x1, 3, kColMajor, Striding::kBlocked>>(c);  // refs: 1377
        add<K<_8x256_1x1, 3, kColMajor, Striding::kBlocked, false, 2>>(c);  // refs: 2
        // add<K<_8x256_1x1, 3, kColMajor, Striding::kBlocked, false, 1, 2>>(c);  // unused
        add<K<_16x256_1x1, 3, kColMajor, Striding::kBlocked>>(c);  // refs: 439
        // add<K<_16x256_1x1, 3, kColMajor, Striding::kBlocked, false, 2>>(c);  // unused
        // add<K<_16x256_1x1, 3, kColMajor, Striding::kBlocked, false, 1, 2>>(c);  // unused
        add<K<_32x256_1x1, 3, kColMajor, Striding::kBlocked>>(c);  // refs: 87
        // add<K<_32x256_1x1, 3, kColMajor, Striding::kBlocked, false, 2>>(c);  // unused
        // add<K<_32x256_1x1, 3, kColMajor, Striding::kBlocked, false, 1, 2>>(c);  // unused
        // add<K<_64x256_1x1, 3, kColMajor, Striding::kBlocked>>(c);  // unused
        // add<K<_64x256_1x1, 3, kColMajor, Striding::kBlocked, false, 2>>(c);  // unused
        // add<K<_64x256_1x1, 3, kColMajor, Striding::kBlocked, false, 1, 2>>(c);  // unused

    }
    {
        // --- Weight-as-A FP8, indexed (gate/up), col raster ---
        using _8x128_1x1 = Config<Shape<8, 128>, Shape<1, 1>, Registers<88, 168>>;
        using _16x128_1x1 = Config<Shape<16, 128>, Shape<1, 1>, Registers<88, 168>>;
        using _32x128_1x1 = Config<Shape<32, 128>, Shape<1, 1>, Registers<88, 168>>;
        using _64x128_1x1 = Config<Shape<64, 128>, Shape<1, 1>, Registers<88, 208>>;
        using _8x256_1x1 = Config<Shape<8, 256>, Shape<1, 1>, Registers<88, 168>>;
        using _16x256_1x1 = Config<Shape<16, 256>, Shape<1, 1>, Registers<88, 168>>;
        using _32x256_1x1 = Config<Shape<32, 256>, Shape<1, 1>, Registers<88, 168>>;
        // using _64x256_1x1 = Config<Shape<64, 256>, Shape<1, 1>, Registers<88, 208>>;
        using _64x256_1x2 = Config<Shape<64, 256>, Shape<1, 2>, Registers<40, 232>>;
        using _128x128_2x1 = Config<Shape<128, 128>, Shape<2, 1>, Registers<40, 232>>;
        using _128x256_2x1 = Config<Shape<128, 256>, Shape<2, 1>, Registers<40, 232>>;

        add<K<_8x128_1x1, 4, kColMajor, Striding::kIndexed>>(c);  // refs: 570
        // add<K<_8x128_1x1, 4, kColMajor, Striding::kIndexed, false, 2>>(c);  // unused
        add<K<_8x128_1x1, 4, kColMajor, Striding::kIndexed, false, 1, 2>>(c);  // refs: 20
        add<K<_16x128_1x1, 4, kColMajor, Striding::kIndexed>>(c);  // refs: 157
        // add<K<_16x128_1x1, 4, kColMajor, Striding::kIndexed, false, 2>>(c);  // unused
        // add<K<_16x128_1x1, 4, kColMajor, Striding::kIndexed, false, 1, 2>>(c);  // unused
        add<K<_32x128_1x1, 4, kColMajor, Striding::kIndexed>>(c);  // refs: 96
        // add<K<_32x128_1x1, 4, kColMajor, Striding::kIndexed, false, 2>>(c);  // unused
        // add<K<_32x128_1x1, 4, kColMajor, Striding::kIndexed, false, 1, 2>>(c);  // unused
        add<K<_64x128_1x1, 4, kColMajor, Striding::kIndexed>>(c);  // refs: 36
        // add<K<_64x128_1x1, 4, kColMajor, Striding::kIndexed, false, 2>>(c);  // unused
        // add<K<_64x128_1x1, 4, kColMajor, Striding::kIndexed, false, 1, 2>>(c);  // unused
        // Indexed OUT=256 kernels also select the epilogue at runtime.
        add<K<_8x256_1x1, 3, kColMajor, Striding::kIndexed, true>>(c);  // refs: 4553
        add<K<_8x256_1x1, 3, kColMajor, Striding::kIndexed, true, 2>>(c);  // refs: 32
        add<K<_8x256_1x1, 3, kColMajor, Striding::kIndexed, true, 1, 2>>(c);  // refs: 4
        add<K<_16x256_1x1, 3, kColMajor, Striding::kIndexed, true>>(c);  // refs: 1253
        add<K<_16x256_1x1, 3, kColMajor, Striding::kIndexed, true, 2>>(c);  // refs: 7
        // add<K<_16x256_1x1, 3, kColMajor, Striding::kIndexed, true, 1, 2>>(c);  // unused
        add<K<_32x256_1x1, 3, kColMajor, Striding::kIndexed, true>>(c);  // refs: 340
        // add<K<_32x256_1x1, 3, kColMajor, Striding::kIndexed, true, 2>>(c);  // unused
        // add<K<_32x256_1x1, 3, kColMajor, Striding::kIndexed, true, 1, 2>>(c);  // unused
        // add<K<_64x256_1x1, 3, kColMajor, Striding::kIndexed, true>>(c);  // unused
        // add<K<_64x256_1x1, 3, kColMajor, Striding::kIndexed, true, 2>>(c);  // unused
        // add<K<_64x256_1x1, 3, kColMajor, Striding::kIndexed, true, 1, 2>>(c);  // unused

        // --- 2 math WGs (per-WG BATCH=64): 128x128 plain, 128x256 fused, 64x256_n2 ---
        add<K<_64x256_1x2, 3, kRowMajor, Striding::kFlat, true>>(c);
        add<K<_64x256_1x2, 3, kRowMajor, Striding::kFlat, true, 2>>(c);
        add<K<_64x256_1x2, 3, kRowMajor, Striding::kFlat, true, 1, 2>>(c);
        add<K<_128x128_2x1, 3, kRowMajor, Striding::kFlat>>(c);
        add<K<_128x128_2x1, 3, kRowMajor, Striding::kFlat, false, 2>>(c);
        add<K<_128x128_2x1, 3, kRowMajor, Striding::kFlat, false, 1, 2>>(c);
        add<K<_128x256_2x1, 3, kRowMajor, Striding::kFlat, true>>(c);
        add<K<_128x256_2x1, 3, kRowMajor, Striding::kFlat, true, 2>>(c);
        add<K<_128x256_2x1, 3, kRowMajor, Striding::kFlat, true, 1, 2>>(c);
        add<K<_128x128_2x1, 3, kColMajor, Striding::kBlocked>>(c);
        add<K<_128x128_2x1, 3, kColMajor, Striding::kBlocked, false, 2>>(c);
        add<K<_128x128_2x1, 3, kColMajor, Striding::kBlocked, false, 1, 2>>(c);
        add<K<_128x256_2x1, 3, kColMajor, Striding::kBlocked>>(c);
        add<K<_128x256_2x1, 3, kColMajor, Striding::kBlocked, false, 2>>(c);
        add<K<_128x256_2x1, 3, kColMajor, Striding::kBlocked, false, 1, 2>>(c);
        add<K<_64x256_1x2, 3, kColMajor, Striding::kBlocked>>(c);
        add<K<_64x256_1x2, 3, kColMajor, Striding::kBlocked, false, 2>>(c);
        add<K<_64x256_1x2, 3, kColMajor, Striding::kBlocked, false, 1, 2>>(c);
    }
    {
        using _128x128_2x1 = Config<Shape<128, 128>, Shape<2, 1>, Registers<88, 208>>;
        using _128x256_2x1 = Config<Shape<128, 256>, Shape<2, 1>, Registers<88, 208>>;
        using _64x256_1x2 = Config<Shape<64, 256>, Shape<1, 2>, Registers<88, 208>>;

        add<K<_128x128_2x1, 3, kColMajor, Striding::kIndexed>>(c);
        add<K<_128x128_2x1, 3, kColMajor, Striding::kIndexed, false, 2>>(c);
        add<K<_128x128_2x1, 3, kColMajor, Striding::kIndexed, false, 1, 2>>(c);
        add<K<_128x256_2x1, 3, kColMajor, Striding::kIndexed, true>>(c);
        add<K<_128x256_2x1, 3, kColMajor, Striding::kIndexed, true, 2>>(c);
        add<K<_128x256_2x1, 3, kColMajor, Striding::kIndexed, true, 1, 2>>(c);
        add<K<_64x256_1x2, 3, kColMajor, Striding::kIndexed, true>>(c);
        add<K<_64x256_1x2, 3, kColMajor, Striding::kIndexed, true, 2>>(c);
        add<K<_64x256_1x2, 3, kColMajor, Striding::kIndexed, true, 1, 2>>(c);
    }
}

Registrar reg(w8a8, [](Collector& c) {
    register_v3<ConfigV3::Type>(c);
    register_wa<ConfigWA::Type>(c);
});
}  // namespace

}  // namespace turbomind::gemm
