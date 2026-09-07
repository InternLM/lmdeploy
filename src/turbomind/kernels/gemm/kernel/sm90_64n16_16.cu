// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/cublas.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel/config.h"
#include "src/turbomind/kernels/gemm/gemm_universal_sm90_bf16.h"
#include "src/turbomind/kernels/gemm/kernel/floating_point.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90_bf16.h"
#include "src/turbomind/kernels/gemm/types.h"

#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

namespace {
void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    TM_CHECK_EQ(linear.weight.dtype(), kBfloat16);
    Tensor trans{{linear.weight.shape(1), linear.weight.shape(0)}, kBfloat16, kDEVICE};
    invokeTransposeAxis01(static_cast<nv_bfloat16*>(trans.raw_data()),
                          static_cast<nv_bfloat16*>(linear.weight.raw_data()),
                          linear.weight.shape(0),
                          linear.weight.shape(1),
                          1,
                          stream);
    linear.weight        = std::move(trans);
    linear.k_desc        = MatrixLayout{kBfloat16,
                                        kColMajor,
                                        linear.input_dim,
                                        linear.output_dim,
                                        (int)linear.weight.stride(0),
                                        0,
                                        0,
                                        nullptr,
                                        nullptr};
    linear.q_desc        = {};
    linear.weight_format = kBfloat16;
}

const Family bf16{
    27, 250, kBfloat16, kBfloat16, 1, 1, 1, 1, true, true, supports_fp<kBfloat16>, pack, 64, kBfloat16};

struct C {
    template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, int MulticastA = 1, int MulticastB = 1, int L2HintW = 0, int MmaN = 0, bool SeparateMmaAtoms = false, int EpiM = 0, int EpiStages = 0>
    using Type = KernelImplSm90Bf16<GemmUniversalSm90_Bf16<Config_, Stages, Raster, Mode, Silu, MulticastA, MulticastB, L2HintW, MmaN, SeparateMmaAtoms, EpiM, EpiStages>>;
};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, int MulticastA = 1, int MulticastB = 1, int L2HintW = 0, int MmaN = 0, bool SeparateMmaAtoms = false, int EpiM = 0, int EpiStages = 0> class K>
void register_kernels(Collector& c)
{
    using config::Config;
    using config::Registers;
    using config::Shape;

    {
        // using _8x128x64_1x2 = Config<Shape<8, 128, 64>, Shape<1, 2>, Registers<24, 80>>;
        // using _16x128x64_1x2 = Config<Shape<16, 128, 64>, Shape<1, 2>, Registers<24, 80>>;
        // using _32x128x64_1x2 = Config<Shape<32, 128, 64>, Shape<1, 2>, Registers<24, 80>>;
        // using _64x128x64_1x2 = Config<Shape<64, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _96x128x64_1x2 = Config<Shape<96, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _128x128x64_1x2 = Config<Shape<128, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _192x128x64_1x2 = Config<Shape<192, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _224x128x64_1x2 = Config<Shape<224, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _256x128x64_1x2 = Config<Shape<256, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _320x128x64_1x2 = Config<Shape<320, 128, 64>, Shape<1, 2>, Registers<40, 232>>;
        using _384x128x64_1x2 = Config<Shape<384, 128, 64>, Shape<1, 2>, Registers<40, 232>>;
        // using _320x128x64_2x1 = Config<Shape<320, 128, 64>, Shape<2, 1>, Registers<40, 232>>;
        // using _384x128x64_2x1 = Config<Shape<384, 128, 64>, Shape<2, 1>, Registers<24, 240>>;
        using _8x256x64_1x2 = Config<Shape<8, 256, 64>, Shape<1, 2>, Registers<24, 80>>;
        using _8x256x64_1x1 = Config<Shape<8, 256, 64>, Shape<1, 1>, Registers<24, 80>>;
        using _16x256x64_2x1 = Config<Shape<16, 256, 64>, Shape<2, 1>, Registers<24, 80>>;
        using _32x256x64_2x1 = Config<Shape<32, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        using _64x256x64_2x1 = Config<Shape<64, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        using _96x256x64_2x1 = Config<Shape<96, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        using _128x256x64_2x1 = Config<Shape<128, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        using _160x256x64_1x2 = Config<Shape<160, 256, 64>, Shape<1, 2>, Registers<40, 232>>;
        using _192x256x64_1x2 = Config<Shape<192, 256, 64>, Shape<1, 2>, Registers<40, 232>>;
        // using _64x128x64_2x1 = Config<Shape<64, 128, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _128x128x64_2x1 = Config<Shape<128, 128, 64>, Shape<2, 1>, Registers<120, 192>>;

        add_cublas(c, Sm90::is_compatible);

        // Catalog pruned per full-suite scan tmp/sm90_bf16_scan5; refs refreshed per
        // tmp/sm90_bf16_scan8 (2026-07-26, H200, TP/EP 1/2/4/8, swizzle 0-3). `refs: N` =
        // dispatch records (tuned selections) the kernel accumulated across the scan;
        // `// unused` entries had zero refs and are kept visible for re-enabling.
        // Only cluster (1,1) was ever selected; (2,1) / (1,2) variants were dropped entirely.

        // --- Dense (kFlat), row raster ---
        // Legacy N128 tiles never selected. The new 384x128 WG_1x2 kernel is the
        // measured large dense winner; the other validated candidates remain visible.
        // add<K<_8x128x64_1x2, 4, kRowMajor, Striding::kFlat>>(c);  // unused
        // add<K<_16x128x64_1x2, 4, kRowMajor, Striding::kFlat>>(c);  // unused
        // add<K<_32x128x64_1x2, 4, kRowMajor, Striding::kFlat>>(c);  // unused
        // add<K<_64x128x64_1x2, 4, kRowMajor, Striding::kFlat>>(c);  // unused
        // add<K<_96x128x64_1x2, 4, kRowMajor, Striding::kFlat>>(c);  // unused
        // add<K<_128x128x64_1x2, 4, kRowMajor, Striding::kFlat>>(c);  // unused
        // add<K<_192x128x64_1x2, 4, kRowMajor, Striding::kFlat>>(c);  // unused
        // add<K<_224x128x64_1x2, 4, kRowMajor, Striding::kFlat>>(c);  // unused
        // add<K<_256x128x64_1x2, 4, kRowMajor, Striding::kFlat>>(c);  // unused
        // add<K<_320x128x64_1x2, 3, kRowMajor, Striding::kFlat, false, 1, 1, 0, 160>>(c);  // validated, slower
        add<K<_384x128x64_1x2, 3, kRowMajor, Striding::kFlat, false, 1, 1, 0, 192>>(c);
        // add<K<_320x128x64_2x1, 3, kRowMajor, Striding::kFlat>>(c);  // validated, slower
        // add<K<_384x128x64_2x1, 3, kRowMajor, Striding::kFlat, false, 1, 1, 0, 0, true, 64>>(c);  // validated, slower
        add<K<_8x256x64_1x2, 3, kRowMajor, Striding::kFlat, true>>(c);  // refs: 274
        add<K<_8x256x64_1x1, 3, kRowMajor, Striding::kFlat, true>>(c);  // refs: 1267
        add<K<_16x256x64_2x1, 3, kRowMajor, Striding::kFlat, true>>(c);  // refs: 21
        add<K<_32x256x64_2x1, 3, kRowMajor, Striding::kFlat, true>>(c);  // refs: 235
        add<K<_64x256x64_2x1, 3, kRowMajor, Striding::kFlat, true>>(c);  // refs: 268
        add<K<_96x256x64_2x1, 3, kRowMajor, Striding::kFlat, true>>(c);  // refs: 295
        add<K<_128x256x64_2x1, 3, kRowMajor, Striding::kFlat, true>>(c);  // refs: 1160
        add<K<_160x256x64_1x2, 3, kRowMajor, Striding::kFlat, true>>(c);
        add<K<_192x256x64_1x2, 3, kRowMajor, Striding::kFlat, true, 1, 1, 0, 0, false, 192>>(c);

        // --- Dense (kFlat), col raster: never selected ---
        // add<K<_8x128x64_1x2, 4, kColMajor, Striding::kFlat>>(c);  // unused
        // add<K<_16x128x64_1x2, 4, kColMajor, Striding::kFlat>>(c);  // unused
        // add<K<_32x128x64_1x2, 4, kColMajor, Striding::kFlat>>(c);  // unused
        // add<K<_64x128x64_1x2, 4, kColMajor, Striding::kFlat>>(c);  // unused
        // add<K<_96x128x64_1x2, 4, kColMajor, Striding::kFlat>>(c);  // unused
        // add<K<_128x128x64_1x2, 4, kColMajor, Striding::kFlat>>(c);  // unused
        // add<K<_192x128x64_1x2, 4, kColMajor, Striding::kFlat>>(c);  // unused
        // add<K<_224x128x64_1x2, 4, kColMajor, Striding::kFlat>>(c);  // unused
        // add<K<_256x128x64_1x2, 4, kColMajor, Striding::kFlat>>(c);  // unused
        // add<K<_8x256x64_1x2, 3, kColMajor, Striding::kFlat, true>>(c);  // unused
        // add<K<_8x256x64_1x1, 3, kColMajor, Striding::kFlat, true>>(c);  // unused
        // add<K<_16x256x64_2x1, 3, kColMajor, Striding::kFlat, true>>(c);  // unused
        // add<K<_32x256x64_2x1, 3, kColMajor, Striding::kFlat, true>>(c);  // unused
        // add<K<_64x256x64_2x1, 3, kColMajor, Striding::kFlat, true>>(c);  // unused
        // add<K<_96x256x64_2x1, 3, kColMajor, Striding::kFlat, true>>(c);  // unused
        // add<K<_128x256x64_2x1, 3, kColMajor, Striding::kFlat, true>>(c);  // unused

        // --- Dense N128 WG_2x1 (either raster): never selected ---
        // add<K<_64x128x64_2x1, 4, kRowMajor, Striding::kFlat>>(c);  // unused
        // add<K<_128x128x64_2x1, 4, kRowMajor, Striding::kFlat>>(c);  // unused

    }
    {
        // --- MoE gate_up (kIndexed), col raster ---
        using _8x128x64_1x2 = Config<Shape<8, 128, 64>, Shape<1, 2>, Registers<40, 80>>;
        // using _16x128x64_1x2 = Config<Shape<16, 128, 64>, Shape<1, 2>, Registers<40, 80>>;
        // using _32x128x64_1x2 = Config<Shape<32, 128, 64>, Shape<1, 2>, Registers<48, 80>>;
        using _64x128x64_1x2 = Config<Shape<64, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        using _96x128x64_1x2 = Config<Shape<96, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        using _128x128x64_1x2 = Config<Shape<128, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _192x128x64_1x2 = Config<Shape<192, 128, 64>, Shape<1, 2>, Registers<104, 200>>;
        // using _224x128x64_1x2 = Config<Shape<224, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        using _256x128x64_1x2 = Config<Shape<256, 128, 64>, Shape<1, 2>, Registers<128, 184>>;
        using _8x256x64_1x2 = Config<Shape<8, 256, 64>, Shape<1, 2>, Registers<40, 80>>;
        using _8x256x64_1x1 = Config<Shape<8, 256, 64>, Shape<1, 1>, Registers<40, 80>>;
        using _16x256x64_2x1 = Config<Shape<16, 256, 64>, Shape<2, 1>, Registers<40, 80>>;
        using _32x256x64_2x1 = Config<Shape<32, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        using _64x256x64_2x1 = Config<Shape<64, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        using _96x256x64_2x1 = Config<Shape<96, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        using _128x256x64_2x1 = Config<Shape<128, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        using _160x256x64_1x2 = Config<Shape<160, 256, 64>, Shape<1, 2>, Registers<104, 200>>;
        // using _64x128x64_2x1 = Config<Shape<64, 128, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _128x128x64_2x1 = Config<Shape<128, 128, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _64x256x64_1x2 = Config<Shape<64, 256, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _128x256x64_1x2 = Config<Shape<128, 256, 64>, Shape<1, 2>, Registers<120, 192>>;
        using _192x256x64_1x2 = Config<Shape<192, 256, 64>, Shape<1, 2>, Registers<40, 232>>;

        add<K<_8x128x64_1x2, 4, kColMajor, Striding::kIndexed>>(c);  // refs: 1436
        // add<K<_16x128x64_1x2, 4, kColMajor, Striding::kIndexed>>(c);  // unused
        // add<K<_32x128x64_1x2, 4, kColMajor, Striding::kIndexed>>(c);  // unused
        add<K<_64x128x64_1x2, 4, kColMajor, Striding::kIndexed>>(c);  // refs: 846
        add<K<_96x128x64_1x2, 4, kColMajor, Striding::kIndexed>>(c);  // refs: 183
        add<K<_128x128x64_1x2, 4, kColMajor, Striding::kIndexed>>(c);  // refs: 665
        // add<K<_192x128x64_1x2, 4, kColMajor, Striding::kIndexed>>(c);  // unused
        // add<K<_224x128x64_1x2, 4, kColMajor, Striding::kIndexed>>(c);  // unused
        add<K<_256x128x64_1x2, 4, kColMajor, Striding::kIndexed>>(c);  // refs: 2207
        add<K<_8x256x64_1x2, 3, kColMajor, Striding::kIndexed, true>>(c);  // refs: 399
        add<K<_8x256x64_1x1, 3, kColMajor, Striding::kIndexed, true>>(c);  // refs: 1362
        add<K<_16x256x64_2x1, 3, kColMajor, Striding::kIndexed, true>>(c);  // refs: 1035
        add<K<_32x256x64_2x1, 3, kColMajor, Striding::kIndexed, true>>(c);  // refs: 495
        add<K<_64x256x64_2x1, 3, kColMajor, Striding::kIndexed, true>>(c);  // refs: 803
        add<K<_96x256x64_2x1, 3, kColMajor, Striding::kIndexed, true>>(c);  // refs: 604
        add<K<_128x256x64_2x1, 3, kColMajor, Striding::kIndexed, true>>(c);  // refs: 2420
        add<K<_160x256x64_1x2, 3, kColMajor, Striding::kIndexed, true>>(c);

        // --- MoE gate_up (kIndexed), row raster ---
        // add<K<_8x128x64_1x2, 4, kRowMajor, Striding::kIndexed>>(c);  // unused
        // add<K<_16x128x64_1x2, 4, kRowMajor, Striding::kIndexed>>(c);  // unused
        // add<K<_32x128x64_1x2, 4, kRowMajor, Striding::kIndexed>>(c);  // unused
        // add<K<_64x128x64_1x2, 4, kRowMajor, Striding::kIndexed>>(c);  // unused
        // add<K<_96x128x64_1x2, 4, kRowMajor, Striding::kIndexed>>(c);  // unused
        // add<K<_128x128x64_1x2, 4, kRowMajor, Striding::kIndexed>>(c);  // unused
        // add<K<_192x128x64_1x2, 4, kRowMajor, Striding::kIndexed>>(c);  // unused
        // add<K<_224x128x64_1x2, 4, kRowMajor, Striding::kIndexed>>(c);  // unused
        add<K<_256x128x64_1x2, 4, kRowMajor, Striding::kIndexed>>(c);  // refs: 1200
        // add<K<_8x256x64_1x2, 3, kRowMajor, Striding::kIndexed, true>>(c);  // unused
        // add<K<_8x256x64_1x1, 3, kRowMajor, Striding::kIndexed, true>>(c);  // unused
        // add<K<_16x256x64_2x1, 3, kRowMajor, Striding::kIndexed, true>>(c);  // unused
        // add<K<_32x256x64_2x1, 3, kRowMajor, Striding::kIndexed, true>>(c);  // unused
        // add<K<_64x256x64_2x1, 3, kRowMajor, Striding::kIndexed, true>>(c);  // unused
        // add<K<_96x256x64_2x1, 3, kRowMajor, Striding::kIndexed, true>>(c);  // unused
        // add<K<_128x256x64_2x1, 3, kRowMajor, Striding::kIndexed, true>>(c);  // unused

        // --- MoE gate_up N128 WG_2x1 / N256 WG_1x2 (either raster): never selected ---
        // add<K<_64x128x64_2x1, 4, kColMajor, Striding::kIndexed>>(c);  // unused
        // add<K<_128x128x64_2x1, 4, kColMajor, Striding::kIndexed>>(c);  // unused
        // add<K<_64x256x64_1x2, 3, kColMajor, Striding::kIndexed>>(c);  // unused
        // add<K<_128x256x64_1x2, 3, kColMajor, Striding::kIndexed>>(c);  // unused

        // --- MoE down: reuse kIndexed except for 192x256, whose gather path cannot
        // reserve enough math registers without C7512 or spills. ---
        add<K<_192x256x64_1x2, 3, kColMajor, Striding::kBlocked, false, 1, 1, 0, 0, false, 192>>(c);
    }
#if 0
    {
        // Dedicated kBlocked kernels.
        using _8x128x64_1x2 = Config<Shape<8, 128, 64>, Shape<1, 2>, Registers<24, 80>>;
        using _16x128x64_1x2 = Config<Shape<16, 128, 64>, Shape<1, 2>, Registers<24, 80>>;
        using _32x128x64_1x2 = Config<Shape<32, 128, 64>, Shape<1, 2>, Registers<24, 80>>;
        // using _64x128x64_1x2 = Config<Shape<64, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        using _96x128x64_1x2 = Config<Shape<96, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _128x128x64_1x2 = Config<Shape<128, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _192x128x64_1x2 = Config<Shape<192, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _224x128x64_1x2 = Config<Shape<224, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _256x128x64_1x2 = Config<Shape<256, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _64x128x64_2x1 = Config<Shape<64, 128, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _128x128x64_2x1 = Config<Shape<128, 128, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _32x256x64_2x1 = Config<Shape<32, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _64x256x64_2x1 = Config<Shape<64, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _96x256x64_2x1 = Config<Shape<96, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _128x256x64_2x1 = Config<Shape<128, 256, 64>, Shape<2, 1>, Registers<120, 192>>;

        add<K<_8x128x64_1x2, 4, kColMajor, Striding::kBlocked>>(c);  // refs: 2637
        add<K<_16x128x64_1x2, 4, kColMajor, Striding::kBlocked>>(c);  // refs: 1419
        add<K<_32x128x64_1x2, 4, kColMajor, Striding::kBlocked>>(c);  // refs: 1375
        // add<K<_64x128x64_1x2, 4, kColMajor, Striding::kBlocked>>(c);  // unused
        add<K<_96x128x64_1x2, 4, kColMajor, Striding::kBlocked>>(c);  // refs: 740
        // add<K<_128x128x64_1x2, 4, kColMajor, Striding::kBlocked>>(c);  // unused
        // add<K<_192x128x64_1x2, 4, kColMajor, Striding::kBlocked>>(c);  // unused
        // add<K<_224x128x64_1x2, 4, kColMajor, Striding::kBlocked>>(c);  // unused
        // add<K<_256x128x64_1x2, 4, kColMajor, Striding::kBlocked>>(c);  // unused

        // --- MoE down (kBlocked), row raster: never selected ---
        // add<K<_8x128x64_1x2, 4, kRowMajor, Striding::kBlocked>>(c);  // unused
        // add<K<_16x128x64_1x2, 4, kRowMajor, Striding::kBlocked>>(c);  // unused
        // add<K<_32x128x64_1x2, 4, kRowMajor, Striding::kBlocked>>(c);  // unused
        // add<K<_96x128x64_1x2, 4, kRowMajor, Striding::kBlocked>>(c);  // unused
        // add<K<_192x128x64_1x2, 4, kRowMajor, Striding::kBlocked>>(c);  // unused
        // add<K<_224x128x64_1x2, 4, kRowMajor, Striding::kBlocked>>(c);  // unused
        // add<K<_256x128x64_1x2, 4, kRowMajor, Striding::kBlocked>>(c);  // unused

        // --- MoE down N128 WG_2x1 (either raster): never selected ---
        // add<K<_64x128x64_2x1, 4, kColMajor, Striding::kBlocked>>(c);  // unused
        // add<K<_128x128x64_2x1, 4, kColMajor, Striding::kBlocked>>(c);  // unused
    }
#endif

    // --- Weight L2 evict-first hint (l2_hint_w=1, desc policy_b=1), 1-CTA tiles only ---
    // On the 2-CTA small tiles (8/16/32x128, 16x256) the hint measured 5-9% worse on H200.
    // Only indexed 256x128 was selected; everything else pruned per the scan.
    {
        // Dense L2 candidates retain their TMA register budgets.
        // using _64x128x64_1x2 = Config<Shape<64, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _96x128x64_1x2 = Config<Shape<96, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _128x128x64_1x2 = Config<Shape<128, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _192x128x64_1x2 = Config<Shape<192, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _224x128x64_1x2 = Config<Shape<224, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _256x128x64_1x2 = Config<Shape<256, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _32x256x64_2x1 = Config<Shape<32, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _64x256x64_2x1 = Config<Shape<64, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _96x256x64_2x1 = Config<Shape<96, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _128x256x64_2x1 = Config<Shape<128, 256, 64>, Shape<2, 1>, Registers<120, 192>>;

        // add<K<_64x128x64_1x2, 4, kRowMajor, Striding::kFlat, false, 1, 1, 1>>(c);  // unused
        // add<K<_96x128x64_1x2, 4, kRowMajor, Striding::kFlat, false, 1, 1, 1>>(c);  // unused
        // add<K<_128x128x64_1x2, 4, kRowMajor, Striding::kFlat, false, 1, 1, 1>>(c);  // unused
        // add<K<_192x128x64_1x2, 4, kRowMajor, Striding::kFlat, false, 1, 1, 1>>(c);  // unused
        // add<K<_224x128x64_1x2, 4, kRowMajor, Striding::kFlat, false, 1, 1, 1>>(c);  // unused
        // add<K<_256x128x64_1x2, 4, kRowMajor, Striding::kFlat, false, 1, 1, 1>>(c);  // unused
        // add<K<_32x256x64_2x1, 3, kRowMajor, Striding::kFlat, true, 1, 1, 1>>(c);  // unused
        // add<K<_64x256x64_2x1, 3, kRowMajor, Striding::kFlat, true, 1, 1, 1>>(c);  // unused
        // add<K<_96x256x64_2x1, 3, kRowMajor, Striding::kFlat, true, 1, 1, 1>>(c);  // unused
        // add<K<_128x256x64_2x1, 3, kRowMajor, Striding::kFlat, true, 1, 1, 1>>(c);  // unused
        // add<K<_64x128x64_1x2, 4, kColMajor, Striding::kFlat, false, 1, 1, 1>>(c);  // unused
        // add<K<_96x128x64_1x2, 4, kColMajor, Striding::kFlat, false, 1, 1, 1>>(c);  // unused
        // add<K<_128x128x64_1x2, 4, kColMajor, Striding::kFlat, false, 1, 1, 1>>(c);  // unused
        // add<K<_192x128x64_1x2, 4, kColMajor, Striding::kFlat, false, 1, 1, 1>>(c);  // unused
        // add<K<_224x128x64_1x2, 4, kColMajor, Striding::kFlat, false, 1, 1, 1>>(c);  // unused
        // add<K<_256x128x64_1x2, 4, kColMajor, Striding::kFlat, false, 1, 1, 1>>(c);  // unused
        // add<K<_32x256x64_2x1, 3, kColMajor, Striding::kFlat, true, 1, 1, 1>>(c);  // unused
        // add<K<_64x256x64_2x1, 3, kColMajor, Striding::kFlat, true, 1, 1, 1>>(c);  // unused
        // add<K<_96x256x64_2x1, 3, kColMajor, Striding::kFlat, true, 1, 1, 1>>(c);  // unused
        // add<K<_128x256x64_2x1, 3, kColMajor, Striding::kFlat, true, 1, 1, 1>>(c);  // unused
    }
    {
        // using _64x128x64_1x2 = Config<Shape<64, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _96x128x64_1x2 = Config<Shape<96, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _128x128x64_1x2 = Config<Shape<128, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _192x128x64_1x2 = Config<Shape<192, 128, 64>, Shape<1, 2>, Registers<104, 200>>;
        // using _224x128x64_1x2 = Config<Shape<224, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        using _256x128x64_1x2 = Config<Shape<256, 128, 64>, Shape<1, 2>, Registers<128, 184>>;
        // using _32x256x64_2x1 = Config<Shape<32, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _64x256x64_2x1 = Config<Shape<64, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _96x256x64_2x1 = Config<Shape<96, 256, 64>, Shape<2, 1>, Registers<120, 192>>;
        // using _128x256x64_2x1 = Config<Shape<128, 256, 64>, Shape<2, 1>, Registers<120, 192>>;

        // add<K<_64x128x64_1x2, 4, kColMajor, Striding::kIndexed, false, 1, 1, 1>>(c);  // unused
        // add<K<_96x128x64_1x2, 4, kColMajor, Striding::kIndexed, false, 1, 1, 1>>(c);  // unused
        // add<K<_128x128x64_1x2, 4, kColMajor, Striding::kIndexed, false, 1, 1, 1>>(c);  // unused
        // add<K<_192x128x64_1x2, 4, kColMajor, Striding::kIndexed, false, 1, 1, 1>>(c);  // unused
        // add<K<_224x128x64_1x2, 4, kColMajor, Striding::kIndexed, false, 1, 1, 1>>(c);  // unused
        add<K<_256x128x64_1x2, 4, kColMajor, Striding::kIndexed, false, 1, 1, 1>>(c);  // refs: 1543
        // add<K<_32x256x64_2x1, 3, kColMajor, Striding::kIndexed, true, 1, 1, 1>>(c);  // unused
        // add<K<_64x256x64_2x1, 3, kColMajor, Striding::kIndexed, true, 1, 1, 1>>(c);  // unused
        // add<K<_96x256x64_2x1, 3, kColMajor, Striding::kIndexed, true, 1, 1, 1>>(c);  // unused
        // add<K<_128x256x64_2x1, 3, kColMajor, Striding::kIndexed, true, 1, 1, 1>>(c);  // unused
    }
    {
        // using _96x128x64_1x2 = Config<Shape<96, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _192x128x64_1x2 = Config<Shape<192, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _224x128x64_1x2 = Config<Shape<224, 128, 64>, Shape<1, 2>, Registers<120, 192>>;
        // using _256x128x64_1x2 = Config<Shape<256, 128, 64>, Shape<1, 2>, Registers<120, 192>>;

        // add<K<_96x128x64_1x2, 4, kColMajor, Striding::kBlocked, false, 1, 1, 1>>(c);  // unused
        // add<K<_192x128x64_1x2, 4, kColMajor, Striding::kBlocked, false, 1, 1, 1>>(c);  // unused
        // add<K<_224x128x64_1x2, 4, kColMajor, Striding::kBlocked, false, 1, 1, 1>>(c);  // unused
        // add<K<_256x128x64_1x2, 4, kColMajor, Striding::kBlocked, false, 1, 1, 1>>(c);  // unused
    }
}

Registrar reg(bf16, register_kernels<C::Type>);
}  // namespace

}  // namespace turbomind::gemm
