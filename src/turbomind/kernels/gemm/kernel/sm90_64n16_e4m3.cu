// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>

#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel/config.h"
#include "src/turbomind/kernels/gemm/kernel/e4m3.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/models/linear_weight.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_mixed_reg.h"

namespace turbomind::gemm {
namespace {
void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    TM_CHECK_EQ(linear.output_dim % kSm90MixedTileN, 0);
    TM_CHECK_EQ(linear.input_dim % Sm90Fp8E4M3Format::kGroupSize, 0);
    TM_CHECK_GE(linear.input_dim, 128);
    PackWeight(linear, Sm90Fp8E4M3Format::kWeightPack, PackSm90Fp8E4M3Weight, stream);

    TM_CHECK_EQ(linear.scales.dtype(), kFloat);
    TM_CHECK_EQ(linear.scales.ndim(), 2);
    TM_CHECK_EQ(linear.scales.shape(0), linear.input_dim / Sm90Fp8E4M3Format::kGroupSize);
    TM_CHECK_EQ(linear.scales.shape(1), linear.output_dim / Sm90Fp8E4M3Format::kScaleGroupN);
    TM_CHECK(linear.scales.is_contiguous());
    MatrixLayout s_desc{kFloat,
                        kRowMajor,
                        (int)linear.scales.shape(0),
                        (int)linear.scales.shape(1),
                        (int)linear.scales.stride(0),
                        0,
                        0,
                        nullptr,
                        nullptr};
    Tensor       packed_q{{linear.scales.size() * Sm90Fp8E4M3Format::kQparamValuesTile}, kBfloat16, kDEVICE};
    PackSm90Fp8E4M3Scales(static_cast<bfloat16_t*>(packed_q.raw_data()),
                          static_cast<const float*>(linear.scales.raw_data()),
                          s_desc.rows,
                          s_desc.cols,
                          stream);
    linear.scales = std::move(packed_q);
    linear.q_desc = MatrixLayout{kBfloat16,
                                 kRowMajor,
                                 s_desc.rows,
                                 s_desc.cols,
                                 s_desc.cols * Sm90Fp8E4M3Format::kQparamValuesTile,
                                 Sm90Fp8E4M3Format::kQparamPack,
                                 0,
                                 nullptr,
                                 nullptr};
    linear.weight_format =
        DataFormat{kFloat8_e4m3, {Sm90Fp8E4M3Format::kGroupSize, Sm90Fp8E4M3Format::kScaleGroupN}, kBfloat16};
}

const Family e4m3{32,
                  250,
                  kBfloat16,
                  kBfloat16,
                  128,
                  128,
                  128,
                  1,
                  true,
                  true,
                  supports_e4m3<kFloat, Sm90Fp8E4M3Format::kScaleGroupN>,
                  pack,
                  64,
                  kBfloat16};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_, int Stages, Order Raster, Striding Mode, bool Silu = false, int MulticastA = 1, int MulticastB = 1, int MmaN = 0, bool SeparateMmaAtoms = false, int EpiM = 0, int EpiStages = 0> class K>
void register_kernels(Collector& c)
{
    using config::Config;
    using config::Registers;
    using config::Shape;

    {
        using _8x128_1x2 = Config<Shape<8, 128>, Shape<1, 2>, Registers<80, 80>>;
        using _16x128_1x2 = Config<Shape<16, 128>, Shape<1, 2>, Registers<80, 80>>;
        using _32x128_1x2 = Config<Shape<32, 128>, Shape<1, 2>, Registers<80, 80>>;
        using _64x128_1x2 = Config<Shape<64, 128>, Shape<1, 2>, Registers<80, 80>>;
        using _96x128_1x2 = Config<Shape<96, 128>, Shape<1, 2>, Registers<80, 96>>;
        using _128x128_1x2 = Config<Shape<128, 128>, Shape<1, 2>, Registers<80, 112>>;
        using _192x128_1x2 = Config<Shape<192, 128>, Shape<1, 2>, Registers<80, 208>>;
        using _224x128_1x2 = Config<Shape<224, 128>, Shape<1, 2>, Registers<80, 208>>;
        using _256x128_1x2 = Config<Shape<256, 128>, Shape<1, 2>, Registers<80, 208>>;
        using _384x128_1x2 = Config<Shape<384, 128>, Shape<1, 2>, Registers<40, 232>>;
        using _8x128_1x1 = Config<Shape<8, 128>, Shape<1, 1>, Registers<120, 128>>;

        ////////////////////////////////// flat //////////////////////////////////
        add<K<_8x128_1x2, 4, kColMajor, Striding::kFlat, true>>(c);
        add<K<_16x128_1x2, 4, kColMajor, Striding::kFlat, true>>(c);
        add<K<_32x128_1x2, 4, kColMajor, Striding::kFlat, true>>(c);
        add<K<_64x128_1x2, 4, kColMajor, Striding::kFlat, true>>(c);
        add<K<_96x128_1x2, 4, kColMajor, Striding::kFlat, true>>(c);
        add<K<_128x128_1x2, 4, kColMajor, Striding::kFlat, true>>(c);
        add<K<_192x128_1x2, 4, kColMajor, Striding::kFlat>>(c);
        add<K<_224x128_1x2, 4, kColMajor, Striding::kFlat>>(c);
        add<K<_256x128_1x2, 4, kColMajor, Striding::kFlat>>(c);
        add<K<_384x128_1x2, 3, kColMajor, Striding::kFlat, false, 1, 1, 192>>(c);

        ////////////////////////////////// blocked //////////////////////////////////
        add<K<_8x128_1x2, 4, kColMajor, Striding::kBlocked>>(c);
        add<K<_16x128_1x2, 4, kColMajor, Striding::kBlocked>>(c);
        add<K<_32x128_1x2, 4, kColMajor, Striding::kBlocked>>(c);
        add<K<_64x128_1x2, 4, kColMajor, Striding::kBlocked>>(c);
        add<K<_96x128_1x2, 4, kColMajor, Striding::kBlocked>>(c);
        add<K<_128x128_1x2, 4, kColMajor, Striding::kBlocked>>(c);
        add<K<_192x128_1x2, 4, kColMajor, Striding::kBlocked, true>>(c);
        add<K<_224x128_1x2, 4, kColMajor, Striding::kBlocked, true>>(c);
        add<K<_256x128_1x2, 4, kColMajor, Striding::kBlocked, true>>(c);
        add<K<_384x128_1x2, 3, kColMajor, Striding::kBlocked, true, 1, 1, 192>>(c);

        ////////////////////////////////// indexed //////////////////////////////////
        add<K<_8x128_1x1, 4, kColMajor, Striding::kIndexed, true>>(c);
        add<K<_16x128_1x2, 4, kColMajor, Striding::kIndexed, true>>(c);
        add<K<_32x128_1x2, 4, kColMajor, Striding::kIndexed, true>>(c);
    }
    {
        using _64x128_1x2 = Config<Shape<64, 128>, Shape<1, 2>, Registers<120, 192>>;
        using _96x128_1x2 = Config<Shape<96, 128>, Shape<1, 2>, Registers<120, 192>>;
        using _192x128_1x2 = Config<Shape<192, 128>, Shape<1, 2>, Registers<120, 192>>;
        using _8x256_1x2 = Config<Shape<8, 256>, Shape<1, 2>, Registers<80, 80>>;
        using _16x256_1x2 = Config<Shape<16, 256>, Shape<1, 2>, Registers<80, 88>>;
        using _32x256_1x2 = Config<Shape<32, 256>, Shape<1, 2>, Registers<120, 192>>;
        using _64x256_1x2 = Config<Shape<64, 256>, Shape<1, 2>, Registers<120, 192>>;
        using _96x256_1x2 = Config<Shape<96, 256>, Shape<1, 2>, Registers<120, 192>>;
        using _128x256_1x2 = Config<Shape<128, 256>, Shape<1, 2>, Registers<120, 192>>;

        add<K<_64x128_1x2, 4, kColMajor, Striding::kIndexed, true>>(c);
        add<K<_96x128_1x2, 4, kColMajor, Striding::kIndexed, true>>(c);
        add<K<_192x128_1x2, 3, kColMajor, Striding::kIndexed, true>>(c);
        add<K<_8x256_1x2, 3, kColMajor, Striding::kIndexed, true>>(c);
        add<K<_16x256_1x2, 3, kColMajor, Striding::kIndexed, true>>(c);
        add<K<_32x256_1x2, 3, kColMajor, Striding::kIndexed, true>>(c);
        add<K<_64x256_1x2, 3, kColMajor, Striding::kIndexed, true>>(c);
        add<K<_96x256_1x2, 3, kColMajor, Striding::kIndexed, true>>(c);
        add<K<_128x256_1x2, 3, kColMajor, Striding::kIndexed, true>>(c);
    }
}

using C = detail::C<Sm90Fp8E4M3Format>;

Registrar reg(e4m3, register_kernels<C::Type>);
}  // namespace
}  // namespace turbomind::gemm

#endif
