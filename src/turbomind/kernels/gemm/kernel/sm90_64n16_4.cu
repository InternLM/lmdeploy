// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>
#include <numeric>

#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel/config.h"
#include "src/turbomind/kernels/gemm/kernel/u4.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/utils/memory_utils.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_mixed_reg.h"

namespace turbomind::gemm {
namespace {
template<int GroupSize, DataType Dtype>
void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    TM_CHECK_EQ(linear.weight_format.block_sizes.size(), 2);

    ApplyWeightBridge(linear, bridge, stream);
    TM_CHECK_EQ(linear.output_dim % kSm90MixedFragmentN, 0);
    TM_CHECK_EQ(linear.input_dim % std::lcm(kSm90MixedTileK, GroupSize), 0);
    TM_CHECK_GE(linear.input_dim, 128);
    PackWeight(linear, kSm90MixedWeightPack, PackSm90U4Weight, stream);

    TM_CHECK_EQ(linear.scales.dtype(), Dtype);
    TM_CHECK(!linear.zeros || linear.zeros.dtype() == Dtype);
    Tensor scales = std::move(linear.scales);
    Tensor zeros  = std::move(linear.zeros);
    Tensor packed_q{{scales.size() / kSm90MixedFragmentN * kSm90U4QparamValuesFragment}, kUint8, kDEVICE};
    PackSm90U4QParams(static_cast<uint8_t*>(packed_q.raw_data()),
                      scales.data<data_type_t<Dtype>>(),
                      zeros ? zeros.data<data_type_t<Dtype>>() : nullptr,
                      linear.output_dim,
                      linear.input_dim / GroupSize,
                      stream);
    linear.scales        = std::move(packed_q);
    linear.zeros         = {};
    linear.q_desc        = transpose(MatrixLayout{kUint8,
                                                  kColMajor,
                                                  linear.output_dim,
                                                  linear.input_dim / GroupSize,
                                                  linear.output_dim / kSm90MixedFragmentN
                                                      * kSm90U4QparamValuesFragment,
                                                  kSm90MixedQParamPack,
                                                  0,
                                                  nullptr,
                                                  nullptr});
    linear.weight_format = DataFormat{kUint4, {GroupSize, 1}, Dtype, kUint4};
}

const Family bf16{29, 250, kBfloat16, kBfloat16, 64, 128, 128, 1, true, true, supports_u4<32, kBfloat16>, pack<32, kBfloat16>, 64, kBfloat16};
const Family f16{35, 250, kHalf, kHalf, 64, 128, 128, 1, true, true, supports_u4<32, kHalf>, pack<32, kHalf>, 64, kHalf};

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
        add<K<_8x128_1x2, 4, kRowMajor, Striding::kFlat, true>>(c);
        add<K<_16x128_1x2, 4, kRowMajor, Striding::kFlat, true>>(c);
        add<K<_32x128_1x2, 4, kRowMajor, Striding::kFlat, true>>(c);
        add<K<_64x128_1x2, 4, kRowMajor, Striding::kFlat, true>>(c);
        add<K<_96x128_1x2, 4, kRowMajor, Striding::kFlat, true>>(c);
        add<K<_128x128_1x2, 4, kRowMajor, Striding::kFlat, true>>(c);
        add<K<_192x128_1x2, 4, kRowMajor, Striding::kFlat>>(c);
        add<K<_224x128_1x2, 4, kRowMajor, Striding::kFlat>>(c);
        add<K<_256x128_1x2, 4, kRowMajor, Striding::kFlat>>(c);
        add<K<_384x128_1x2, 3, kRowMajor, Striding::kFlat, false, 1, 1, 192>>(c);

        ////////////////////////////////// blocked //////////////////////////////////
        add<K<_8x128_1x2, 4, kRowMajor, Striding::kBlocked>>(c);
        add<K<_16x128_1x2, 4, kRowMajor, Striding::kBlocked>>(c);
        add<K<_32x128_1x2, 4, kRowMajor, Striding::kBlocked>>(c);
        add<K<_64x128_1x2, 4, kRowMajor, Striding::kBlocked>>(c);
        add<K<_96x128_1x2, 4, kRowMajor, Striding::kBlocked>>(c);
        add<K<_128x128_1x2, 4, kRowMajor, Striding::kBlocked>>(c);
        add<K<_192x128_1x2, 4, kRowMajor, Striding::kBlocked, true>>(c);
        add<K<_224x128_1x2, 4, kRowMajor, Striding::kBlocked, true>>(c);
        add<K<_256x128_1x2, 4, kRowMajor, Striding::kBlocked, true>>(c);
        add<K<_384x128_1x2, 3, kRowMajor, Striding::kBlocked, true, 1, 1, 192>>(c);

        ////////////////////////////////// indexed //////////////////////////////////
        add<K<_8x128_1x1, 4, kRowMajor, Striding::kIndexed, true>>(c);
        add<K<_16x128_1x2, 4, kRowMajor, Striding::kIndexed, true>>(c);
        add<K<_32x128_1x2, 4, kRowMajor, Striding::kIndexed, true>>(c);
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

        add<K<_64x128_1x2, 4, kRowMajor, Striding::kIndexed, true>>(c);
        add<K<_96x128_1x2, 4, kRowMajor, Striding::kIndexed, true>>(c);
        add<K<_192x128_1x2, 3, kRowMajor, Striding::kIndexed, true>>(c);
        add<K<_8x256_1x2, 3, kRowMajor, Striding::kIndexed, true>>(c);
        add<K<_16x256_1x2, 3, kRowMajor, Striding::kIndexed, true>>(c);
        add<K<_32x256_1x2, 3, kRowMajor, Striding::kIndexed, true>>(c);
        add<K<_64x256_1x2, 3, kRowMajor, Striding::kIndexed, true>>(c);
        add<K<_96x256_1x2, 3, kRowMajor, Striding::kIndexed, true>>(c);
        add<K<_128x256_1x2, 3, kRowMajor, Striding::kIndexed, true>>(c);
    }
}

using BF16 = detail::C<Sm90U4Format<32, kBfloat16>>;
using FP16 = detail::C<Sm90U4Format<32, kHalf>>;

Registrar reg[]{
    {bf16, register_kernels<BF16::Type>},
    {f16, register_kernels<FP16::Type>},
};
}  // namespace
}  // namespace turbomind::gemm

#endif
