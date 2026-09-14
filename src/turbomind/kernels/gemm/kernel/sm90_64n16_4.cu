// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>

#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel/geometry.h"
#include "src/turbomind/kernels/gemm/kernel/u4.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/utils/memory_utils.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_mixed_reg.h"

namespace turbomind::gemm {
namespace {
using config::Shape;
using namespace config::geometry;

template<int GroupSize, DataType Dtype>
void pack(LinearWeight& linear, cudaStream_t stream)
{
    TM_CHECK_EQ(linear.weight_format.block_sizes.size(), 2);
    TM_CHECK_EQ(linear.output_dim % kSm90MixedFragmentN, 0);
    TM_CHECK_EQ(linear.input_dim % kSm90MixedTileK, 0);
    TM_CHECK_GE(linear.input_dim, 128);
    PackWeight(linear, kSm90MixedWeightPack, PackSm90U4Weight, stream);

    TM_CHECK_EQ(linear.scales.dtype(), Dtype);
    TM_CHECK(!linear.zeros || linear.zeros.dtype() == Dtype);
    // The weight bridge must have expanded the qparams to at least one row per
    // K group and at least one column per output column; extra rows or a wider
    // row are tolerated since only the leading group_count rows and output_dim
    // columns are read.
    const int group_count   = (linear.input_dim + GroupSize - 1) / GroupSize;
    const int scales_stride = linear.scales.shape(1);
    TM_CHECK_GE(linear.scales.shape(0), group_count);
    TM_CHECK_GE(scales_stride, linear.output_dim);
    if (linear.zeros) {
        TM_CHECK_GE(linear.zeros.shape(0), group_count);
        TM_CHECK_GE(linear.zeros.shape(1), linear.output_dim);
    }
    Tensor scales = std::move(linear.scales);
    Tensor zeros  = std::move(linear.zeros);
    Tensor packed_q{{scales.size() / kSm90MixedFragmentN * kSm90U4QparamValuesFragment}, kUint8, kDEVICE};
    PackSm90U4QParams(static_cast<uint8_t*>(packed_q.raw_data()),
                      scales.data<data_type_t<Dtype>>(),
                      zeros ? zeros.data<data_type_t<Dtype>>() : nullptr,
                      linear.output_dim,
                      group_count,
                      scales_stride,
                      stream);
    linear.scales        = std::move(packed_q);
    linear.zeros         = {};
    linear.q_desc        = transpose(MatrixLayout{kUint8,
                                           kColMajor,
                                           linear.output_dim,
                                           group_count,
                                           linear.output_dim / kSm90MixedFragmentN * kSm90U4QparamValuesFragment,
                                           kSm90MixedQParamPack,
                                           0,
                                           nullptr,
                                           nullptr});
    linear.weight_format = DataFormat{kUint4, {GroupSize, 1}, Dtype, kUint4};
}

const Family bf16{29,
                  250,
                  kBfloat16,
                  kBfloat16,
                  64,
                  64,
                  128,
                  1,
                  true,
                  true,
                  supports_u4<32, kBfloat16>,
                  pack<32, kBfloat16>,
                  64,
                  kBfloat16};
const Family f16{35, 250, kHalf, kHalf, 64, 64, 128, 1, true, true, supports_u4<32, kHalf>, pack<32, kHalf>, 64, kHalf};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_,
                  int      Stages,
                  Order    Raster,
                  Striding Mode,
                  bool     Silu         = false,
                  class ClusterShape    = Shape<1, 1>,
                  int  MmaN             = 0,
                  bool SeparateMmaAtoms = false,
                  int  EpiM             = 0,
                  int  EpiStages        = 0>
         class K>
void register_kernels(Collector& c)
{
    ////////////////////////////////// flat //////////////////////////////////
    add<K<_8x128_1x2<80, 80>, 4, kRowMajor, Striding::kFlat, true>>(c);
    add<K<_16x128_1x2<80, 80>, 4, kRowMajor, Striding::kFlat, true>>(c);
    add<K<_32x128_1x2<80, 80>, 4, kRowMajor, Striding::kFlat, true>>(c);
    add<K<_64x128_1x2<80, 80>, 4, kRowMajor, Striding::kFlat, true>>(c);
    add<K<_96x128_1x2<80, 96>, 4, kRowMajor, Striding::kFlat, true>>(c);
    add<K<_128x128_1x2<80, 112>, 4, kRowMajor, Striding::kFlat, true>>(c);
    add<K<_192x128_1x2<80, 208>, 4, kRowMajor, Striding::kFlat>>(c);
    add<K<_224x128_1x2<80, 208>, 4, kRowMajor, Striding::kFlat>>(c);
    add<K<_256x128_1x2<80, 208>, 4, kRowMajor, Striding::kFlat>>(c);
    add<K<_384x128_1x2<40, 232>, 3, kRowMajor, Striding::kFlat, false, Shape<1, 1>, 192>>(c);

    ////////////////////////////////// blocked //////////////////////////////////
    add<K<_8x128_1x2<80, 80>, 4, kRowMajor, Striding::kBlocked>>(c);
    add<K<_16x128_1x2<80, 80>, 4, kRowMajor, Striding::kBlocked>>(c);
    add<K<_32x128_1x2<80, 80>, 4, kRowMajor, Striding::kBlocked>>(c);
    add<K<_64x128_1x2<80, 80>, 4, kRowMajor, Striding::kBlocked>>(c);
    add<K<_96x128_1x2<80, 96>, 4, kRowMajor, Striding::kBlocked>>(c);
    add<K<_128x128_1x2<80, 112>, 4, kRowMajor, Striding::kBlocked>>(c);
    add<K<_192x128_1x2<80, 208>, 4, kRowMajor, Striding::kBlocked, true>>(c);
    add<K<_224x128_1x2<80, 208>, 4, kRowMajor, Striding::kBlocked, true>>(c);
    add<K<_256x128_1x2<80, 208>, 4, kRowMajor, Striding::kBlocked, true>>(c);
    add<K<_384x128_1x2<40, 232>, 3, kRowMajor, Striding::kBlocked, true, Shape<1, 1>, 192>>(c);

    ////////////////////////////////// indexed //////////////////////////////////
    add<K<_8x128_1x1<120, 128>, 4, kRowMajor, Striding::kIndexed, true>>(c);
    add<K<_16x128_1x2<80, 80>, 4, kRowMajor, Striding::kIndexed, true>>(c);
    add<K<_32x128_1x2<80, 80>, 4, kRowMajor, Striding::kIndexed, true>>(c);

    add<K<_64x128_1x2<120, 192>, 4, kRowMajor, Striding::kIndexed, true>>(c);
    add<K<_96x128_1x2<120, 192>, 4, kRowMajor, Striding::kIndexed, true>>(c);
    add<K<_192x128_1x2<120, 192>, 3, kRowMajor, Striding::kIndexed, true>>(c);
    add<K<_8x256_1x2<80, 80>, 3, kRowMajor, Striding::kIndexed, true>>(c);
    add<K<_16x256_1x2<80, 88>, 3, kRowMajor, Striding::kIndexed, true>>(c);
    add<K<_32x256_1x2<120, 192>, 3, kRowMajor, Striding::kIndexed, true>>(c);
    add<K<_64x256_1x2<120, 192>, 3, kRowMajor, Striding::kIndexed, true>>(c);
    add<K<_96x256_1x2<120, 192>, 3, kRowMajor, Striding::kIndexed, true>>(c);
    add<K<_128x256_1x2<120, 192>, 3, kRowMajor, Striding::kIndexed, true>>(c);
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
