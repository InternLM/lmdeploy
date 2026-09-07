// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdlib>
#include <cuda.h>

#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel/e4m3.h"
#include "src/turbomind/kernels/gemm/kernel/geometry.h"
#include "src/turbomind/kernels/gemm/kernel/mxfp4.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/gemm_universal_sm90_mxfp4_fp8_folded.h"
#include "src/turbomind/kernels/gemm/kernel_impl_sm90_mxfp4_fp8.h"
#include "src/turbomind/kernels/gemm/registrar.h"

namespace turbomind::gemm {
namespace {
using config::Shape;
using namespace config::geometry;

void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    TM_CHECK_EQ(linear.output_dim % kSm90MixedFragmentN, 0);
    TM_CHECK_EQ(linear.input_dim % 128, 0);
    TM_CHECK_GE(linear.input_dim, 128);
    PackWeight(linear, Sm90MxFp4Fp8FoldedFormat::kWeightPack, PackSm90MxFp4Fp8FoldedWeight, stream);

    TM_CHECK_EQ(linear.scales.dtype(), kUint8);
    constexpr int values_per_record = 4 * kSm90MixedFragmentN;
    TM_CHECK_EQ(linear.scales.size() % values_per_record, 0);
    Tensor tmp_q = empty_like(linear.scales);
    TM_CUDA_CHECK(cudaMemcpyAsync(
        tmp_q.raw_data(), linear.scales.raw_data(), linear.scales.byte_size(), cudaMemcpyDefault, stream));
    Tensor packed_q{
        {linear.scales.size() / values_per_record * Sm90MxFp4Fp8FoldedFormat::kQparamValuesFragment}, kUint8, kDEVICE};
    Sm90MxFp4Fp8FoldedPackStats* stats{};
    TM_CUDA_CHECK(cudaMallocAsync(&stats, sizeof(*stats), stream));
    TM_CUDA_CHECK(cudaMemsetAsync(stats, 0, sizeof(*stats), stream));
    PackSm90MxFp4Fp8FoldedQParams(static_cast<uint8_t*>(packed_q.raw_data()),
                                  static_cast<const uint8_t*>(tmp_q.raw_data()),
                                  linear.output_dim,
                                  linear.input_dim / Sm90MxFp4Fp8FoldedFormat::kGroupSize,
                                  stream,
                                  stats);
    Sm90MxFp4Fp8FoldedPackStats host{};
    TM_CUDA_CHECK(cudaMemcpyAsync(&host, stats, sizeof(host), cudaMemcpyDeviceToHost, stream));
    TM_CUDA_CHECK(cudaStreamSynchronize(stream));
    TM_CHECK_GT(host.total_records, 0ull);
    TM_CHECK_EQ(host.foldable_records, host.total_records);
    const char* stats_env = std::getenv("TM_GEMM_MXFP4_FOLD_STATS");
    if (stats_env && stats_env[0] == '1' && stats_env[1] == '\0') {
        TM_LOG_INFO("SM90 MXFP4 foldable records: {}/{} ({:.2f}%)",
                    host.foldable_records,
                    host.total_records,
                    100. * host.foldable_records / host.total_records);
    }
    TM_CUDA_CHECK(cudaFreeAsync(stats, stream));
    linear.scales        = std::move(packed_q);
    linear.q_desc        = transpose(MatrixLayout{kUint8,
                                           kColMajor,
                                           linear.output_dim,
                                           linear.input_dim / Sm90MxFp4Fp8FoldedFormat::kGroupSize,
                                           linear.output_dim,
                                           Sm90MxFp4Fp8FoldedFormat::kQparamPack,
                                           0,
                                           nullptr,
                                           nullptr});
    linear.weight_format = DataFormat{kFloat4_e2m1, {Sm90MxFp4Fp8FoldedFormat::kGroupSize, 1}, kUint8};
}

const Family folded{33,
                    300,
                    DataFormat{kFloat8_e4m3, {128, 1}, kFloat},
                    kBfloat16,
                    128,
                    64,
                    256,
                    1,
                    true,
                    true,
                    supports_mxfp4,
                    pack,
                    128,
                    DataFormat{kFloat8_e4m3, {128, 1}, kFloat},
                    true,
                    fp8_output_spec};

struct C {
    template<class Config_,
             int      Stages,
             Order    Raster,
             Striding Mode,
             bool     Silu      = false,
             class ClusterShape = Shape<1, 1>,
             int MmaN           = Config_::Tile::M / Config_::Groups::M,
             int EpiStages      = 2>
    using Type = KernelImplSm90MxFp4Fp8<
        GemmUniversalSm90MxFp4Fp8Folded<Config_, Stages, Raster, Mode, Silu, ClusterShape, MmaN, EpiStages>>;
};

// NVCC requires defaults on the template-template parameter.
template<template<class Config_,
                  int      Stages,
                  Order    Raster,
                  Striding Mode,
                  bool     Silu      = false,
                  class ClusterShape = Shape<1, 1>,
                  int MmaN           = Config_::Tile::M / Config_::Groups::M,
                  int EpiStages      = 2>
         class K>
void register_kernels(Collector& c)
{
    add<K<_8x128_1x2<40, 232>, 4, kRowMajor, Striding::kFlat, false, Shape<1, 1>, 8, 1>>(c);
    add<K<_16x128_1x2<40, 232>, 4, kRowMajor, Striding::kFlat, false, Shape<1, 1>, 16, 1>>(c);
    add<K<_32x128_1x2<40, 232>, 4, kRowMajor, Striding::kFlat, false, Shape<1, 1>, 32, 1>>(c);
    add<K<_64x128_1x2<40, 232>, 4, kRowMajor, Striding::kFlat>>(c);
    add<K<_96x128_1x2<40, 232>, 4, kRowMajor, Striding::kFlat>>(c);
    add<K<_128x128_1x2<40, 232>, 4, kRowMajor, Striding::kFlat>>(c);
    add<K<_192x128_1x2<40, 232>, 4, kRowMajor, Striding::kFlat, false, Shape<1, 1>, 96>>(c);
    add<K<_256x128_1x2<40, 232>, 4, kRowMajor, Striding::kFlat, false, Shape<1, 1>, 128>>(c);
    add<K<_256x128_1x2<40, 232>, 4, kRowMajor, Striding::kFlat, false, Shape<1, 2>, 128>>(c);
    add<K<_256x128_1x2<40, 232>, 4, kRowMajor, Striding::kFlat, false, Shape<2, 1>, 128>>(c);
    add<K<_8x256_1x2<40, 232>, 3, kRowMajor, Striding::kFlat>>(c);
    add<K<_16x256_1x2<40, 232>, 3, kRowMajor, Striding::kFlat>>(c);
    add<K<_32x256_1x2<40, 232>, 3, kRowMajor, Striding::kFlat>>(c);
    add<K<_64x256_1x2<40, 232>, 3, kRowMajor, Striding::kFlat>>(c);
    add<K<_96x256_1x2<40, 232>, 3, kRowMajor, Striding::kFlat>>(c);
    add<K<_128x128_1x2<40, 232>, 3, kRowMajor, Striding::kBlocked>>(c);

    add<K<_64x128_1x2<72, 216>, 3, kRowMajor, Striding::kIndexed>>(c);
    add<K<_64x256_1x2<40, 232>, 2, kRowMajor, Striding::kFlat, true>>(c);

    add<K<_64x256_1x2<72, 216>, 2, kRowMajor, Striding::kIndexed, true>>(c);
}

Registrar reg(folded, register_kernels<C::Type>);
}  // namespace
}  // namespace turbomind::gemm

#endif
