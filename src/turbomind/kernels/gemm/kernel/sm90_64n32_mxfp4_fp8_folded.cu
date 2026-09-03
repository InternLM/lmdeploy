// Copyright (c) OpenMMLab. All rights reserved.

#include <cstdlib>
#include <cuda.h>

#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel/e4m3.h"
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

template<Order    Raster,
         int      MulticastA,
         int      MulticastB,
         bool     Grouped,
         Striding StridingA,
         int      TileM,
         int      TileN,
         int      Stages,
         class WGLayout,
         int  MmaN,
         int  ProducerRegsTma,
         int  MathRegsTma,
         int  ProducerRegsIndexed,
         int  MathRegsIndexed,
         int  EpilogueTileM,
         int  EpilogueTileN,
         int  EpilogueStages,
         bool SupportsFusedSilu>
void add(Collector& c)
{
    c.add<KernelImplSm90MxFp4Fp8<GemmUniversalSm90MxFp4Fp8Folded<Raster,
                                                                 MulticastA,
                                                                 MulticastB,
                                                                 Grouped,
                                                                 StridingA,
                                                                 TileM,
                                                                 TileN,
                                                                 Stages,
                                                                 WGLayout,
                                                                 MmaN,
                                                                 ProducerRegsTma,
                                                                 MathRegsTma,
                                                                 ProducerRegsIndexed,
                                                                 MathRegsIndexed,
                                                                 EpilogueTileM,
                                                                 EpilogueTileN,
                                                                 EpilogueStages,
                                                                 SupportsFusedSilu>>>();
}

Registrar reg(folded, [](Collector& c) {
    add<kRowMajor, 1, 1, false, Striding::kFlat, 8, 128, 4, WG_1x2, 8, 40, 232, 72, 216, 8, 128, 1, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 16, 128, 4, WG_1x2, 16, 40, 232, 72, 216, 16, 128, 1, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 32, 128, 4, WG_1x2, 32, 40, 232, 72, 216, 32, 128, 1, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 64, 128, 4, WG_1x2, 64, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 96, 128, 4, WG_1x2, 96, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 128, 128, 4, WG_1x2, 128, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 192, 128, 4, WG_1x2, 96, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 256, 128, 4, WG_1x2, 128, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 2, 1, false, Striding::kFlat, 256, 128, 4, WG_1x2, 128, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 2, false, Striding::kFlat, 256, 128, 4, WG_1x2, 128, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 8, 256, 3, WG_1x2, 8, 40, 232, 72, 216, 8, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 16, 256, 3, WG_1x2, 16, 40, 232, 72, 216, 16, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 32, 256, 3, WG_1x2, 32, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 64, 256, 3, WG_1x2, 64, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 96, 256, 3, WG_1x2, 96, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, true, Striding::kBlocked, 128, 128, 3, WG_1x2, 128, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, true, Striding::kIndexed, 64, 128, 3, WG_1x2, 64, 40, 232, 72, 216, 32, 128, 2, false>(c);
    add<kRowMajor, 1, 1, false, Striding::kFlat, 64, 256, 2, WG_1x2, 64, 40, 232, 72, 216, 32, 128, 2, true>(c);
    add<kRowMajor, 1, 1, true, Striding::kIndexed, 64, 256, 2, WG_1x2, 64, 40, 232, 72, 216, 32, 128, 2, true>(c);
});

}  // namespace
}  // namespace turbomind::gemm

#endif
