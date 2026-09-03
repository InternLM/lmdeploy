// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda.h>
#include <numeric>

#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel/u4.h"
#include "src/turbomind/kernels/gemm/sm90_mixed_pack.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/utils/memory_utils.h"

#if TM_GEMM_HAS_SM90_MIXED

#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_4_config.h"
#include "src/turbomind/kernels/gemm/kernel/sm90_64n16_mixed_reg.h"

namespace turbomind::gemm {
namespace {
std::optional<WeightBridge> supports(const DataFormat& format, bool grouped)
{
    if (format.block_sizes.size() != 2) {
        return std::nullopt;
    }
    return supports_u4<32, kBfloat16>(format, grouped);
    // return format.block_sizes[0] % 128 == 0 ? supports_u4<128, kBfloat16>(format, grouped) :
    //                                           supports_u4<32, kBfloat16>(format, grouped);
}

template<int GroupSize>
void pack(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    TM_CHECK_EQ(linear.weight_format.block_sizes.size(), 2);

    ApplyWeightBridge(linear, bridge, stream);
    TM_CHECK_EQ(linear.output_dim % kSm90MixedFragmentN, 0);
    TM_CHECK_EQ(linear.input_dim % std::lcm(kSm90MixedTileK, GroupSize), 0);
    TM_CHECK_GE(linear.input_dim, 128);
    PackWeight(linear, kSm90MixedWeightPack, PackSm90U4Weight, stream);

    TM_CHECK_EQ(linear.scales.dtype(), kBfloat16);
    TM_CHECK(!linear.zeros || linear.zeros.dtype() == kBfloat16);
    Tensor scales = std::move(linear.scales);
    Tensor zeros  = std::move(linear.zeros);
    Tensor packed_q{{scales.size() / kSm90MixedFragmentN * kSm90U4QparamValuesFragment}, kUint8, kDEVICE};
    PackSm90U4QParams(static_cast<uint8_t*>(packed_q.raw_data()),
                      scales.data<bfloat16_t>(),
                      zeros ? zeros.data<bfloat16_t>() : nullptr,
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
    linear.weight_format = DataFormat{kUint4, {GroupSize, 1}, kBfloat16, kUint4};
}
}  // namespace

const Family Sm90U4Family{
    29, 250, kBfloat16, kBfloat16, 64, 128, 128, 1, true, true, supports, pack<32>, 64, kBfloat16};

namespace {

using Format = Sm90U4Format<128>;
using detail::add;

Registrar reg(Sm90U4Family, [](Collector& c) {
    using G32 = Sm90U4Format<32>;

    ////////////////////////////////// flat //////////////////////////////////
    add<G32, kColMajor, Striding::kFlat, Tile_8x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kFlat, Tile_16x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kFlat, Tile_32x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kFlat, Tile_64x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kFlat, Tile_96x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kFlat, Tile_128x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kFlat, Tile_192x128_S4_1x2>(c);
    add<G32, kColMajor, Striding::kFlat, Tile_224x128_S4_1x2>(c);
    add<G32, kColMajor, Striding::kFlat, Tile_256x128_S4_1x2>(c);
    add<G32, kColMajor, Striding::kFlat, Tile_384x128_S3_1x2>(c);

    ////////////////////////////////// blocked //////////////////////////////////
    add<G32, kColMajor, Striding::kBlocked, Tile_8x128_S4_1x2>(c);
    add<G32, kColMajor, Striding::kBlocked, Tile_16x128_S4_1x2>(c);
    add<G32, kColMajor, Striding::kBlocked, Tile_32x128_S4_1x2>(c);
    add<G32, kColMajor, Striding::kBlocked, Tile_64x128_S4_1x2>(c);
    add<G32, kColMajor, Striding::kBlocked, Tile_96x128_S4_1x2>(c);
    add<G32, kColMajor, Striding::kBlocked, Tile_128x128_S4_1x2>(c);
    add<G32, kColMajor, Striding::kBlocked, Tile_192x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kBlocked, Tile_224x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kBlocked, Tile_256x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kBlocked, Tile_384x128_S3_1x2, true>(c);

    ////////////////////////////////// indexed //////////////////////////////////
    add<G32, kColMajor, Striding::kIndexed, Tile_8x128_S4_1x1, true>(c);
    add<G32, kColMajor, Striding::kIndexed, Tile_16x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kIndexed, Tile_32x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kIndexed, Tile_64x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kIndexed, Tile_96x128_S4_1x2, true>(c);
    add<G32, kColMajor, Striding::kIndexed, Tile_192x128_S3_1x2, true>(c);
    add<G32, kColMajor, Striding::kIndexed, Tile_8x256_S3_1x2, true>(c);
    add<G32, kColMajor, Striding::kIndexed, Tile_16x256_S3_1x2, true>(c);
    add<G32, kColMajor, Striding::kIndexed, Tile_32x256_S3_1x2, true>(c);
    add<G32, kColMajor, Striding::kIndexed, Tile_64x256_S3_1x2, true>(c);
    add<G32, kColMajor, Striding::kIndexed, Tile_96x256_S3_1x2, true>(c);
    add<G32, kColMajor, Striding::kIndexed, Tile_128x256_S3_1x2, true>(c);
});

}  // namespace
}  // namespace turbomind::gemm

#endif
