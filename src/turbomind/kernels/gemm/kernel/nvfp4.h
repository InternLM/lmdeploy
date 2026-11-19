// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/family.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

// NVFP4: E2M1 codes (packed B operand) with per-group (16) scales, consumed
// natively on every arch: the checkpoint's raw layout -- e4m3 block scales
// plus a separate per-tensor global scale. The bridge widens the scales to
// half (every e4m3 value is exactly representable in fp16) and the global
// scale rides in LinearWeight to the GEMM epilogue, which multiplies the
// output by it.
inline std::optional<WeightBridge> supports_nvfp4(const DataFormat& format, bool)
{
    if (format.dtype != kFloat4_e2m1) {
        return std::nullopt;
    }
    if (format.block_sizes.size() != 2) {
        return std::nullopt;
    }
    if (format.block_sizes[0] != 16) {
        return std::nullopt;
    }
    if (format.block_sizes[1] != 1) {
        return std::nullopt;
    }
    if (format.zeros.present()) {
        return std::nullopt;
    }
    if (format.scales.dtype != kFloat8_e4m3) {
        return std::nullopt;
    }
    WeightBridge bridge;
    bridge.convert_scales = kHalf;
    return bridge;
}

template<class Arch, Order WeightOrder, uint32_t WeightPack, Order QParamOrder, uint32_t QParamPack>
void pack_nvfp4(LinearWeight& linear, cudaStream_t stream)
{
    // Dense layout: E2M1 weights are the packed B operand, half scales are V.
    // Both modes arrive here with half scales: folded (SM75) or exact-widened
    // raw e4m3 (SM80+). `global_scale`, when present, is untouched by packing
    // and consumed by the epilogue.
    PackWeight(linear, GetImpl<Arch, WeightOrder, WeightPack, uint16_t, uint4_t>(), stream);
    PackQParams(
        linear, GetImpl<Arch, QParamOrder, QParamPack, uint16_t, uint16_t>(), QuantDesc{QuantType::kK, 16}, stream);
    linear.weight_format = DataFormat{kFloat4_e2m1, {16, 1}, kHalf};
}

}  // namespace turbomind::gemm
