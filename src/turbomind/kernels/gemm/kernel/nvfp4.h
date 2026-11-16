// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/family.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

// NVFP4 for SM75: E2M1 codes (packed B operand) with a per-group (16) half
// scale. The per-tensor global scale is folded into the scales on the Python
// side, so the kernel sees a plain half scale and reuses the SM75 E2M1 W4A16
// path.
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
    if (format.scales.dtype != kHalf) {
        return std::nullopt;
    }
    if (format.zeros.present()) {
        return std::nullopt;
    }
    return WeightBridge{};
}

template<class Arch, Order WeightOrder, uint32_t WeightPack, Order QParamOrder, uint32_t QParamPack>
void pack_nvfp4(LinearWeight& linear, cudaStream_t stream)
{
    // Dense layout: E2M1 weights are the packed B operand, half scales are V.
    PackWeight(linear, GetImpl<Arch, WeightOrder, WeightPack, uint16_t, uint4_t>(), stream);
    PackQParams(
        linear, GetImpl<Arch, QParamOrder, QParamPack, uint16_t, uint16_t>(), QuantDesc{QuantType::kK, 16}, stream);
    linear.weight_format = DataFormat{kFloat4_e2m1, {16, 1}, kHalf};
}

}  // namespace turbomind::gemm
