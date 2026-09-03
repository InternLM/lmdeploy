// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/family.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

inline std::optional<WeightBridge> supports_mxfp4(const DataFormat& format, bool)
{
    if (format.dtype != kFloat4_e2m1) {
        return std::nullopt;
    }
    if (format.block_sizes.size() != 2) {
        return std::nullopt;
    }
    if (format.block_sizes[0] != 32) {
        return std::nullopt;
    }
    if (format.block_sizes[1] != 1) {
        return std::nullopt;
    }
    if (format.scales.dtype != kUint8) {
        return std::nullopt;
    }
    if (format.zeros.present()) {
        return std::nullopt;
    }
    return WeightBridge{};
}

template<class Arch, Order WeightOrder, uint32_t WeightPack, Order QParamOrder, uint32_t QParamPack>
void pack_mxfp4(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    PackWeight(linear, GetImpl<Arch, WeightOrder, WeightPack, uint16_t, uint4_t>(), stream);
    PackQParams(
        linear, GetImpl<Arch, QParamOrder, QParamPack, uint8_t, uint8_t>(), QuantDesc{QuantType::kK, 32}, stream);
    linear.weight_format = DataFormat{kFloat4_e2m1, {32, 1}, kUint8};
}

}  // namespace turbomind::gemm
