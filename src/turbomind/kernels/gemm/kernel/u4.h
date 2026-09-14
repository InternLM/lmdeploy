// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/family.h"

namespace turbomind::gemm {

template<int GroupSize, DataType QparamDtype>
std::optional<WeightBridge> supports_u4(const DataFormat& format, bool)
{
    if (format.dtype != kUint4) {
        return std::nullopt;
    }
    if (format.block_sizes.size() != 2) {
        return std::nullopt;
    }
    if (format.block_sizes[1] != 1) {
        return std::nullopt;
    }
    if (format.block_sizes[0] % GroupSize != 0) {
        return std::nullopt;
    }
    if (!IsFloatFormatType(format.scales.dtype)) {
        return std::nullopt;
    }
    if (!IsFloatFormatType(format.zeros.dtype)) {
        return std::nullopt;
    }
    WeightBridge bridge;
    bridge.replicate_scales.x = format.block_sizes[0] / GroupSize;
    bridge.convert_scales     = QparamDtype;
    bridge.convert_zeros      = QparamDtype;
    return bridge;
}

}  // namespace turbomind::gemm
