// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/family.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

inline OutputSpec fp8_output_spec(core::Layout layout, const DataFormat& format, Epilogue epilogue)
{
    if (!format.scales.present()) {
        return plain_output_spec(std::move(layout), format, epilogue);
    }

    TM_CHECK_EQ(format.dtype, kFloat8_e4m3);
    TM_CHECK_EQ(format.block_sizes.size(), 2);
    TM_CHECK_EQ(format.block_sizes[0], 128);
    TM_CHECK_EQ(format.block_sizes[1], 1);
    TM_CHECK_EQ(format.scales.dtype, kFloat);
    TM_CHECK(!format.zeros.present());

    constexpr int kGroupSize    = 128;
    constexpr int kRowAlignment = 4;

    OutputSpec spec;
    spec.layout = apply_output_epilogue(std::move(layout), epilogue);
    spec.dtype  = kFloat8_e4m3;

    const int           output_dim = spec.layout.shape(-1);
    const core::ssize_t rows       = spec.layout.size() / output_dim;

    spec.scales_layout = core::Layout{{cdiv(output_dim, kGroupSize), rows}, {round_up(rows, static_cast<core::ssize_t>(kRowAlignment)), 1}};
    spec.scales_dtype = kFloat;
    return spec;
}

template<DataType ScaleDtype, int ScaleGroupN>
std::optional<WeightBridge> supports_e4m3(const DataFormat& format, bool)
{
    static_assert(ScaleGroupN == 1 || ScaleGroupN == 128);
    if (format.dtype != kFloat8_e4m3) {
        return std::nullopt;
    }
    if (format.block_sizes.size() != 2) {
        return std::nullopt;
    }

    const bool groupwise = format.block_sizes[0] == 128 && format.block_sizes[1] == 1;
    const bool blockwise = format.block_sizes[0] == 128 && format.block_sizes[1] == 128;
    if (!groupwise && !blockwise) {
        return std::nullopt;
    }
    if (!IsFloatFormatType(format.scales.dtype)) {
        return std::nullopt;
    }
    if (format.zeros.present()) {
        return std::nullopt;
    }

    if (groupwise && ScaleGroupN == 128) {
        return std::nullopt;
    }

    WeightBridge bridge;
    bridge.replicate_scales.y = blockwise && ScaleGroupN == 1 ? 128 : 1;
    bridge.convert_scales     = ScaleDtype;
    return bridge;
}

template<class Arch,
         Order    WeightOrder,
         uint32_t WeightPack,
         Order    QParamOrder,
         uint32_t QParamPack,
         DataType ScaleDtype>
void pack_e4m3(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    PackWeight(linear, GetImpl<Arch, WeightOrder, WeightPack, uint16_t, uint8_t>(), stream);
    PackQParams(linear,
                GetImpl<Arch, QParamOrder, QParamPack, uint16_t, uint16_t>(),
                QuantDesc{QuantType::kK, 128},
                stream);
    linear.weight_format = DataFormat{kFloat8_e4m3, {128, 1}, ScaleDtype};
}

}  // namespace turbomind::gemm
