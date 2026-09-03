// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/convert.cuh"
#include "src/turbomind/kernels/gemm/family.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind::gemm {

template<DataType Dtype>
std::optional<WeightBridge> supports_fp(const DataFormat& format, bool)
{
    return format == DataFormat{Dtype} ? std::optional{WeightBridge{}} : std::nullopt;
}

template<class Arch, Order WeightOrder, uint32_t WeightPack, DataType Dtype>
void pack_fp(LinearWeight& linear, const WeightBridge& bridge, cudaStream_t stream)
{
    ApplyWeightBridge(linear, bridge, stream);
    PackWeight(linear, GetImpl<Arch, WeightOrder, WeightPack, uint16_t, uint16_t>(), stream);
    linear.q_desc        = {};
    linear.weight_format = DataFormat{Dtype};
}

}  // namespace turbomind::gemm
