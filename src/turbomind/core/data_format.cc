// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/data_format.h"
#include "src/turbomind/core/check.h"

namespace turbomind {

bool DataFormat::is_quantized() const noexcept
{
    if (scales.present() || zeros.present()) {
        return true;
    }
    for (int bs : block_sizes) {
        if (bs > 1) {
            return true;
        }
    }
    return false;
}

DataFormat ResolveLinearWeightFormat(DataType data_type, DataType weight_dtype, int block_in, int block_out)
{
    DataFormat fmt;
    fmt.dtype = weight_dtype;

    if (IsTrivialFloatType(weight_dtype)) {
        TM_CHECK(block_in == 1 && block_out == 1)
            << "Trivial float weight requires block_in==1 and block_out==1, got " << block_in << ", " << block_out;
        fmt.block_sizes = {1, 1};
        return fmt;
    }

    if (weight_dtype == kFloat8_e4m3) {
        TM_CHECK(block_in == 128 && block_out == 128)
            << "FP8 weight format requires block_in==128 and block_out==128, got " << block_in << ", " << block_out;
        fmt.block_sizes  = {128, 128};
        fmt.scales.dtype = kFloat;
        return fmt;
    }

    if (weight_dtype == kFloat4_e2m1) {
        TM_CHECK(block_in > 0 && block_out == 1)
            << "FP4 weight format requires block_in>0 and block_out==1, got " << block_in << ", " << block_out;
        fmt.block_sizes  = {block_in, 1};
        fmt.scales.dtype = kUint8;
        return fmt;
    }

    const bool is_qweight = weight_dtype == kUint4 || weight_dtype == kUint8;
    if (is_qweight) {
        TM_CHECK(block_in > 0 && block_in <= 256 && block_out == 1)
            << "Quantized integer weight requires 0 < block_in <= 256 and block_out==1, got " << block_in << ", "
            << block_out;
        fmt.block_sizes  = {block_in, 1};
        fmt.scales.dtype = data_type;
        fmt.zeros.dtype  = data_type;
        return fmt;
    }

    TM_CHECK(0) << "Unsupported weight format: " << to_string(weight_dtype);
    return fmt;
}

}  // namespace turbomind
