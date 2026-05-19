// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/data_type.h"
#include <vector>

namespace turbomind {

/// True for trivial (non-quantized) float dtypes: FP32, FP16, BF16.
inline bool IsTrivialFloatType(DataType t) noexcept
{
    return t == kFloat || t == kHalf || t == kBfloat16;
}

/// Descriptor for a single quantization parameter (scales or zeros).
struct QuantParamDesc {
    DataType dtype{};       // kNull means "not present"
    bool     transposed{};  // stored transposed w.r.t. data tensor

    bool present() const noexcept
    {
        return dtype != kNull;
    }
};

/// Universal descriptor for the storage format of a (possibly quantized) tensor.
struct DataFormat {
    DataType         dtype{};      // element type of the data tensor
    std::vector<int> block_sizes;  // per-dimension block sizes (1 = no quantization)
    QuantParamDesc   scales{};
    QuantParamDesc   zeros{};

    /// True if any quantization parameter is present or any block_size > 1.
    bool is_quantized() const noexcept;

    /// Number of dimensions described by this format.
    int rank() const noexcept
    {
        return static_cast<int>(block_sizes.size());
    }
};

/// Construct the DataFormat for a linear weight tensor in TM [in, out] layout.
/// block_sizes stored in tensor-shape order: {block_in, block_out}, so
/// block_sizes[0] is the K-axis group size and block_sizes[1] is the N-axis.
/// Scales / zeros dtypes are derived from (data_type, weight_dtype) per the
/// format's GEMM convention. Validates that the combination is supported.
DataFormat ResolveLinearWeightFormat(DataType data_type, DataType weight_dtype, int block_in, int block_out);

}  // namespace turbomind
