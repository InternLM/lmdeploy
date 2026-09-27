// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/data_type.h"
#include <utility>
#include <vector>

namespace turbomind {

/// True for trivial (non-quantized) float dtypes: FP32, FP16, BF16.
inline bool IsTrivialFloatType(DataType t) noexcept
{
    return t == kFloat || t == kHalf || t == kBfloat16;
}

/// True for concrete trivial floats and the metadata-only generic-float matcher.
inline bool IsFloatFormatType(DataType t) noexcept
{
    return t == kGenericFloat || IsTrivialFloatType(t);
}

/// Descriptor for a single quantization parameter (scales or zeros).
struct QuantParamDesc {
    DataType dtype{};  // kNull means "not present"

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

    DataFormat() = default;

    DataFormat(DataType dtype): dtype{dtype}, block_sizes{1, 1} {}

    DataFormat(DataType         dtype,
               std::vector<int> block_sizes,
               DataType         scales_dtype = kNull,
               DataType         zeros_dtype  = kNull):
        dtype{dtype}, block_sizes{std::move(block_sizes)}, scales{scales_dtype}, zeros{zeros_dtype}
    {
    }

    /// True if any quantization parameter is present or any block_size > 1.
    bool is_quantized() const noexcept
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

    /// Number of dimensions described by this format.
    int rank() const noexcept
    {
        return static_cast<int>(block_sizes.size());
    }
};

inline bool operator==(const DataFormat& a, const DataFormat& b) noexcept
{
    return a.dtype == b.dtype && a.block_sizes == b.block_sizes && a.scales.dtype == b.scales.dtype
           && a.zeros.dtype == b.zeros.dtype;
}

inline bool operator!=(const DataFormat& a, const DataFormat& b) noexcept
{
    return !(a == b);
}

}  // namespace turbomind
