// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_format.h"
#include "src/turbomind/core/module.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::core {

struct LinearConfig: ModuleConfig {
    LinearConfig(): ModuleConfig{"LinearWeight"} {}

#define LINEAR_FIELDS(X)                                                                                               \
    X(int, input_dim)                                                                                                  \
    X(int, output_dim)                                                                                                 \
    X(DataType, data_type)                                                                                             \
    X(DataFormat, format)                                                                                              \
    X(bool, has_bias)

    LINEAR_FIELDS(TM_MEMBER)
    TM_FOR_EACH(LinearConfig, LINEAR_FIELDS)

#undef LINEAR_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

using gemm::Epilogue;
using gemm::MatrixLayout;

/// Derive (input_format, output_format) for a GEMM whose weight uses
/// `weight_format`, given the model's activation dtype and hardware SM.
std::pair<DataFormat, DataFormat> DeriveActivationFormats(const DataFormat& weight_format, DataType data_type, int sm);

/// Derive GEMM QuantDesc for an operand described by DataFormat.
/// For unquantized formats, returns {QuantType::kNone, 0}.
gemm::QuantDesc MakeQuantDesc(const DataFormat& fmt);

class LinearWeight: public core::Module {
public:
    const char* type() const override
    {
        return "LinearWeight";
    }

    LinearWeight() = default;
    LinearWeight(const core::LinearConfig& cfg);

    void prepare() override;
    void copy_metadata_to(LinearWeight& dst) const;

    /// Set grouped-GEMM mode (for MoE expert weights that need row-major layout).
    void set_grouped(bool grouped)
    {
        is_grouped_ = grouped;
    }

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(weight);
    }

    // --- three DataFormats fully describe the GEMM ---
    DataFormat weight_format{};  // from cfg.format
    DataFormat input_format{};   // derived in ctor
    DataFormat output_format{};  // derived in ctor

    DataType input_dtype() const
    {
        return input_format.dtype;
    }
    DataType output_dtype() const
    {
        return output_format.dtype;
    }

    // --- dimensions + model activation dtype ---
    int      input_dim  = 0;
    int      output_dim = 0;
    DataType data_type{};  // model activation dtype, copied from cfg.data_type

    // --- GEMM knobs ---
    Epilogue     epilogue{};
    MatrixLayout k_desc{};
    MatrixLayout q_desc{};

#define LINEAR_WEIGHT_CHILDREN(X)

#define LINEAR_WEIGHT_PARAMS(X)                                                                                        \
    X(weight)                                                                                                          \
    X(bias)                                                                                                            \
    X(scales)                                                                                                          \
    X(zeros)

    TM_MODULE_DECLARE(LinearWeight, LINEAR_WEIGHT_CHILDREN, LINEAR_WEIGHT_PARAMS)

private:
    bool has_bias_   = false;
    bool is_grouped_ = false;
};

}  // namespace turbomind
