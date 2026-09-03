// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/linear_weight.h"

#include <utility>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

LinearWeight::LinearWeight(const core::LinearConfig& cfg):
    input_dim(cfg.input_dim),
    output_dim(cfg.output_dim),
    data_type(cfg.data_type),
    weight_format(cfg.format),
    has_bias_(cfg.has_bias)
{
}

gemm::QuantDesc MakeQuantDesc(const DataFormat& fmt)
{
    if (!fmt.is_quantized()) {
        return {gemm::QuantType::kNone, 0};
    }
    const int group_size = fmt.block_sizes.empty() ? 1 : fmt.block_sizes[0];
    if (fmt.dtype == kFloat8_e4m3 && fmt.block_sizes.size() > 1 && fmt.block_sizes[1] > 1) {
        return {gemm::QuantType::kB, group_size};
    }
    return {gemm::QuantType::kK, group_size};
}

void LinearWeight::copy_metadata_to(LinearWeight& dst) const
{
    dst.input_dim     = input_dim;
    dst.output_dim    = output_dim;
    dst.data_type     = data_type;
    dst.family        = family;
    dst.weight_format = weight_format;
    dst.input_format  = input_format;
    dst.output_format = output_format;
    dst.epilogue      = epilogue;
    dst.has_bias_     = has_bias_;
    dst.prepared_     = prepared_;
    dst.k_desc        = k_desc;
    dst.q_desc        = q_desc;
}

void LinearWeight::set_plan(gemm::WeightPlan plan)
{
    plan_ = std::move(plan);
}

bool LinearWeight::is_graph_compatible() const
{
    return TM_CHECK_NOTNULL(family)->is_graph_compatible();
}

void LinearWeight::prepare()
{
    if (!weight) {
        return;
    }
    if (prepared_) {
        return;
    }
    TM_CHECK(plan_);
    plan_->pack(*this, core::Context::stream().handle());
    EnsureFloatDtype(bias, data_type);
    prepared_ = true;
}

TM_MODULE_REGISTER(LinearWeight, core::LinearConfig);

TM_MODULE_METHODS(LinearWeight, LINEAR_WEIGHT_CHILDREN, LINEAR_WEIGHT_PARAMS)

}  // namespace turbomind
