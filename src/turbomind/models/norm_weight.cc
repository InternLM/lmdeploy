// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/norm_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

NormWeight::NormWeight(const core::NormConfig& cfg): shape_{cfg.dim}, dtype_{cfg.data_type}, norm_eps_{cfg.norm_eps} {}

void NormWeight::prepare()
{
    EnsureFloatDtype(weight, dtype_);
}

TM_MODULE_REGISTER(NormWeight, core::NormConfig);

TM_MODULE_METHODS(NormWeight, NORM_WEIGHT_CHILDREN, NORM_WEIGHT_PARAMS)

}  // namespace turbomind
