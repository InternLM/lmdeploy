// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/dflash2/dflash2_weight.h"

#include "src/turbomind/core/registry.h"

namespace turbomind {

TM_MODULE_REGISTER(GroupedConvWeight, core::GroupedConvWeightConfig);
TM_MODULE_METHODS(GroupedConvWeight, GROUPED_CONV_WEIGHT_CHILDREN, GROUPED_CONV_WEIGHT_PARAMS)

}  // namespace turbomind
