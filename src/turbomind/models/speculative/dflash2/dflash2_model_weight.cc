// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/dflash2/dflash2_weight.h"

#include "src/turbomind/core/registry.h"

namespace turbomind {

TM_MODULE_REGISTER(Dflash2Weight, core::Dflash2WeightConfig);
TM_MODULE_METHODS(Dflash2Weight, DFLASH2_WEIGHT_CHILDREN, DFLASH2_WEIGHT_PARAMS)

}  // namespace turbomind
