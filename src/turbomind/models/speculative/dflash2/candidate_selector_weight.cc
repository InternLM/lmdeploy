// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/dflash2/dflash2_weight.h"

#include "src/turbomind/core/registry.h"

namespace turbomind {

TM_MODULE_REGISTER(CandidateSelectorWeight, core::CandidateSelectorWeightConfig);
TM_MODULE_METHODS(CandidateSelectorWeight, CANDIDATE_SELECTOR_WEIGHT_CHILDREN, CANDIDATE_SELECTOR_WEIGHT_PARAMS)

}  // namespace turbomind
