// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/qwen3_5_mtp/qwen3_5_mtp_weight.h"

#include "src/turbomind/core/registry.h"

namespace turbomind {

TM_MODULE_REGISTER(Qwen35MtpWeight, core::Qwen35MtpWeightConfig);
TM_MODULE_METHODS(Qwen35MtpWeight, QWEN35_MTP_WEIGHT_CHILDREN, QWEN35_MTP_WEIGHT_PARAMS)

}  // namespace turbomind
