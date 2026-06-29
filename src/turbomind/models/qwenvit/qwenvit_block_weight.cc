// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/qwenvit/qwenvit_block_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/layer_norm_weight.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind {

TM_MODULE_REGISTER(QwenVitBlockWeight, core::QwenVitBlockConfig);

TM_MODULE_METHODS(QwenVitBlockWeight, QWENVIT_BLOCK_CHILDREN, QWENVIT_BLOCK_PARAMS)

}  // namespace turbomind
