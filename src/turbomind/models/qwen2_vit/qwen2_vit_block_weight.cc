// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/qwen2_vit/qwen2_vit_block_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/layer_norm_weight.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind {

TM_MODULE_REGISTER(Qwen2VitBlockWeight, core::Qwen2VitBlockConfig);

TM_MODULE_METHODS(Qwen2VitBlockWeight, QWEN2VIT_BLOCK_CHILDREN, QWEN2VIT_BLOCK_PARAMS)

}  // namespace turbomind
