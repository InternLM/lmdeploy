// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/qwen3_5vit/qwen3_5vit_block_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/layer_norm_weight.h"
#include "src/turbomind/models/linear_weight.h"

namespace turbomind {

TM_MODULE_REGISTER(Qwen3_5VitBlockWeight, core::Qwen3_5VitBlockConfig);

TM_MODULE_METHODS(Qwen3_5VitBlockWeight, QWEN3_5VIT_BLOCK_CHILDREN, QWEN3_5VIT_BLOCK_PARAMS)

}  // namespace turbomind
