// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/layer_norm_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

LayerNormWeight::LayerNormWeight(const core::LayerNormConfig& cfg):
    shape_{cfg.dim}, dtype_{cfg.data_type}, norm_eps_{cfg.norm_eps}
{
}

void LayerNormWeight::prepare()
{
    EnsureFloatDtype(weight, dtype_);
    if (bias) {
        EnsureFloatDtype(bias, dtype_);
    }
}

TM_MODULE_REGISTER(LayerNormWeight, core::LayerNormConfig);

TM_MODULE_METHODS(LayerNormWeight, LAYER_NORM_WEIGHT_CHILDREN, LAYER_NORM_WEIGHT_PARAMS)

}  // namespace turbomind
