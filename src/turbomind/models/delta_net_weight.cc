// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/delta_net_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

DeltaNetWeight::DeltaNetWeight(const core::DeltaNetConfig& cfg):
    hidden_dim(cfg.hidden_dim),
    num_k_heads(cfg.num_k_heads),
    num_v_heads(cfg.num_v_heads),
    key_head_dim(cfg.key_head_dim),
    value_head_dim(cfg.value_head_dim),
    d_conv(cfg.d_conv),
    bias(cfg.has_bias),
    tp_size(cfg.tp_size),
    tp_rank(cfg.tp_rank),
    data_type(cfg.data_type)
{
}

void DeltaNetWeight::prepare()
{
    Module::prepare();

    EnsureFloatDtype(A_log, data_type);
    EnsureFloatDtype(dt_bias, data_type);
    EnsureFloatDtype(conv1d, data_type);
}

TM_MODULE_REGISTER(DeltaNetWeight, core::DeltaNetConfig);

TM_MODULE_METHODS(DeltaNetWeight, DELTA_NET_WEIGHT_CHILDREN, DELTA_NET_WEIGHT_PARAMS)

}  // namespace turbomind
