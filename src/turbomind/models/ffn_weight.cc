// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/ffn_weight.h"

namespace turbomind {

FfnWeight::FfnWeight(const core::FfnConfig& cfg):
    hidden_dim{cfg.hidden_dim},
    inter_size{cfg.inter_size / cfg.tp_size},
    act_type{static_cast<ActivationType>(cfg.act_type)},
    is_fused_silu{cfg.fuse_silu && act_type == ActivationType::kSilu},
    is_expert_{cfg.is_expert},
    data_type_{cfg.data_type},
    tp_size{cfg.tp_size},
    tp_rank{cfg.tp_rank}
{
}

void FfnWeight::prepare()
{
    Module::prepare();
}

TM_MODULE_REGISTER(FfnWeight, core::FfnConfig);

TM_MODULE_METHODS(FfnWeight, FFN_WEIGHT_CHILDREN, FFN_WEIGHT_PARAMS)

}  // namespace turbomind
