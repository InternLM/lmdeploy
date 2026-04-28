// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/ffn_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind {

FfnWeight::FfnWeight(const core::FfnConfig& cfg):
    hidden_dim_{cfg.hidden_dim},
    inter_size_{cfg.inter_size / cfg.tp_size},
    bias_{cfg.has_bias},
    tp_size_{cfg.tp_size},
    tp_rank_{cfg.tp_rank},
    data_type_{cfg.data_type},
    act_type_{static_cast<ActivationType>(cfg.act_type)},
    is_fused_silu_{cfg.fuse_silu && static_cast<ActivationType>(cfg.act_type) == ActivationType::kSilu},
    is_fused_moe_{cfg.fused_moe}
{
}

void FfnWeight::prepare()
{
    // Set epilogue on existing w1w3 child if fused silu is active.
    if (w1w3) {
        auto* fused = static_cast<LinearWeight*>(w1w3.get());
        if (is_fused_silu_) {
            fused->epilogue = gemm::Epilogue::kGatedSilu;
        }
    }

    // Propagate grouped-GEMM flag for MoE expert weights
    if (is_fused_moe_) {
        auto set_grouped = [](const char*, Module* m) {
            if (auto* lw = dynamic_cast<LinearWeight*>(m)) {
                lw->set_grouped(true);
            }
        };
        for_each_child(set_grouped);
    }

    Module::prepare();  // recurse into children
}

TM_MODULE_REGISTER(FfnWeight, core::FfnConfig);

TM_MODULE_METHODS(FfnWeight, FFN_WEIGHT_CHILDREN, FFN_WEIGHT_PARAMS)

}  // namespace turbomind
