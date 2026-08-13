// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/ffn_weight.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/utils/cuda_utils.h"

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
    // Set epilogue on existing w1w3 child if fused silu is active.
    if (w1w3) {
        auto* fused = static_cast<LinearWeight*>(w1w3.get());
        if (is_fused_silu) {
            fused->epilogue = gemm::Epilogue::kGatedSilu;
            // SM90 FP8 fused SiLU quantizes to e4m3 + dynamic group-128 scales in-kernel.
            if (fused->weight_format.dtype == kFloat8_e4m3 && getSMVersion() == 90) {
                fused->set_fp8_fused_silu_output();
            }
        }
    }

    // Propagate grouped-GEMM flag for MoE expert weights
    if (is_expert_) {
        for_each_child([](const char*, Module* m) {
            if (auto* linear = dynamic_cast<LinearWeight*>(m)) {
                linear->set_grouped(true);
            }
        });
    }

    Module::prepare();  // recurse into children
}

TM_MODULE_REGISTER(FfnWeight, core::FfnConfig);

TM_MODULE_METHODS(FfnWeight, FFN_WEIGHT_CHILDREN, FFN_WEIGHT_PARAMS)

}  // namespace turbomind
