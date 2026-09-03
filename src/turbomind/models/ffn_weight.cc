// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/ffn_weight.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/kernels/gemm/convert.h"
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
    if (w1w3 && is_fused_silu) {
        static_cast<LinearWeight*>(w1w3.get())->epilogue = gemm::Epilogue::kGatedSilu;
    }

    // MXFP4 remains BF16 by default in DeriveActivationFormats.  FFN is the
    // production boundary that opts compatible SM90 children into the native
    // E4M3-K128 x MXFP4-K32 path before LinearWeight::prepare() packs them.
    for_each_child([](const char*, Module* child) {
        auto* linear = dynamic_cast<LinearWeight*>(child);
        if (!linear || getSMVersion() != 90 || !gemm::HasSm90MixedKernel() || gemm::WeightPackEnv() == 0
            || linear->data_type != kBfloat16 || linear->weight_format.dtype != kFloat4_e2m1
            || linear->weight_format.block_sizes.empty() || linear->weight_format.block_sizes[0] != 32
            || linear->input_dim < 256 || linear->input_dim % 128 != 0 || linear->output_dim < 64
            || linear->output_dim % 64 != 0) {
            return;
        }
        const bool fuse_silu = linear->epilogue == gemm::Epilogue::kGatedSilu;
        if ((linear->epilogue != gemm::Epilogue::kNone && !fuse_silu)
            || (fuse_silu && linear->output_dim % 256 != 0)) {
            return;
        }
        linear->input_format.dtype = kFloat8_e4m3;
        linear->input_format.block_sizes = {128, 1};
        linear->input_format.scales.dtype = kFloat;
        linear->input_format.zeros.dtype = kNull;
    });

    // Set epilogue on existing w1w3 child if fused silu is active.
    if (w1w3) {
        auto* fused = static_cast<LinearWeight*>(w1w3.get());
        if (is_fused_silu) {
            const bool is_mxfp4_k32 = fused->weight_format.dtype == kFloat4_e2m1
                                       && !fused->weight_format.block_sizes.empty()
                                       && fused->weight_format.block_sizes[0] == 32;
            const bool supports_fp8_fused_output = getSMVersion() == 90
                                                    && fused->input_dtype() == kFloat8_e4m3
                                                    && (fused->weight_format.dtype == kFloat8_e4m3 || is_mxfp4_k32);
            if (supports_fp8_fused_output) {
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
