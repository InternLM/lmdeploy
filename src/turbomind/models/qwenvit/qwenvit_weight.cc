// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/qwenvit/qwenvit_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/models/layer_norm_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/qwenvit/qwenvit_block_weight.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

QwenVitWeight::QwenVitWeight(const core::QwenVitConfig& cfg): config_{cfg} {}

void QwenVitWeight::prepare()
{
    for_each_child([](const char* /*name*/, Module* child) {
        if (child)
            child->prepare();
    });

    // Qwen3.5 carries a learned positional-embedding table; keep it in the ViT dtype.
    if (pos_embed) {
        EnsureFloatDtype(pos_embed, config_.data_type);
    }
}

bool QwenVitWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    if (!patch_embed) {
        missing.push_back(full_path() + ": missing patch_embed");
    }
    // pos_embed only exists for models with a learned position-embedding table (Qwen3.5).
    if (config_.num_position_embeddings > 0 && !pos_embed) {
        missing.push_back(full_path() + ": missing pos_embed");
    }
    if (!blocks || blocks->size() != config_.depth) {
        missing.push_back(full_path() + ": blocks count mismatch (expected " + std::to_string(config_.depth) + ")");
    }
    if (!merger_fc1 || !merger_fc2 || !merger_norm) {
        missing.push_back(full_path() + ": missing merger");
    }
    return missing.empty();
}

QwenVitBlockWeight* QwenVitWeight::block(int i) const
{
    if (!blocks) {
        return nullptr;
    }
    return static_cast<QwenVitBlockWeight*>(blocks->child(std::to_string(i)));
}

TM_MODULE_REGISTER(QwenVitWeight, core::QwenVitConfig);

TM_MODULE_METHODS(QwenVitWeight, QWENVIT_WEIGHT_CHILDREN, QWENVIT_WEIGHT_PARAMS)

}  // namespace turbomind
