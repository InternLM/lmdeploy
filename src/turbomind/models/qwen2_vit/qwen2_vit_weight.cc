// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/qwen2_vit/qwen2_vit_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/models/layer_norm_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/qwen2_vit/qwen2_vit_block_weight.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

Qwen2VitWeight::Qwen2VitWeight(const core::Qwen2VitConfig& cfg): config_{cfg} {}

void Qwen2VitWeight::prepare()
{
    for_each_child([](const char* /*name*/, Module* child) {
        if (child)
            child->prepare();
    });
}

bool Qwen2VitWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    if (!patch_embed) {
        missing.push_back(full_path() + ": missing patch_embed");
    }
    if (!blocks || blocks->size() != config_.depth) {
        missing.push_back(full_path() + ": blocks count mismatch (expected " + std::to_string(config_.depth) + ")");
    }
    if (!merger_fc1 || !merger_fc2 || !merger_norm) {
        missing.push_back(full_path() + ": missing merger");
    }
    return missing.empty();
}

Qwen2VitBlockWeight* Qwen2VitWeight::block(int i) const
{
    if (!blocks) {
        return nullptr;
    }
    return static_cast<Qwen2VitBlockWeight*>(blocks->child(std::to_string(i)));
}

TM_MODULE_REGISTER(Qwen2VitWeight, core::Qwen2VitConfig);

TM_MODULE_METHODS(Qwen2VitWeight, QWEN2VIT_WEIGHT_CHILDREN, QWEN2VIT_WEIGHT_PARAMS)

}  // namespace turbomind
