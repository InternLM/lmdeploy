// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/internvit/internvit_weight.h"

#include "src/turbomind/core/registry.h"
#include "src/turbomind/models/internvit/internvit_block_weight.h"
#include "src/turbomind/models/layer_norm_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

InternVitWeight::InternVitWeight(const core::InternVitConfig& cfg): config_{cfg} {}

void InternVitWeight::prepare()
{
    for_each_child([](const char* /*name*/, core::Module* child) {
        if (child) {
            child->prepare();
        }
    });

    if (cls_token) {
        EnsureFloatDtype(cls_token, config_.data_type);
    }
    if (position_embeddings) {
        EnsureFloatDtype(position_embeddings, config_.data_type);
    }
}

bool InternVitWeight::verify(std::vector<std::string>& missing)
{
    core::Module::verify(missing);
    if (!patch_embed) {
        missing.push_back(full_path() + ": missing patch_embed");
    }
    if (!cls_token) {
        missing.push_back(full_path() + ": missing cls_token");
    }
    if (!position_embeddings) {
        missing.push_back(full_path() + ": missing position_embeddings");
    }
    if (!blocks || blocks->size() != config_.depth) {
        missing.push_back(full_path() + ": blocks count mismatch (expected " + std::to_string(config_.depth) + ")");
    }
    if (!projector_norm || !projector_fc1 || !projector_fc2) {
        missing.push_back(full_path() + ": missing projector");
    }
    return missing.empty();
}

InternVitBlockWeight* InternVitWeight::block(int i) const
{
    if (!blocks) {
        return nullptr;
    }
    return static_cast<InternVitBlockWeight*>(blocks->child(std::to_string(i)));
}

TM_MODULE_REGISTER(InternVitWeight, core::InternVitConfig);

TM_MODULE_METHODS(InternVitWeight, INTERNVIT_WEIGHT_CHILDREN, INTERNVIT_WEIGHT_PARAMS)

}  // namespace turbomind
