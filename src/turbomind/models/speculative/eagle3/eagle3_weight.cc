// Copyright (c) OpenMMLab. All rights reserved.
#include "src/turbomind/models/speculative/eagle3/eagle3_weight.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/registry.h"

namespace turbomind {

bool Eagle3Weight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);
    if (!target_hidden_proj) {
        missing.push_back(full_path() + ": missing target_hidden_proj");
    }
    if (!hidden_norms || hidden_norms->size() == 0) {
        missing.push_back(full_path() + ": missing hidden_norms");
    }
    return missing.empty();
}

NormWeight* Eagle3Weight::hidden_norm(int layer) const
{
    if (!hidden_norms) {
        return nullptr;
    }
    return static_cast<NormWeight*>(hidden_norms->child(std::to_string(layer)));
}

TM_MODULE_REGISTER(Eagle3Weight, core::Eagle3WeightConfig);
TM_MODULE_METHODS(Eagle3Weight, EAGLE3_WEIGHT_CHILDREN, EAGLE3_WEIGHT_PARAMS)

}  // namespace turbomind
