// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

#include <string>
#include <vector>

namespace turbomind::core {

struct Eagle3WeightConfig: ModuleConfig {
    Eagle3WeightConfig(): ModuleConfig{"Eagle3Weight"} {}

#define EAGLE3_WEIGHT_FIELDS(X) X(DataType, data_type)

    EAGLE3_WEIGHT_FIELDS(TM_MEMBER)
    TM_FOR_EACH(Eagle3WeightConfig, EAGLE3_WEIGHT_FIELDS)

#undef EAGLE3_WEIGHT_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

/// EAGLE3's own weight tree. Attached at ModelWeight::spec on the draft tree.
class Eagle3Weight: public core::Module {
public:
    const char* type() const override
    {
        return "Eagle3Weight";
    }

    Eagle3Weight() = default;

    explicit Eagle3Weight(const core::Eagle3WeightConfig&) {}

    bool verify(std::vector<std::string>& missing) override;

    /// Per-draft-layer hidden norm; null when the layer has none.
    NormWeight* hidden_norm(int layer) const;

#define EAGLE3_WEIGHT_CHILDREN(X)                                                                                      \
    X(LinearWeight, target_hidden_proj)                                                                                \
    X(core::ModuleList, hidden_norms)

#define EAGLE3_WEIGHT_PARAMS(X)

    TM_MODULE_DECLARE(Eagle3Weight, EAGLE3_WEIGHT_CHILDREN, EAGLE3_WEIGHT_PARAMS)
};

}  // namespace turbomind
