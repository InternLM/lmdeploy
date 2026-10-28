// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

namespace turbomind::core {

struct Qwen35MtpWeightConfig: ModuleConfig {
    Qwen35MtpWeightConfig(): ModuleConfig{"Qwen35MtpWeight"} {}

#define QWEN35_MTP_WEIGHT_FIELDS(X) X(DataType, data_type)

    QWEN35_MTP_WEIGHT_FIELDS(TM_MEMBER)
    TM_FOR_EACH(Qwen35MtpWeightConfig, QWEN35_MTP_WEIGHT_FIELDS)

#undef QWEN35_MTP_WEIGHT_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class Qwen35MtpWeight final: public core::Module {
public:
    const char* type() const override
    {
        return "Qwen35MtpWeight";
    }

    Qwen35MtpWeight() = default;
    explicit Qwen35MtpWeight(const core::Qwen35MtpWeightConfig&) {}

#define QWEN35_MTP_WEIGHT_CHILDREN(X)                                                                                 \
    X(LinearWeight, fc)                                                                                               \
    X(NormWeight, pre_fc_norm_embedding)                                                                              \
    X(NormWeight, pre_fc_norm_hidden)

#define QWEN35_MTP_WEIGHT_PARAMS(X)

    TM_MODULE_DECLARE(Qwen35MtpWeight, QWEN35_MTP_WEIGHT_CHILDREN, QWEN35_MTP_WEIGHT_PARAMS)
};

}  // namespace turbomind
