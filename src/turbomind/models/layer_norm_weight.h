// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"

namespace turbomind::core {

struct LayerNormConfig: ModuleConfig {
    LayerNormConfig(): ModuleConfig{"LayerNormWeight"} {}

#define LAYER_NORM_FIELDS(X)                                                                                           \
    X(int, dim)                                                                                                        \
    X(DataType, data_type)                                                                                             \
    X(float, norm_eps, 1e-6f)

    LAYER_NORM_FIELDS(TM_MEMBER)
    TM_FOR_EACH(LayerNormConfig, LAYER_NORM_FIELDS)

#undef LAYER_NORM_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

/// Affine LayerNorm with learnable gain and bias (gamma + beta).
/// Distinct from ``NormWeight`` (RMSNorm), which has only ``weight``.
class LayerNormWeight: public core::Module {
public:
    const char* type() const override
    {
        return "LayerNormWeight";
    }

    LayerNormWeight() = default;

    explicit LayerNormWeight(const core::LayerNormConfig& cfg);

    /// Post-load: cast weight + bias to configured dtype if needed.
    void prepare() override;

#define LAYER_NORM_WEIGHT_CHILDREN(X)

#define LAYER_NORM_WEIGHT_PARAMS(X)                                                                                    \
    X(weight)                                                                                                          \
    X(bias)

    TM_MODULE_DECLARE(LayerNormWeight, LAYER_NORM_WEIGHT_CHILDREN, LAYER_NORM_WEIGHT_PARAMS)

    float norm_eps_{};

private:
    std::vector<ssize_t> shape_;
    DataType             dtype_{};
};

}  // namespace turbomind
