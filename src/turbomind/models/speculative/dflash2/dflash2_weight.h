// Copyright (c) OpenMMLab. All rights reserved.
//
// DFlash2 draft weight tree (TurboMind). Specifies the extra weight modules the
// DFlash2 speculative model needs beyond the plain draft decoder layers:
//   * fc               : context projection  (5 * hidden -> hidden)
//   * hidden_norm/norm : context + final RMSNorm
//   * attention_convs  : per-layer grouped conv (attention branch)
//   * mlp_convs        : per-layer grouped conv (MLP branch)
//   * selector         : candidate selector (low-rank projection + codebooks)
//
// Attached at ModelWeight::spec on the draft tree, like Eagle3Weight.
#pragma once

#include "src/turbomind/core/module.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

namespace turbomind::core {

struct GroupedConvWeightConfig: ModuleConfig {
    GroupedConvWeightConfig(): ModuleConfig{"GroupedConvWeight"} {}

#define GROUPED_CONV_WEIGHT_FIELDS(X) X(DataType, data_type)

    GROUPED_CONV_WEIGHT_FIELDS(TM_MEMBER)
    TM_FOR_EACH(GroupedConvWeightConfig, GROUPED_CONV_WEIGHT_FIELDS)

#undef GROUPED_CONV_WEIGHT_FIELDS
};

struct CandidateSelectorWeightConfig: ModuleConfig {
    CandidateSelectorWeightConfig(): ModuleConfig{"CandidateSelectorWeight"} {}

#define CANDIDATE_SELECTOR_WEIGHT_FIELDS(X) X(DataType, data_type)

    CANDIDATE_SELECTOR_WEIGHT_FIELDS(TM_MEMBER)
    TM_FOR_EACH(CandidateSelectorWeightConfig, CANDIDATE_SELECTOR_WEIGHT_FIELDS)

#undef CANDIDATE_SELECTOR_WEIGHT_FIELDS
};

struct Dflash2WeightConfig: ModuleConfig {
    Dflash2WeightConfig(): ModuleConfig{"Dflash2Weight"} {}

#define DFLASH2_WEIGHT_FIELDS(X) X(DataType, data_type)

    DFLASH2_WEIGHT_FIELDS(TM_MEMBER)
    TM_FOR_EACH(Dflash2WeightConfig, DFLASH2_WEIGHT_FIELDS)

#undef DFLASH2_WEIGHT_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

/// Content-generated grouped depthwise conv: `base_kernel` [2, taps, hidden]
/// plus a `kernel_projection` producing the per-token coefficients.
class GroupedConvWeight: public core::Module {
public:
    const char* type() const override
    {
        return "GroupedConvWeight";
    }

    GroupedConvWeight() = default;
    explicit GroupedConvWeight(const core::GroupedConvWeightConfig&) {}

#define GROUPED_CONV_WEIGHT_CHILDREN(X) X(LinearWeight, kernel_projection)

#define GROUPED_CONV_WEIGHT_PARAMS(X) X(base_kernel)

    TM_MODULE_DECLARE(GroupedConvWeight, GROUPED_CONV_WEIGHT_CHILDREN, GROUPED_CONV_WEIGHT_PARAMS)
};

/// Candidate selector: `hidden_projection` (hidden -> rank) plus the
/// predecessor/successor codebooks [vocab, rank].
class CandidateSelectorWeight: public core::Module {
public:
    const char* type() const override
    {
        return "CandidateSelectorWeight";
    }

    CandidateSelectorWeight() = default;
    explicit CandidateSelectorWeight(const core::CandidateSelectorWeightConfig&) {}

#define CANDIDATE_SELECTOR_WEIGHT_CHILDREN(X) X(LinearWeight, hidden_projection)

#define CANDIDATE_SELECTOR_WEIGHT_PARAMS(X)                                                                            \
    X(predecessor_codebook)                                                                                            \
    X(successor_codebook)

    TM_MODULE_DECLARE(CandidateSelectorWeight, CANDIDATE_SELECTOR_WEIGHT_CHILDREN, CANDIDATE_SELECTOR_WEIGHT_PARAMS)
};

/// DFlash2's own weight tree (ModelWeight::spec). Per-layer convs live on the
/// draft's DecoderLayerWeight (`attention_conv`/`mlp_conv`); this tree holds the
/// context projection, hidden norm and candidate selector.
class Dflash2Weight: public core::Module {
public:
    const char* type() const override
    {
        return "Dflash2Weight";
    }

    Dflash2Weight() = default;
    explicit Dflash2Weight(const core::Dflash2WeightConfig&) {}

#define DFLASH2_WEIGHT_CHILDREN(X)                                                                                     \
    X(LinearWeight, fc)                                                                                                \
    X(NormWeight, hidden_norm)                                                                                         \
    X(CandidateSelectorWeight, selector)

#define DFLASH2_WEIGHT_PARAMS(X)

    TM_MODULE_DECLARE(Dflash2Weight, DFLASH2_WEIGHT_CHILDREN, DFLASH2_WEIGHT_PARAMS)
};

}  // namespace turbomind
