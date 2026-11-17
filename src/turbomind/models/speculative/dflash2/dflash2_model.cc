// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/dflash2/dflash2_model.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/speculative/dflash2/dflash2_weight.h"
#include "src/turbomind/models/speculative/registry.h"

namespace turbomind {

int Dflash2Policy::max_proposals() const
{
    return n_;
}

int Dflash2Policy::token_row_tail() const
{
    return n_ + 1;
}

RoundExtent Dflash2Policy::Extent(const RoundRequest&) const
{
    // n proposals are verified as n + 1 query rows; the draft writes no private
    // cache tail (context KV is materialized separately).
    return {n_ + 1, 0};
}

std::optional<BootstrapExtent> Dflash2Policy::Bootstrap(int prompt_len) const
{
    return BootstrapExtent{prompt_len + n_, prompt_len + 2 * n_ + 1};
}

Dflash2Model::Dflash2Model(const SpeculativeModelArgs& args):
    spec_weights_{*TM_CHECK_NOTNULL(args.draft_weights.get<Dflash2Weight>("spec"))},
    policy_{args.param.spec_num_draft_tokens}
{
}

Dflash2Model::~Dflash2Model() = default;

const SpeculativePolicy& Dflash2Model::policy() const
{
    return policy_;
}

void Dflash2Model::Run(BatchOp, int, TensorMap&) {}

HiddenStateTap* Dflash2Model::Tap(int)
{
    return nullptr;
}

void Dflash2Model::RunDraft(int, const DraftContext&, TensorMap&) {}

TM_REGISTER_SPECULATIVE_MODEL("dflash2", Dflash2Model);

}  // namespace turbomind
