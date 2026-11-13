// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/speculative/fixed_chain_policy.h"

namespace turbomind {

int FixedChainPolicy::max_proposals() const
{
    return proposal_count_;
}

int FixedChainPolicy::token_row_tail() const
{
    return 2;
}

RoundExtent FixedChainPolicy::Extent(const RoundRequest&) const
{
    return {proposal_count_ + 1, proposal_count_ - 1};
}

std::optional<BootstrapExtent> FixedChainPolicy::Bootstrap(int prompt_len) const
{
    return BootstrapExtent{prompt_len + proposal_count_ - 1, prompt_len + 2 * proposal_count_};
}

}  // namespace turbomind
