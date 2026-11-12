// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/models/speculative/speculative_policy.h"

namespace turbomind {

/// Fixed-width chain drafting: k proposals are verified as k + 1 query rows,
/// with a k - 1 private cache tail for the next proposal round.
class FixedChainPolicy final: public SpeculativePolicy {
public:
    explicit FixedChainPolicy(int proposal_count): proposal_count_{proposal_count} {}

    int max_proposals() const override;

    int token_row_tail() const override;

    RoundExtent Extent(const RoundRequest& request) const override;

    std::optional<BootstrapExtent> Bootstrap(int prompt_len) const override;

private:
    const int proposal_count_;
};

}  // namespace turbomind
