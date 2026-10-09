// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <optional>

namespace turbomind {

/// What one speculative round costs in rows and cache.
struct RoundExtent {
    int query_rows;    // target query rows the round submits
    int private_tail;  // cache rows written past key capacity, never committed
};

/// Extra room the final prompt forward needs so it can write the first proposals.
struct BootstrapExtent {
    int cache_write_end;
    int min_session_len;
};

/// Fixed-width methods ignore this. It carries what the scheduler already knows
/// about the row being planned, so a width that depends on position within a
/// sequence needs no signature change. Adapting to a request's own acceptance
/// history additionally needs request identity and a feedback call; see Deferred.
struct RoundRequest {
    int query_begin;  // where this round's first query row lands
    int prompt_len;
};

/// The scheduling and verification contract of a speculative method. Consumed by
/// Scheduler and Engine, neither of which knows the algorithm.
class SpeculativePolicy {
public:
    virtual ~SpeculativePolicy() = default;

    /// Upper bound on proposals per round. Sizes verification buffers and the
    /// selected-span stride; not the count actually proposed in a given round.
    virtual int max_proposals() const = 0;

    /// Extra token-row columns past session_len that proposal writes need. Sizes
    /// Generation's token row; distinct from RoundExtent::private_tail, which sizes
    /// the KV cache.
    virtual int token_row_tail() const = 0;

    virtual RoundExtent Extent(const RoundRequest& req) const = 0;

    /// Nullopt when the method does not bootstrap from the prompt pass.
    virtual std::optional<BootstrapExtent> Bootstrap(int prompt_len) const = 0;

    /// True when the final prompt forward must have a token row already allocated
    /// so it can write the first proposals into it. Read by Generation, which
    /// otherwise allocates a row only for generating requests.
    bool needs_prompt_token_row(int prompt_len) const
    {
        return Bootstrap(prompt_len).has_value();
    }
};

}  // namespace turbomind
