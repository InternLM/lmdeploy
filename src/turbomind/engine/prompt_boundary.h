// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

namespace turbomind {

// Pure geometry of the prompt-boundary publish point. Decides where the reusable
// boundary B = prompt_len - skip lands and whether a partial sibling node is
// needed (B strictly inside a block) or B is block-aligned (whole blocks already
// tile [0, B); only the clamp/checkpoint applies). `miss` is the first prompt
// block index not matched in the trie (AcceptState::miss). No scheduler state.
struct PromptBoundaryPlan {
    bool valid     = false;  // a boundary exists (set prompt_boundary_node)
    bool partial   = false;  // needs a partial sibling node; else block-aligned
    int  pos       = 0;      // B
    int  block     = 0;      // j = block holding the last token before B
    int  node_size = 0;      // partial node length (B - j*block_size) when partial
};

inline PromptBoundaryPlan PlanPromptBoundary(int prompt_len, int block_size, int skip, int miss)
{
    PromptBoundaryPlan p{};
    if (skip < 1) {
        skip = 1;  // defensive; the scheduler also clamps at construction
    }
    const int B = prompt_len - skip;
    if (B < 1) {
        return p;  // boundary before the first token: nothing to publish
    }
    const int j = (B - 1) / block_size;  // block holding the last token before B
    if (B % block_size != 0) {
        // B strictly inside block j: a partial sibling node is required so [0, B)
        // is fully matchable (whole blocks [0, j*bs) + this node). miss < j keeps
        // the parent chain [0..j-1] indexed and the node off the matcher-side miss
        // block. miss < j (with miss >= 0) implies j >= 1, so block j-1 exists.
        if (miss < j) {
            p.valid     = true;
            p.partial   = true;
            p.pos       = B;
            p.block     = j;
            p.node_size = B - j * block_size;
        }
    }
    else {
        // B block-aligned: block j ends exactly at B and already tiles [0, B);
        // no partial node, only the clamp target. miss <= j: block j is matched
        // or created-and-indexed.
        if (miss <= j) {
            p.valid   = true;
            p.partial = false;
            p.pos     = B;
            p.block   = j;
        }
    }
    return p;
}

}  // namespace turbomind
