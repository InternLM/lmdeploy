// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <memory>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"

namespace turbomind {

class SpeculativePolicy;
struct GenerationShared;

// Spec-side verification half of a speculative round: verifier initialization,
// target-block verification processing, stop-span clamping, and accepted-state
// commit, together with their per-phase buffers and staging. Constructed by
// the generation module only when a speculative policy is present and driven
// by the composed executor's speculative-round routine; it borrows the
// generation module's rows, random states, and sampler infrastructure through
// GenerationShared.
class TargetVerification {
public:
    TargetVerification(GenerationShared&     shared,
                       const SpeculativePolicy& policy,
                       int                   hidden_units,
                       DataType              hidden_dtype,
                       bool                  enable_metrics,
                       int                   phases);
    ~TargetVerification();

    // Spec-side participation in the shared BatchOp lifecycle.
    void Setup(int phase, TensorMap& env);
    void Fetch(int phase, TensorMap& env);

    // Publishes the verification draft inputs the composed round reads:
    // request token-row pointers, request-to-generation row offsets, accepted
    // lengths, entry finished flags, and the per-row speculative flag. Driven
    // by the executor's prepare bracket after generation's prepare and before
    // the draft's, so the draft's dependency is bracket code, not fanout
    // order.
    void PublishDraftInputs(int phase, TensorMap& env);

    // The round's selected-states storage, sliced to the selected row count.
    // The executor publishes it as the target decoder's selected-hidden
    // buffer before the target pass.
    Tensor SelectedHiddenBuffer(int phase, core::ssize_t rows);

    // Driven by the speculative-round routine.
    void InitializeTargetVerification(int phase, int position_count, TensorMap& env);
    void ProcessTargetBlock(int phase, int position_count, const Tensor& target_logits, TensorMap& env);
    void ClampSelectedSpan(int phase, TensorMap& env);
    void CommitAcceptedSpan(int phase, Buffer_<int> sequence_length);

private:
    struct Data {
        Buffer_<int*> request_token_ids_ptrs;
        Buffer_<int>  selected_span_ids;
        Buffer_<int>  accept_len;
        Buffer_<bool> finished_on_entry;
        Buffer_<int>  accepted_draft_count;
        Buffer_<bool> speculative_row;
        Buffer_<bool> block_logits_active;
        Buffer_<int>  effective_history;
        Buffer_<int>  verification_draft_ids;
        Tensor        selected_hidden;
    };

    GenerationShared& shared_;
    const int         draft_count_;
    const bool        enable_metrics_;

    std::vector<std::unique_ptr<Data>> data_;

    Buffer_<int*> request_token_ids_ptrs_buf_;
    Buffer_<bool> speculative_row_buf_;
    Buffer_<int>  selected_span_ids_buf_;
    Buffer_<int>  accept_len_buf_;
    Buffer_<int>  accepted_draft_count_buf_;
};

}  // namespace turbomind
