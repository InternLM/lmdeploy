// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/speculative/fixed_chain_policy.h"
#include "src/turbomind/models/speculative/fixed_chain_setup.h"
#include "src/turbomind/models/speculative/speculative_model.h"

#include <vector>

namespace turbomind {

/// Shared skeleton for fixed-chain speculators. Implements the draft pass as one
/// refresh step followed by k - 1 extension steps, and delegates the per-model
/// math (embedding table, carry, norm + projection, LM head) to the hooks below.
/// See docs/adr/0001-fixed-chain-speculative-model-base.md.
class FixedChainSpeculativeModel: public SpeculativeModel {
public:
    explicit FixedChainSpeculativeModel(const SpeculativeModelArgs& args);
    ~FixedChainSpeculativeModel() override;

    const SpeculativePolicy& policy() const override;

    void Run(BatchOp op, int phase, TensorMap& env) override;

    HiddenStateTap*         Tap(int phase) override;

    void RunDraft(int phase, const DraftContext& ctx, TensorMap& env) final;

protected:
    struct CommonData {
        Buffer_<int>  draft_input_ids;
        Buffer_<int>  draft_selected_token_pos;
        Buffer_<bool> draft_candidate_active;
        Buffer_<int>  draft_proposal_ids;
        Tensor        draft_selected_normalized_hidden;
    };

    enum class EmbedStage { kRefresh, kExtension };

    struct CombineResult {
        Tensor residual;
        Tensor attention_input;
    };

    /// The tap whose captured state feeds the draft pass.
    virtual HiddenStateTap* TapSource() = 0;

    /// Carry entering the refresh step: the target's tapped state, collected.
    virtual Tensor InitialCarry(int phase, cudaStream_t stream) = 0;

    /// Embedding lookup for the token entering the current step. `rows` sizes the
    /// storage; refresh-stage hooks may patch the embeddings in place.
    virtual Tensor Embed(int                phase,
                         const Buffer_<int>& ids,
                         int                rows,
                         const DraftContext& ctx,
                         TensorMap&         env,
                         EmbedStage         stage) = 0;

    /// Norm-concat of embeddings and carry, projected to the decoder input. The
    /// returned residual feeds the decoder; attention_input is optional.
    virtual CombineResult Combine(int phase, Tensor embeddings, Tensor carry, int rows, TensorMap& env) = 0;

    /// Carry entering step `step + 1`, derived from the previous decoder output.
    /// `step` is 0 when `out` is the refresh output.
    virtual Tensor NextCarry(const LanguageModel::DecoderOutputs& out,
                             int                                 step,
                             int                                 phase,
                             TensorMap&                          env,
                             cudaStream_t                        stream) = 0;

    /// The language model whose LM head turns draft hiddens into logits.
    virtual LanguageModel& HeadModel() = 0;

    CommonData& common(int phase)
    {
        return data_.at(phase);
    }

    LanguageModel&       target_;
    LanguageModel        draft_;
    const Communicators& comm_;
    const bool           use_ag2d_;
    const int            draft_hidden_;

    FixedChainPolicy policy_;
    FixedChainSetup   fixed_chain_;

    Buffer_<int> draft_identity_token_pos_;

private:
    void Setup(int phase, TensorMap& env);

    std::vector<CommonData> data_;
};

}  // namespace turbomind
