// Copyright (c) OpenMMLab. All rights reserved.
//
// DFlash2 speculative method — M2 skeleton.
//
// At this stage the model only validates and holds the DFlash2 weight tree so
// the engine can load target + draft together (smoke). The draft forward
// (context-KV, grouped conv, non-causal SWA, selector) lands in M3; verification
// reuses the D7 SM75 kernel in M4.
#pragma once

#include "src/turbomind/models/speculative/speculative_model.h"
#include "src/turbomind/models/speculative/speculative_policy.h"

#include <memory>

namespace turbomind {

class Dflash2Weight;

/// Block-diffusion policy: one block of `n_proposals + 1` query rows
/// (`[next, mask x n]`), no private cache tail.
class Dflash2Policy final: public SpeculativePolicy {
public:
    explicit Dflash2Policy(int n_proposals): n_{n_proposals} {}

    int max_proposals() const override;
    int token_row_tail() const override;
    RoundExtent Extent(const RoundRequest& request) const override;
    std::optional<BootstrapExtent> Bootstrap(int prompt_len) const override;

private:
    int n_{};
};

class Dflash2Model final: public SpeculativeModel {
public:
    explicit Dflash2Model(const SpeculativeModelArgs& args);
    ~Dflash2Model() override;

    const SpeculativePolicy& policy() const override;
    void Run(BatchOp op, int phase, TensorMap& env) override;
    HiddenStateTap* Tap(int phase) override;
    void RunDraft(int phase, const DraftContext& ctx, TensorMap& env) override;

private:
    const Dflash2Weight& spec_weights_;
    Dflash2Policy        policy_;
};

}  // namespace turbomind
