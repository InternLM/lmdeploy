// Copyright (c) OpenMMLab. All rights reserved.
//
// DFlash2 speculative method: block-diffusion drafting with a 5-layer decoder,
// incremental context-KV materialization from the target-hidden tap, grouped
// depthwise convs, a candidate-selector lattice, and a bespoke non-causal block
// attention. The target keeps its normal forward and verification.
#pragma once

#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/speculative/speculative_model.h"
#include "src/turbomind/models/speculative/speculative_policy.h"

#include <memory>
#include <vector>

namespace turbomind {

class Dflash2Weight;
class TargetHiddenProjection;
class Context;

/// Block-diffusion policy: one block of `n_proposals + 1` query rows
/// (`[next, mask x n]`), no private cache tail.
class Dflash2Policy final: public SpeculativePolicy {
public:
    explicit Dflash2Policy(int n_proposals): n_{n_proposals} {}

    int                            max_proposals() const override;
    int                            token_row_tail() const override;
    RoundExtent                    Extent(const RoundRequest& request) const override;
    std::optional<BootstrapExtent> Bootstrap(int prompt_len) const override;

private:
    int n_{};
};

class Dflash2Model final: public SpeculativeModel {
public:
    explicit Dflash2Model(const SpeculativeModelArgs& args);
    ~Dflash2Model() override;

    const SpeculativePolicy& policy() const override;
    void                     Run(BatchOp op, int phase, TensorMap& env) override;
    HiddenStateTap*          Tap(int phase) override;
    void                     RunDraft(int phase, const DraftContext& ctx, TensorMap& env) override;

private:
    const Dflash2Weight&                    spec_weights_;
    Dflash2Policy                           policy_;
    LanguageModel                           draft_;
    std::unique_ptr<TargetHiddenProjection> projection_;

    const Context&     ctx_;
    const EngineParam& param_;
    LanguageModel&     target_;
    int                hidden_{}, hd_{}, local_nh_{}, local_nkv_{}, total_heads_{}, inter_local_{};
    int                top_k_{}, sel_rank_{}, window_{}, block_{}, mask_token_id_{}, max_ctx_{};
    float              rope_base_{};

    std::vector<Tensor> ctx_k_;
    std::vector<Tensor> ctx_v_;
    std::vector<Tensor> ctx_k_data_;
    std::vector<Tensor> ctx_k_param_;
    std::vector<Tensor> ctx_v_data_;
    std::vector<Tensor> ctx_v_param_;
    Buffer_<int>        ctx_len_;
    bool                ctx_kv_int4_{};

    void MaterializeContext(const Tensor&       context_hidden,
                            const Buffer_<int>& base_dev,
                            const Buffer_<int>& committed_dev,
                            int                 rows,
                            cudaStream_t        stream);
    Tensor
    RunBlock(const Tensor& block_embeddings, const Tensor& positions, int rows, int ctx_rows, cudaStream_t stream);
};

}  // namespace turbomind
