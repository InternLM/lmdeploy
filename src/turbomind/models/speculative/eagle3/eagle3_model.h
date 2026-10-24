// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/models/speculative/fixed_chain_model.h"

#include <memory>
#include <vector>

namespace turbomind {

class Eagle3Weight;
class TargetHiddenProjection;

class Eagle3Model final: public FixedChainSpeculativeModel {
public:
    explicit Eagle3Model(const SpeculativeModelArgs& args);
    ~Eagle3Model() override;

protected:
    HiddenStateTap* TapSource() override;
    Tensor          InitialCarry(int phase, cudaStream_t stream) override;
    Tensor          Embed(int                phase,
                          const Buffer_<int>& ids,
                          int                rows,
                          const DraftContext& ctx,
                          TensorMap&         env,
                          EmbedStage         stage) override;
    CombineResult   Combine(int phase, Tensor embeddings, Tensor carry, int rows, TensorMap& env) override;
    Tensor          NextCarry(const LanguageModel::DecoderOutputs& out,
                              int                                 step,
                              int                                 phase,
                              TensorMap&                          env,
                              cudaStream_t                        stream) override;
    LanguageModel&  HeadModel() override;

private:
    struct Data {
        Tensor draft_attention_input;
        Tensor draft_carry;
    };

    const Eagle3Weight& spec_weights_;

    std::unique_ptr<TargetHiddenProjection> projection_;
    std::vector<Data>                       data_;
};

}  // namespace turbomind
