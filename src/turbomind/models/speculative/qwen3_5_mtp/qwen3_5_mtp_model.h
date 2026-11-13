// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/models/speculative/fixed_chain_model.h"

#include <memory>
#include <vector>

namespace turbomind {

class Qwen35MtpWeight;
class TargetFinalHidden;

class Qwen35MtpModel final: public FixedChainSpeculativeModel {
public:
    explicit Qwen35MtpModel(const SpeculativeModelArgs& args);
    ~Qwen35MtpModel() override;

    bool requires_successor_input_embeddings() const override
    {
        return true;
    }

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
        Tensor normalized_concat;
        Tensor projected_full;
    };

    Tensor ProjectAndGatherFc(Data&              data,
                              const Tensor&      normalized_concat,
                              int                rows,
                              cudaStream_t       stream,
                              const TensorMap&   env);

    const Qwen35MtpWeight& spec_weights_;
    LlamaLinear&           linear_;

    const int hidden_units_;
    const int model_tp_rank_;
    const int model_tp_size_;

    std::unique_ptr<TargetFinalHidden> final_hidden_;
    std::vector<Data>                  data_;
};

}  // namespace turbomind
