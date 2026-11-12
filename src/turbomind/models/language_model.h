#pragma once

#include <memory>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class ModelWeight;
class HiddenStateTap;
struct AttentionForwardMetadata;
struct Sequence;
class CacheRegistry;

class LanguageModel {
public:
    ~LanguageModel();

    LanguageModel() = default;

    LanguageModel(LanguageModel&&) noexcept;

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(impl_);
    }

    LanguageModel(CacheRegistry&     registry,
                  const EngineParam& engine,
                  const Context&     context,
                  const ModelWeight& weights,
                  int                phases);

    void Run(BatchOp op, int phase, TensorMap& env);

    bool has_embedding() const;
    bool has_head() const;

    Tensor Embed(const Buffer_<int>& input_ids, Tensor out, const TensorMap& env);

    struct DecoderInputs {
        Tensor       residual;
        Tensor       attention_input;
        Buffer_<int> selected_token_pos;
        Tensor       selected_hidden_buffer;

        const AttentionForwardMetadata* attention_metadata{};
        HiddenStateTap*                 taps{};
    };

    struct DecoderOutputs {
        Tensor selected_hidden;
        Tensor pre_final_residual;
    };

    DecoderOutputs RunDecoder(int phase, const DecoderInputs& in, TensorMap& env);

    Tensor Logits(const Tensor& hidden, Tensor out, const TensorMap& env);

    const ModelWeight& weights() const;

    int max_logits_len(const TensorMap& env) const;

    bool logits_use_workspace() const;

    void CommitAcceptedState(int phase, const Buffer_<int>& accept_len);

    size_t SpeculativeStateJournalBytes(int request_count, int verification_positions) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
