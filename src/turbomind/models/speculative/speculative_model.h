// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/speculative/speculative_policy.h"

namespace turbomind {

class CacheRegistry;
struct Context;
struct EngineParam;
class HiddenStateTap;
class InputProcessor;
class LanguageModel;
class ModelWeight;

struct DraftContext {
    int           batch_size;
    Buffer_<int>  target_q_offsets;
    Buffer_<int>  target_k_offsets;
    Buffer_<int>  accept_len;
    Buffer_<int>  sequence_length;
    Buffer_<bool> finished_on_entry;
    int* const*   request_token_ids_ptrs;
    Tensor        target_pre_final_residual;
    Tensor        embedding_storage;
    Tensor        head_storage;
    InputProcessor* input_processor{};
};

struct SpeculativeModelArgs {
    CacheRegistry&     registry;
    const EngineParam& param;
    const Context&     ctx;
    LanguageModel&     target;
    const ModelWeight& draft_weights;
    int                phases;
};

class SpeculativeModel {
public:
    virtual ~SpeculativeModel() = default;

    virtual const SpeculativePolicy& policy() const = 0;

    /// The draft model's batch-op lifecycle: kAdd, kSetup, and kPrepare.
    virtual void Run(BatchOp op, int phase, TensorMap& env) = 0;

    /// The one target-pass hook: the tap the target decoder captures through,
    /// armed by the executor before the target pass. Null when the policy
    /// taps no layer.
    virtual HiddenStateTap* Tap(int phase)
    {
        return nullptr;
    }

    virtual bool requires_successor_input_embeddings() const
    {
        return false;
    }

    virtual void RunDraft(int phase, const DraftContext& ctx, TensorMap& env) = 0;
};

}  // namespace turbomind
