// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/generation/generation_impl.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/generation/logits_processor.h"
#include "src/turbomind/generation/sampling.h"
#include "src/turbomind/generation/stop_criteria.h"

#include "src/turbomind/kernels/sampling_topk_kernels.h"  // InitializeRandomStates
#include "src/turbomind/kernels/speculative_sequence_kernels.h"

#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/speculative/speculative_model.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

TargetVerification::TargetVerification(GenerationShared&        shared,
                                       const SpeculativePolicy& policy,
                                       int                      hidden_units,
                                       DataType                 hidden_dtype,
                                       bool                     enable_metrics,
                                       int                      phases):
    shared_{shared},
    draft_count_{policy.max_proposals()},
    enable_metrics_{enable_metrics}
{
    const int max_batch_size = shared_.max_batch_size;
    const int K              = draft_count_ + 1;

    request_token_ids_ptrs_buf_ = {max_batch_size, kCPUpinned};
    speculative_row_buf_        = {max_batch_size, kCPUpinned};
    selected_span_ids_buf_      = {max_batch_size * (ssize_t)K, kCPUpinned};
    accept_len_buf_             = {max_batch_size, kCPUpinned};
    if (enable_metrics_) {
        accepted_draft_count_buf_ = {max_batch_size, kCPUpinned};
    }

    for (int i = 0; i < phases; ++i) {
        auto d = std::make_unique<Data>();

        d->request_token_ids_ptrs = {max_batch_size, kDEVICE};
        d->selected_span_ids      = {max_batch_size * (ssize_t)K, kDEVICE};
        d->accept_len             = {max_batch_size, kDEVICE};
        d->finished_on_entry      = {max_batch_size, kDEVICE};
        d->speculative_row        = {max_batch_size, kDEVICE};

        if (enable_metrics_ && draft_count_ > 0) {
            d->accepted_draft_count = {max_batch_size, kDEVICE};
        }

        d->block_logits_active    = {K * (ssize_t)max_batch_size, kDEVICE};
        d->effective_history      = {K * (ssize_t)max_batch_size, kDEVICE};
        d->verification_draft_ids = {draft_count_ * (ssize_t)max_batch_size, kDEVICE};
        d->selected_hidden        = {{max_batch_size * (ssize_t)K, hidden_units}, hidden_dtype, kDEVICE};

        data_.push_back(std::move(d));
    }
}

TargetVerification::~TargetVerification() = default;

void TargetVerification::Setup(int phase, TensorMap& env)
{
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];
    auto& d    = *data_.at(phase);
    auto& g    = *shared_.data.at(phase);

    Buffer_<Sequence*> rc = env.at("requests").buffer();

    for (int i = 0; i < rc.size(); ++i) {
        auto& c = *rc[i];

        request_token_ids_ptrs_buf_[i] =
            c.generation_token_ids_row >= 0 ? shared_.RowPtr(c.generation_token_ids_row) : nullptr;
        speculative_row_buf_[i] = c.submitted->is_verification_row();
    }

    const int position_count = env.at("verification_positions").data<int>()[0];
    for (int position = 1; position < position_count; ++position) {
        std::copy_n(shared_.token_ids_ptrs_buf.data(),
                    g.generation_size,
                    shared_.token_ids_ptrs_buf.data() + position * g.generation_size);
    }
    copy(shared_.token_ids_ptrs_buf, position_count * g.generation_size, g.token_ids_ptrs);
    copy(request_token_ids_ptrs_buf_, rc.size(), d.request_token_ids_ptrs);
    copy(speculative_row_buf_, rc.size(), d.speculative_row);
}

void TargetVerification::PublishDraftInputs(int phase, TensorMap& env)
{
    auto& d = *data_.at(phase);
    auto& g = *shared_.data.at(phase);

    env.produce("request_token_ids_ptrs", d.request_token_ids_ptrs.slice(0, g.request_count));
    env.produce("request_to_generation_row_offsets",
                g.request_to_generation_row_offsets.slice(0, g.request_count + 1));
    env.produce("accept_len", d.accept_len.slice(0, g.request_count));
    env.produce("finished_on_entry", d.finished_on_entry.slice(0, g.request_count));
    env.produce("speculative_row", d.speculative_row.slice(0, g.request_count));
}

void TargetVerification::Fetch(int phase, TensorMap& env)
{
    auto& d    = *data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    auto&     batch = *env.at("batch").data<BatchData*>()[0];
    const int B     = batch.bsz;
    const int K     = draft_count_ + 1;

    copy(d.selected_span_ids, B * K, selected_span_ids_buf_);
    copy(d.accept_len, B, accept_len_buf_);

    env.produce("selected_span_ids", selected_span_ids_buf_.slice(0, B * K));
    env.produce("accept_len", accept_len_buf_.slice(0, B));

    if (enable_metrics_) {
        copy(d.accepted_draft_count, B, accepted_draft_count_buf_);
        env.produce("accepted_draft_count", accepted_draft_count_buf_.slice(0, B));
    }
}

Tensor TargetVerification::SelectedHiddenBuffer(int phase, core::ssize_t rows)
{
    return data_.at(phase)->selected_hidden.slice(0, rows);
}

void TargetVerification::InitializeTargetVerification(int phase, int position_count, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    auto& d = *data_.at(phase);
    auto& g = *shared_.data.at(phase);

    const auto stream = core::Context::stream().handle();

    Copy(env.at("finished").buffer(), g.request_count, d.finished_on_entry);
    Clear(d.accept_len.slice(0, g.request_count));

    if (g.random_init_needed) {
        InitializeRandomStates((curandState_t*)shared_.random_states.raw_data(),
                               g.random_seed.data(),
                               g.random_init.data(),
                               shared_.max_batch_size,
                               stream);
    }

    const Buffer_<int> entry_sequence_length = env.at("sequence_length").buffer();

    invokeInitializeTargetVerification(d.block_logits_active.data(),
                                       d.effective_history.data(),
                                       d.verification_draft_ids.data(),
                                       reinterpret_cast<const int* const*>(d.request_token_ids_ptrs.data()),
                                       entry_sequence_length.data(),
                                       d.finished_on_entry.data(),
                                       d.speculative_row.data(),
                                       enable_metrics_ ? d.accepted_draft_count.data() : nullptr,
                                       g.request_to_generation_row_offsets.data(),
                                       g.request_count,
                                       g.generation_size,
                                       position_count,
                                       stream);
}

void TargetVerification::ProcessTargetBlock(int phase, int position_count, const Tensor& target_logits, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    auto& d = *data_.at(phase);
    auto& g = *shared_.data.at(phase);

    const int B    = g.request_count;
    const int G    = g.generation_size;
    const int rows = position_count * G;

    const auto stream = core::Context::stream().handle();

    if (rows == 0) {
        return;
    }

    Tensor_<float> probabilities = empty_like(target_logits, kFloat32);
    invokeCastFloat2D(target_logits, probabilities, stream);

    shared_.logits_processor->ForwardVerificationBlock(phase,
                                                       probabilities,
                                                       g.token_ids_ptrs.slice(0, rows),
                                                       d.effective_history.slice(0, rows),
                                                       d.block_logits_active.slice(0, rows));

    VerifyTargetBlockParams p{};
    p.verification_draft_ids = d.verification_draft_ids.data();
    p.draft_row_stride       = G;
    p.logits_active          = d.block_logits_active.data();
    p.random_states          = reinterpret_cast<curandState_t*>(shared_.random_states.raw_data());
    p.random_state_indices   = g.random_state_indices.data();
    p.request_token_ids_ptrs = d.request_token_ids_ptrs.data();
    p.entry_sequence_length  = env.at("sequence_length").data<int>();
    p.request_to_generation_offsets = g.request_to_generation_row_offsets.data();
    p.speculative_row              = d.speculative_row.data();
    p.selected_span_ids            = d.selected_span_ids.data();
    p.selected_span_stride         = draft_count_ + 1;
    p.accept_len                   = d.accept_len.data();
    p.accepted_draft_count = enable_metrics_ ? d.accepted_draft_count.data() : nullptr;
    p.request_count        = B;
    p.generation_count     = G;
    p.position_count       = position_count;

    shared_.sampling->VerifyTargetBlock(phase, probabilities, p);
}

void TargetVerification::ClampSelectedSpan(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    auto& d = *data_.at(phase);
    auto& g = *shared_.data.at(phase);

    const Buffer_<int> entry_sequence_length = env.at("sequence_length").buffer();

    shared_.stop_criteria->ForwardSpeculative(phase,
                                              d.request_token_ids_ptrs.slice(0, g.request_count),
                                              entry_sequence_length,
                                              d.accept_len.slice(0, g.request_count),
                                              env.at("finished").buffer(),
                                              env);
}

void TargetVerification::CommitAcceptedSpan(int phase, Buffer_<int> sequence_length)
{
    auto& d = *data_.at(phase);
    auto& g = *shared_.data.at(phase);

    invokeAdvanceSequenceByAcceptedSpan(sequence_length.data(),
                                        d.accept_len.data(),
                                        enable_metrics_ ? d.accepted_draft_count.data() : nullptr,
                                        g.request_count,
                                        core::Context::stream().handle());
}

}  // namespace turbomind
