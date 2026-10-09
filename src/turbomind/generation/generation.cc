// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>

#include "src/turbomind/generation/generation.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/generation/generation_impl.h"
#include "src/turbomind/generation/guided_decoding.h"
#include "src/turbomind/generation/logits_processor.h"
#include "src/turbomind/generation/sampling.h"
#include "src/turbomind/generation/stop_criteria.h"

#include "src/turbomind/kernels/sampling_topk_kernels.h"  // InitializeRandomStates

#include "src/turbomind/models/llama/llama_kernels.h"  // invokePadLastTokenIds

namespace turbomind {

using std::unique_ptr;
using std::shared_ptr;
using std::vector;

Generation::Impl::Impl(DataType              dtype,
                       int                   max_batch_size,
                       int                   session_len,
                       int                   vocab_size,
                       int                   vocab_size_padded,
                       int                   hidden_units,
                       DataType              hidden_dtype,
                       const comm::HostComm& tp_group,
                       int                   phases,
                       const SpeculativePolicy* policy,
                       bool                     enable_metrics):
    max_batch_size_{max_batch_size},
    session_len_{session_len},
    token_row_width_{session_len + (policy ? policy->token_row_tail() : 0)},
    policy_{policy},
    shared_{logits_processor_,
            sampling_,
            stop_criteria_,
            random_states_,
            token_ids_,
            max_batch_size_,
            token_row_width_,
            token_ids_ptrs_buf_,
            data_}
{
    TM_CHECK_EQ(dtype, kFloat32);
    BaseGenerationParam base{max_batch_size, vocab_size, vocab_size_padded};
    const int verification_capacity = policy ? policy->max_proposals() + 1 : 1;
    const int parameter_capacity    = max_batch_size_ * verification_capacity;
    logits_processor_ =
        std::make_unique<LogitsProcessor>(base, phases, policy != nullptr, parameter_capacity);
    sampling_ = std::make_unique<Sampling>(base, phases, tp_group->rank(), parameter_capacity);
    stop_criteria_   = std::make_unique<StopCriteria>(base, phases);
    guided_decoding_ = std::make_unique<GuidedDecoding>(base, tp_group, phases);

    static_assert(sizeof(curandState_t) % alignof(curandState_t) == 0);
    random_states_ = {{max_batch_size_, (int)sizeof(curandState_t)}, kDEVICE};
    token_ids_     = {{max_batch_size_, token_row_width_}, kDEVICE};
    output_ids_    = {max_batch_size_, kDEVICE};
    for (int i = 0; i < max_batch_size_; ++i) {
        free_token_rows_.push_back(i);
        free_random_state_rows_.push_back(i);
    }

    random_seed_buf_          = {max_batch_size_, kCPUpinned};
    random_init_buf_          = {max_batch_size_, kCPUpinned};
    random_state_indices_buf_ = {max_batch_size_, kCPUpinned};

    token_ids_ptrs_buf_                    = {parameter_capacity, kCPUpinned};
    token_ids_buf_                         = {max_batch_size_ * (ssize_t)session_len_, kCPUpinned};
    output_ids_buf_                        = {max_batch_size_, kCPUpinned};
    request_to_generation_row_offsets_buf_ = {max_batch_size_ + 1, kCPUpinned};

    for (int i = 0; i < phases; ++i) {
        auto d = std::make_unique<GenerationData>();

        d->random_seed                       = empty_like(random_seed_buf_, kDEVICE);
        d->random_init                       = empty_like(random_init_buf_, kDEVICE);
        d->random_state_indices              = empty_like(random_state_indices_buf_, kDEVICE);
        d->token_ids_ptrs                    = empty_like(token_ids_ptrs_buf_, kDEVICE);
        d->request_to_generation_row_offsets = {max_batch_size_ + 1, kDEVICE};
        d->output_ids                        = empty_like(output_ids_, kDEVICE);

        data_.push_back(std::move(d));
    }

    if (policy_) {
        verification_ = std::make_unique<TargetVerification>(
            shared_, *policy_, hidden_units, hidden_dtype, enable_metrics, phases);
    }
}

void Generation::Impl::Setup(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    auto& d = *data_.at(phase);

    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    Buffer_<Sequence*> rc = env.at("requests").buffer();

    // random states
    d.random_init_needed = false;
    std::fill_n(random_init_buf_.data(), max_batch_size_, false);

    int* token_ids_buf                        = token_ids_buf_.data();
    int  generation_size                      = 0;
    request_to_generation_row_offsets_buf_[0] = 0;
    for (int i = 0; i < rc.size(); ++i) {
        auto&               c         = *rc[i];
        const SubmittedRow& submitted = *c.submitted;

        // An eagerly allocated row also serves a prompt whose extent the
        // policy extends (the bootstrapping forward writes proposals into it).
        const bool needs_row = submitted.generating || (policy_ && policy_->needs_prompt_token_row(c.prompt_len));

        if (needs_row && c.generation_token_ids_row < 0) {
            TM_CHECK(!free_token_rows_.empty());

            c.generation_token_ids_row = free_token_rows_.back();
            free_token_rows_.pop_back();

            auto* dst = shared_.RowPtr(c.generation_token_ids_row);
            std::copy_n(c.token_ids, c.seq_len, token_ids_buf);
            copy(token_ids_buf, c.seq_len, dst);
            token_ids_buf += c.seq_len;
        }

        if (submitted.generating) {
            if (c.generation_random_state_row < 0) {
                TM_CHECK(!free_random_state_rows_.empty());

                c.generation_random_state_row = free_random_state_rows_.back();
                free_random_state_rows_.pop_back();

                random_init_buf_[c.generation_random_state_row] = true;
                random_seed_buf_[c.generation_random_state_row] = c.gen_cfg.random_seed;
                d.random_init_needed                            = true;
            }

            random_state_indices_buf_[generation_size] = c.generation_random_state_row;
            token_ids_ptrs_buf_[generation_size]       = shared_.RowPtr(c.generation_token_ids_row);
            ++generation_size;
        }

        request_to_generation_row_offsets_buf_[i + 1] = generation_size;
    }

    if (d.random_init_needed) {
        copy(random_init_buf_, max_batch_size_, d.random_init);
        copy(random_seed_buf_, max_batch_size_, d.random_seed);
    }
    if (!verification_) {
        copy(token_ids_ptrs_buf_, generation_size, d.token_ids_ptrs);
    }
    else {
        copy(request_to_generation_row_offsets_buf_, rc.size() + 1, d.request_to_generation_row_offsets);
    }
    copy(random_state_indices_buf_, generation_size, d.random_state_indices);
    d.request_count   = rc.size();
    d.generation_size = generation_size;

    logits_processor_->Setup(phase, env);
    sampling_->Setup(phase, env);
    stop_criteria_->Setup(phase, env);
    guided_decoding_->Setup(phase, env);

    if (verification_) {
        verification_->Setup(phase, env);
    }
}

void Generation::Impl::Del(TensorMap& env)
{
    Buffer_<Sequence*> rc = env.at("requests").buffer();

    for (int i = 0; i < rc.size(); ++i) {
        auto& token_row = rc[i]->generation_token_ids_row;
        if (token_row >= 0) {
            free_token_rows_.push_back(token_row);
            token_row = -1;
        }

        auto& random_row = rc[i]->generation_random_state_row;
        if (random_row >= 0) {
            free_random_state_rows_.push_back(random_row);
            random_row = -1;
        }
    }
}

void Generation::Impl::Unprep(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    auto& d    = *data_.at(phase);
    auto& b    = *env.at("batch").data<BatchData*>()[0];
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    if (!verification_) {
        copy(output_ids_, b.bsz, d.output_ids);
    }
}

void Generation::Impl::Fetch(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    auto& d    = *data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    if (verification_) {
        verification_->Fetch(phase, env);
    }
    else {
        copy(d.output_ids, d.output_ids.size(), output_ids_buf_);
        env.produce("output_ids", output_ids_buf_);

        sampling_->Fetch(phase, env);
    }
}

void Generation::Impl::Update(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    sampling_->Update(phase, env);
}

void Generation::Impl::Forward(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    auto& d = *data_.at(phase);

    const auto stream = core::Context::stream().handle();

    if (d.random_init_needed) {
        InitializeRandomStates((curandState_t*)random_states_.raw_data(),
                               d.random_seed.data(),
                               d.random_init.data(),
                               max_batch_size_,
                               stream);
    }

    env.emplace("output_ids", output_ids_);       // out
    env.emplace("curand_state", random_states_);  // inout

    if (const int gs = d.generation_size) {

        env.emplace("token_ids_ptrs", d.token_ids_ptrs.slice(0, gs));
        env.emplace("curand_state_indices", d.random_state_indices.slice(0, gs));

        auto logits = env.consume("logits");

        if (logits.dtype() != kFloat32) {
            auto tmp = empty_like(logits, kFloat32);
            TM_SCOPE_CALL(invokeCastFloat2D(logits, tmp, stream));
            logits = std::move(tmp);
        }

        env.produce("logits", logits.slice(0, gs));

        logits_processor_->Forward(phase, env);

        guided_decoding_->FillMask(phase, env);
        guided_decoding_->ApplyMask(phase, env);

        sampling_->Forward(phase, env);

        guided_decoding_->ScheduleUpdate(phase, env);

        invokeAppendOneTokenAndAdvanceSequence(
            d.token_ids_ptrs.data(), output_ids_.data(), env.at("sequence_length").data<int>(), gs, stream);

        stop_criteria_->Forward(phase, env);

        guided_decoding_->FinishUpdate(phase, env);
    }
}

Generation::~Generation() = default;

Generation::Generation(DataType              dtype,
                       int                   max_batch_size,
                       int                   session_len,
                       int                   vocab_size,
                       int                   vocab_size_padded,
                       int                   hidden_units,
                       DataType              hidden_dtype,
                       const comm::HostComm& tp_group,
                       int                   phases,
                       const SpeculativePolicy* policy,
                       bool                     enable_metrics):
    impl_{std::make_unique<Impl>(dtype,
                                 max_batch_size,
                                 session_len,
                                 vocab_size,
                                 vocab_size_padded,
                                 hidden_units,
                                 hidden_dtype,
                                 tp_group,
                                 phases,
                                 policy,
                                 enable_metrics)}
{
}

void Generation::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kSetup) {
        return impl_->Setup(phase, env);
    }
    else if (op == BatchOp::kDel) {
        return impl_->Del(env);
    }
    else if (op == BatchOp::kForward) {
        return impl_->Forward(phase, env);
    }
    else if (op == BatchOp::kUnprep) {
        return impl_->Unprep(phase, env);
    }
    else if (op == BatchOp::kFetch) {
        return impl_->Fetch(phase, env);
    }
    else if (op == BatchOp::kUpdate) {
        return impl_->Update(phase, env);
    }
}

TargetVerification* Generation::Verification() noexcept
{
    return impl_->verification_.get();
}

}  // namespace turbomind
