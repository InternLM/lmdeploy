#include "src/turbomind/generation/guided_decoding.h"

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/kernels/apply_token_bitmask_inplace_cuda.h"
#include "xgrammar/matcher.h"
#include <dlpack/dlpack.h>

namespace turbomind {

struct GuidedDecoding::Data {
    Tensor_<int32_t> bitmask;
    bool             active{};

    std::vector<std::shared_ptr<xgrammar::GrammarMatcher>> matchers;
};

GuidedDecoding::GuidedDecoding(const BaseGenerationParam& base, const comm::HostComm& tp_group, int phases):
    BaseGenerationParam{base},        //
    tp_group_{tp_group->Split(0, 0)}  // duplicate to avoid data race
{
    const auto bitmask_size = xgrammar::GetBitmaskSize(vocab_size_padded_);

    bitmask_buf_    = {{max_batch_size_, bitmask_size}, kCPUpinned};
    output_ids_buf_ = {max_batch_size_, kCPUpinned};

    d2h_stream_    = core::Stream::create();
    sampling_done_ = core::Event::create();
    d2h_done_      = core::Event::create();

    for (int i = 0; i < phases; ++i) {
        auto& d    = data_.emplace_back(std::make_shared<Data>());
        d->bitmask = empty_like(bitmask_buf_);
    }
}

void GuidedDecoding::Setup(int phase, TensorMap& env)
{
    auto& d = *data_.at(phase);
    auto& b = *env.at("batch").data<BatchData*>()[0];

    d.matchers.clear();
    d.active = false;
    for (const auto& r : b.rc) {
        if (d.matchers.emplace_back(r->req->matcher)) {
            d.active = true;
        }
    }
}

void GuidedDecoding::FillMask(int phase, TensorMap& env)
{
    if (auto& d = *data_.at(phase); d.active) {
        static_assert(sizeof(ssize_t) == sizeof(int64_t));
        DLTensor dlbitmask{bitmask_buf_.data(),
                           DLDevice{kDLCPU, 0},
                           bitmask_buf_.ndim(),
                           xgrammar::GetBitmaskDLType(),
                           (int64_t*)bitmask_buf_.shape().data(),
                           nullptr,
                           0};

        std::vector<xgrammar::GrammarMatcher> active_matchers;
        std::vector<int32_t>                  active_indices;
        active_matchers.reserve(d.matchers.size());
        active_indices.reserve(d.matchers.size());

        if (tp_group_->rank() == 0) {
            for (size_t i = 0; i < d.matchers.size(); ++i) {
                if (const auto& m = d.matchers[i]; m && !m->IsTerminated()) {
                    active_matchers.emplace_back(*m);
                    active_indices.emplace_back(static_cast<int32_t>(i));
                }
                else {
                    std::fill_n(bitmask_buf_.data() + i * bitmask_buf_.stride(0),
                                bitmask_buf_.stride(0),
                                static_cast<int32_t>(-1));
                }
            }

            if (!active_matchers.empty()) {
                batch_matcher_.BatchFillNextTokenBitmask(&active_matchers, &dlbitmask, active_indices);
            }
        }
    }
}

void GuidedDecoding::ApplyMask(int phase, TensorMap& env)
{
    if (auto& d = *data_.at(phase); d.active) {
        const ssize_t numel = d.matchers.size() * bitmask_buf_.stride(0);
        if (tp_group_->n_ranks() > 1) {
            // bcast the data instead of `bitmask_buf` instance (which may avoid copying the data)
            comm::Broadcast(tp_group_, bitmask_buf_.data(), numel, 0);
        }
        Copy(bitmask_buf_.buffer(), numel, d.bitmask.buffer());
        // Use logits shape(0) instead of d.matchers.size() to ensure dimension match.
        // d.matchers.size() is the total number of requests in batch, but logits may be
        // sliced to only include requests that are still generating (generation_size).
        auto logits = env.at("logits");
        ApplyTokenBitmaskInplace(logits, d.bitmask.slice(0, logits.shape(0)));
    }
}

void GuidedDecoding::ScheduleUpdate(int phase, TensorMap& env)
{
    if (auto& d = *data_.at(phase); d.active) {
        // Record event on main stream after sampling GPU work is submitted.
        // The secondary stream will wait for this before issuing the D2H copy,
        // ensuring it reads the output_ids written by sampling.
        sampling_done_.Record(core::Context::stream());

        // D2H copy on secondary stream — overlaps with subsequent GPU kernels
        // on the main stream (AppendTokenIds, stop_criteria).
        d2h_stream_.Wait(sampling_done_);
        Copy(env.at("output_ids").buffer(), d.matchers.size(), output_ids_buf_, d2h_stream_);
        d2h_done_.Record(d2h_stream_);
    }
}

void GuidedDecoding::FinishUpdate(int phase)
{
    if (auto& d = *data_.at(phase); d.active) {
        // Wait only for the D2H copy to complete — the main stream's
        // AppendTokenIds + stop_criteria may still be executing on GPU.
        d2h_done_.Sync();

        if (tp_group_->rank() == 0) {
            // Collect active matchers and their token IDs for batch AcceptToken
            std::vector<xgrammar::GrammarMatcher> active_matchers;
            std::vector<int32_t>                  active_token_ids;
            active_matchers.reserve(d.matchers.size());
            active_token_ids.reserve(d.matchers.size());

            for (size_t i = 0; i < d.matchers.size(); ++i) {
                if (const auto& m = d.matchers[i]; m && !m->IsTerminated()) {
                    active_matchers.emplace_back(*m);
                    active_token_ids.emplace_back(output_ids_buf_[i]);
                }
            }

            if (!active_matchers.empty()) {
                xgrammar::BatchGrammarMatcher::BatchAcceptToken(&active_matchers, active_token_ids);
            }
        }
        // active_matchers destroyed here: refcount-- for each entry
    }
}

}  // namespace turbomind
