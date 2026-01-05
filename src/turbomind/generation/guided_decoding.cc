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
    BaseGenerationParam{base}, tp_group_{tp_group}
{
    const auto bitmask_size = xgrammar::GetBitmaskSize(vocab_size_padded_);

    bitmask_buf_    = {{max_batch_size_, bitmask_size}, kCPUpinned};
    output_ids_buf_ = {max_batch_size_, kCPUpinned};

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
        if (tp_group_->rank() == 0) {
            for (size_t i = 0; i < d.matchers.size(); ++i) {
                if (const auto& matcher = d.matchers[i]; matcher && !matcher->IsTerminated()) {
                    matcher->FillNextTokenBitmask(&dlbitmask, i);
                }
                else {
                    std::fill_n(bitmask_buf_.data() + i * bitmask_buf_.stride(0), bitmask_buf_.stride(0), 0);
                }
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
        ApplyTokenBitmaskInplace(env.at("logits"), d.bitmask.slice(0, d.matchers.size()));
    }
}

void GuidedDecoding::Update(int phase, TensorMap& env)
{
    if (auto& d = *data_.at(phase); d.active) {
        Copy(env.at("output_ids").buffer(), d.matchers.size(), output_ids_buf_);
        core::Context::stream().Sync();
        if (tp_group_->rank() == 0) {
            for (size_t i = 0; i < d.matchers.size(); ++i) {
                if (const auto& matcher = d.matchers[i]; matcher && !matcher->IsTerminated()) {
                    matcher->AcceptToken(output_ids_buf_[i]);
                }
            }
        }
    }
}

}  // namespace turbomind
