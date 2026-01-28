// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>

#include "src/turbomind/core/check.h"
#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/request_queue.h"

namespace turbomind {

Gateway::Gateway(int size, std::function<std::shared_ptr<void>()> ctx_factory):
    size_{size}, queues_(size_), dp_thr_{1}, ctx_factory_{ctx_factory}, next_{0}
{
    for (int i = 0; i < size_; ++i) {
        queues_[i] = std::make_unique<RequestQueue>();
    }

    signal_thread_ = std::thread(&Gateway::signal_thread_entry, this);
}

void Gateway::shutdown()
{
    for (auto& q : queues_) {
        q->close();
    }

    signal_buffer_.close();
    signal_thread_.join();
}

void Gateway::push(std::shared_ptr<Request> r)
{
    int rank = -1;

    if (TM_UNLIKELY(!r->session.start_flag)) {
        // route to corresponding rank
        rank = binding_.find(r->session.id);
    }
    else if (TM_LIKELY(size_)) {
        rank = next_.fetch_add(1, std::memory_order_relaxed) % size_;
    }
    else {
        TM_LOG_ERROR("[Gateway] No queues available for submitting the request");
        notify({[r = std::move(r)] { UpdateState(*r, Request::kNoQueue, 0); }});
        return;
    }

    if (TM_LIKELY(rank >= 0)) {
        queues_[rank]->push({std::move(r)});
    }
    else {
        TM_LOG_ERROR("[Gateway] Failed to find a binded queue for %lu", r->session.id);
        notify({[r = std::move(r)] { UpdateState(*r, Request::kInvalid, 0); }});
    }
}

void Gateway::pop(std::vector<std::shared_ptr<Request>>& infer_reqs,
                  std::vector<std::shared_ptr<Request>>& kill_reqs,
                  unsigned                               max_infer,
                  bool                                   blocking,
                  bool&                                  abort,
                  comm::HostComm&                        dp_group,
                  int                                    qid)
{
    TM_CHECK_GE(qid, 0);

    auto& q = *queues_.at(qid);

    infer_reqs.clear();
    kill_reqs.clear();

    if (dp_group->n_ranks() == 1) {
        q.pop(infer_reqs, kill_reqs, max_infer, blocking, abort);
    }
    else {
        union {
            uint16_t data[2];
            uint32_t value;
        };
        while (true) {
            q.pop(infer_reqs, kill_reqs, max_infer, false, abort);
            data[0] = !(blocking && infer_reqs.empty() && kill_reqs.empty());  // ready?
            data[1] = abort;
            value   = comm::AllReduce(dp_group, value, comm::RedOp::kSum);
            if (data[0] >= dp_thr_ || data[1]) {
                break;
            }
        }
        abort = data[1];
    }

    // Assign a monotonic increasing id for each infer request
    q.assign_unique_ids(infer_reqs);

    // Bind for stateful inference
    std::vector<uint64_t> bind_ids;
    for (const auto& r : infer_reqs) {
        if (r->session.start_flag && !r->session.end_flag) {  // started but not ended
            bind_ids.push_back(r->session.id);
        }
    }

    /// TODO: fix qid <-> rank mapping
    if (!bind_ids.empty()) {
        binding_.bind(bind_ids, qid);
    }

    // Unbind for stateful kill
    std::vector<uint64_t> unbind_ids;
    for (const auto& r : kill_reqs) {
        unbind_ids.push_back(r->session.id);
    }
    if (!unbind_ids.empty()) {
        binding_.unbind(unbind_ids, qid);
    }
}

void Gateway::cancel(std::shared_ptr<Request> r)
{
    // {-1: canceled, 0: queued, 1: active}
    if (r->cancel_flag.exchange(-1, std::memory_order_acq_rel) == 0) {
        notify({[r = std::move(r)] {  //
            UpdateState(*r, Request::kCancel, 0);
        }});
    }
    else {
        // request is picked up by engine
    }
}

void Gateway::kill(std::shared_ptr<Request> r)
{
    if (auto rank = binding_.find(r->session.id); rank >= 0) {
        queues_[rank]->kill(std::move(r));
    }
    else {
        TM_LOG_ERROR("[Gateway] Failed to find a binded queue for %lu", r->session.id);
        notify({[r = std::move(r)] {  //
            UpdateState(*r, Request::kInvalid, 0);
        }});
    }
}

void Gateway::notify(std::vector<Signal> signals, bool pred)
{
    if (pred) {
        signal_buffer_.push(std::move(signals));
    }
}

void Gateway::signal_thread_entry() noexcept
{
    while (true) {
        bool                abort{};
        std::vector<Signal> signals = signal_buffer_.take_all(abort);
        if (abort) {
            break;
        }
        else {
            auto ctx = ctx_factory_();
            for (const auto& s : signals) {
                s();
            }
        }
    }
}

}  // namespace turbomind
