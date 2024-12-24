// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/engine/request_queue.h"
#include "src/turbomind/engine/gateway.h"

#include "src/turbomind/engine/request.h"

namespace turbomind {

void RequestQueue::push(std::vector<std::shared_ptr<Request>> reqs)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            throw std::runtime_error("Queue is closed");
        }
        for (auto& r : reqs) {
            queue_.push(std::move(r));
        }
    }
    cv_.notify_one();
}

void RequestQueue::cancel(std::shared_ptr<Request> r)
{
    // -1 canceled
    //  0 queued
    //  1 active
    if (r->cancel_flag.exchange(-1, std::memory_order_acq_rel) != 0) {
        // request is picked up by engine
        return;
    }
    else {
        // not picked by engine yet, skip directly
        gateway_->notify({[r = std::move(r)] {  //
            UpdateState(*r, Request::kCancel, 0);
        }});
    }
}

void RequestQueue::kill(std::shared_ptr<Request> r)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            throw std::runtime_error("Queue is closed");
        }
        kill_.push_back(std::move(r));
    }
    cv_.notify_one();
}

void RequestQueue::pop(std::vector<std::shared_ptr<Request>>& infer_reqs,
                       std::vector<std::shared_ptr<Request>>& kill_reqs,
                       unsigned                               max_infer_num,
                       bool                                   blocking,
                       bool&                                  abort)
{
    std::unique_lock<std::mutex> lock(mutex_);

    if (blocking) {
        cv_.wait(lock, [this] { return !queue_.empty() || !kill_.empty() || closed_; });
        if (closed_) {
            abort = true;
            return;
        }
    }

    infer_reqs.clear();
    while (!queue_.empty() && infer_reqs.size() <= max_infer_num) {
        auto& r = queue_.front();
        if (r->cancel_flag.exchange(1, std::memory_order_acq_rel) == 0) {
            infer_reqs.push_back(std::move(r));
        }
        else {
            // Canceled requests are simply ignored
        }
        queue_.pop();
    }

    kill_reqs = std::move(kill_);
}

void RequestQueue::close()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }
    cv_.notify_all();
}

}  // namespace turbomind
