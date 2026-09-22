// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <condition_variable>
#include <list>
#include <memory_resource>
#include <mutex>

#include "src/turbomind/engine/request.h"

namespace turbomind {

class RequestQueue {
public:
    explicit RequestQueue(): queue_{&pool_} {}

    void push(std::shared_ptr<Request> r)
    {
        {
            std::lock_guard lock{mutex_};
            if (closed_) {
                throw std::runtime_error("Queue is closed");
            }
            queue_.push_back(std::move(r));
        }
        cv_.notify_one();
    }

    void pop(std::vector<std::shared_ptr<Request>>& infer_reqs, unsigned max_infer, bool blocking, bool& abort)
    {
        std::unique_lock lock{mutex_};

        if (blocking) {
            cv_.wait(lock, [this] { return !queue_.empty() || closed_; });
        }

        if (closed_) {
            abort = true;
        }

        while (!queue_.empty() && infer_reqs.size() < max_infer) {
            auto& r = queue_.front();
            if (r->cancel_flag.exchange(1, std::memory_order_acq_rel) == 0) {
                infer_reqs.push_back(std::move(r));
            }
            queue_.pop_front();
        }
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    void notify()
    {
        cv_.notify_all();
    }

    void assign_unique_ids(std::vector<std::shared_ptr<Request>>& rs)
    {
        for (auto& r : rs) {
            r->unique_id = unique_id_.fetch_add(1, std::memory_order_relaxed);
        }
    }

private:
    std::atomic<uint64_t> unique_id_{1};

    std::pmr::unsynchronized_pool_resource   pool_;
    std::pmr::list<std::shared_ptr<Request>> queue_;

    std::mutex              mutex_;
    std::condition_variable cv_;

    bool closed_{};
};

}  // namespace turbomind
