// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "src/turbomind/engine/request.h"
#include "src/turbomind/engine/schedule_policy.h"

namespace turbomind {

class RequestQueue {
public:
    RequestQueue() = default;

    virtual ~RequestQueue() = default;

    static std::unique_ptr<RequestQueue> create(SchedulePolicy schedule_policy);

    void push(std::shared_ptr<Request> r)
    {
        {
            std::lock_guard lock{mutex_};
            if (closed_) {
                throw std::runtime_error("Queue is closed");
            }
            push_infer(std::move(r));
        }
        cv_.notify_one();
    }

    void kill(std::shared_ptr<Request> r)
    {
        {
            std::lock_guard lock{mutex_};
            if (closed_) {
                throw std::runtime_error("Queue is closed");
            }
            kill_.push_back(std::move(r));
        }
        cv_.notify_one();
    }

    void pop(std::vector<std::shared_ptr<Request>>& infer_reqs,
             std::vector<std::shared_ptr<Request>>& kill_reqs,
             unsigned                               max_infer,
             bool                                   blocking,
             bool&                                  abort)
    {
        std::unique_lock lock{mutex_};

        if (blocking) {
            cv_.wait(lock, [this] { return !(infer_queue_empty() && kill_.empty()) || closed_; });
        }

        if (closed_) {
            abort = true;
        }

        pop_infer(infer_reqs, max_infer);

        kill_reqs.insert(kill_reqs.end(), kill_.begin(), kill_.end());
        kill_.clear();
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

protected:
    static bool try_claim_request(const std::shared_ptr<Request>& r)
    {
        return r->cancel_flag.exchange(1, std::memory_order_acq_rel) == 0;
    }

    virtual bool infer_queue_empty() const                                                        = 0;
    virtual void push_infer(std::shared_ptr<Request> r)                                           = 0;
    virtual void pop_infer(std::vector<std::shared_ptr<Request>>& infer_reqs, unsigned max_infer) = 0;

    std::atomic<uint64_t> unique_id_{};

    std::pmr::unsynchronized_pool_resource pool_;

    std::vector<std::shared_ptr<Request>> kill_;

    std::mutex              mutex_;
    std::condition_variable cv_;

    bool closed_{};
};

class FifoRequestQueue: public RequestQueue {
public:
    FifoRequestQueue(): queue_{&pool_} {}

private:
    bool infer_queue_empty() const override
    {
        return queue_.empty();
    }

    void push_infer(std::shared_ptr<Request> r) override
    {
        queue_.push_back(std::move(r));
    }

    void pop_infer(std::vector<std::shared_ptr<Request>>& infer_reqs, unsigned max_infer) override
    {
        while (!queue_.empty() && infer_reqs.size() < max_infer) {
            auto& r = queue_.front();
            if (try_claim_request(r)) {
                infer_reqs.push_back(std::move(r));
            }
            queue_.pop_front();
        }
    }

    std::pmr::list<std::shared_ptr<Request>> queue_;
};

class PriorityRequestQueue: public RequestQueue {
public:
    PriorityRequestQueue(): queue_{&pool_} {}

private:
    struct PriorityEntry {
        uint8_t                  priority;
        uint64_t                 enqueue_order;
        std::shared_ptr<Request> request;
    };

    static bool priority_entry_is_worse(const PriorityEntry& lhs, const PriorityEntry& rhs)
    {
        return std::tie(lhs.priority, lhs.enqueue_order) > std::tie(rhs.priority, rhs.enqueue_order);
    }

    bool infer_queue_empty() const override
    {
        return queue_.empty();
    }

    void push_infer(std::shared_ptr<Request> r) override
    {
        const auto priority = r->gen_cfg.priority;
        queue_.push_back(PriorityEntry{priority, enqueue_order_++, std::move(r)});
        std::push_heap(queue_.begin(), queue_.end(), priority_entry_is_worse);
    }

    void pop_infer(std::vector<std::shared_ptr<Request>>& infer_reqs, unsigned max_infer) override
    {
        while (!queue_.empty() && infer_reqs.size() < max_infer) {
            std::pop_heap(queue_.begin(), queue_.end(), priority_entry_is_worse);
            auto& entry = queue_.back();
            if (try_claim_request(entry.request)) {
                infer_reqs.push_back(std::move(entry.request));
            }
            queue_.pop_back();
        }
    }

    uint64_t                        enqueue_order_{};
    std::pmr::vector<PriorityEntry> queue_;
};

}  // namespace turbomind
