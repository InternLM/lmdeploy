// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <atomic>
#include <condition_variable>
#include <list>
#include <memory_resource>
#include <mutex>
#include <queue>

#include "src/turbomind/engine/request.h"

namespace turbomind {

class Gateway;

class RequestQueue {
public:
    RequestQueue(Gateway* gateway): gateway_{gateway} {}

    void push(std::vector<std::shared_ptr<Request>> reqs);

    void pop(std::vector<std::shared_ptr<Request>>& infer_reqs,
             std::vector<std::shared_ptr<Request>>& kill_reqs,
             unsigned                               max_infer_num,
             bool                                   blocking,
             bool&                                  abort);

    void cancel(std::shared_ptr<Request> r);

    void kill(std::shared_ptr<Request> r);

    void close();

private:
    Gateway* gateway_;

    std::queue<std::shared_ptr<Request>> queue_;

    std::vector<std::shared_ptr<Request>> kill_;

    std::mutex              mutex_;
    std::condition_variable cv_;

    bool closed_{false};
};

class RequestQueueV2 {
public:
    explicit RequestQueueV2(Gateway* gateway): gateway_{gateway}, queue_{&memory_pool_} {}

    void push(std::shared_ptr<Request> r)
    {
        {
            std::lock_guard lock{mutex_};
            if (closed_) {
                throw std::runtime_error("Queue is clsoed");
            }
            queue_.push_back(std::move(r));
        }
        cv_.notify_one();
    }

    void kill(std::shared_ptr<Request> r)
    {
        {
            std::lock_guard lock{mutex_};
            if (closed_) {
                throw std::runtime_error("Queue is clsoed");
            }
            kill_.push_back(std::move(r));
        }
        cv_.notify_one();
    }

    bool try_pop(std::shared_ptr<Request>& r)
    {
        std::lock_guard lock{mutex_};

        /// TODO: cache the search start pos
        auto it = std::find_if(queue_.begin(), queue_.end(), [](const auto& r) {  //
            return r->session.start_flag;
        });

        if (it != queue_.end()) {
            r = std::move(*it);
            queue_.erase(it);
            return true;
        }

        return false;
    }

    void pop(std::vector<std::shared_ptr<Request>>& infer_reqs,
             std::vector<std::shared_ptr<Request>>& kill_reqs,
             unsigned                               max_infer,
             bool                                   blocking,
             bool&                                  abort)
    {
        std::unique_lock lock{mutex_};

        if (blocking) {
            cv_.wait(lock, [this] { return !(queue_.empty() && kill_.empty()) || closed_; });
            if (closed_) {
                abort = true;
                return;
            }
        }

        while (!queue_.empty() && infer_reqs.size() < max_infer) {
            auto& r = queue_.front();
            if (r->cancel_flag.exchange(1, std::memory_order_acq_rel) == 0) {
                infer_reqs.push_back(std::move(r));
            }
            queue_.pop_front();
        }

        kill_reqs.insert(kill_reqs.end(), kill_.begin(), kill_.end());
        kill_.clear();
    }

    void cancel(std::shared_ptr<Request> r);

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    Gateway* gateway_;

    std::pmr::list<std::shared_ptr<Request>> queue_;
    std::pmr::unsynchronized_pool_resource   memory_pool_;

    std::vector<std::shared_ptr<Request>> kill_;

    std::mutex              mutex_;
    std::condition_variable cv_;

    bool closed_{};
};

}  // namespace turbomind
