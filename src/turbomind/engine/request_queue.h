// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <atomic>
#include <condition_variable>
#include <list>
#include <memory_resource>
#include <mutex>

#include "src/turbomind/engine/request.h"

namespace turbomind {

class RequestQueue {
public:
    explicit RequestQueue(std::atomic<uint64_t>* flag): flag_{flag}, queue_{&pool_} {}

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

    int try_pop(std::vector<std::shared_ptr<Request>>& rs, int max_rs_size, int max_count)
    {
        std::lock_guard lock{mutex_};

        auto it = queue_.begin();
        int  count{};
        while (rs.size() < max_rs_size && count < max_count && it != queue_.end()) {
            if ((*it)->session.start_flag) {
                rs.push_back(std::move(*it));
                ++count;
                auto tmp = it;
                ++it;
                queue_.erase(tmp);
            }
            else {
                ++it;
            }
        }

        return count;
    }

    bool pop(std::vector<std::shared_ptr<Request>>& infer_reqs,
             std::vector<std::shared_ptr<Request>>& kill_reqs,
             unsigned                               max_infer,
             bool                                   blocking,
             bool&                                  abort)
    {
        std::unique_lock lock{mutex_};

        ++expected_;

        if (blocking) {
            cv_.wait(lock, [this] {
                return !(queue_.empty() && kill_.empty())                      //
                       || flag_->load(std::memory_order_relaxed) == expected_  //
                       || closed_;
            });
        }
        if (closed_) {
            abort = true;
            return false;
        }

        bool is_first = false;
        // Update the flag of current sync DP group
        if (auto old = flag_->exchange(expected_); old < expected_) {
            is_first = true;
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

        return is_first;
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
    std::atomic<uint64_t>* flag_;
    uint64_t               expected_{};

    std::atomic<uint64_t> unique_id_{};

    std::pmr::unsynchronized_pool_resource   pool_;
    std::pmr::list<std::shared_ptr<Request>> queue_;

    std::vector<std::shared_ptr<Request>> kill_;

    std::mutex              mutex_;
    std::condition_variable cv_;

    bool closed_{};
};

}  // namespace turbomind
