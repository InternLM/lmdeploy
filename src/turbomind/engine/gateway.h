// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "src/turbomind/engine/request_queue.h"
#include "src/turbomind/engine/signal_buffer.h"

namespace turbomind {

class SeqMap {
public:
    int find(uint64_t seq_id)
    {
        std::lock_guard lock{mutex_};
        if (auto it = seq_map_.find(seq_id); it != seq_map_.end()) {
            return it->second;
        }
        return -1;
    }

    void insert(uint64_t seq_id, int rank)
    {
        std::lock_guard lock{mutex_};
        seq_map_.emplace(seq_id, rank);
    }

    void erase(uint64_t seq_id)
    {
        std::lock_guard lock{mutex_};
        seq_map_.erase(seq_id);
    }

private:
    std::mutex                        mutex_;
    std::unordered_map<uint64_t, int> seq_map_;
};

class Gateway {
public:
    Gateway(int n_ranks, std::function<std::shared_ptr<void>()> ctx_factory);

    void shutdown();

    void push(std::shared_ptr<Request> r)
    {
        int rank = -1;

        if (!r->session.start_flag) {
            // route to corresponding rank
            rank = seq_map_.find(r->session.id);
        }
        else {
            rank = next_.fetch_add(1, std::memory_order_relaxed) % n_ranks_;
        }

        if (rank >= 0) {
            queues_[rank]->push({std::move(r)});
        }
        else {
            /// TODO: report failure
        }
    }

    void pop(std::vector<std::shared_ptr<Request>>& infer_reqs,
             std::vector<std::shared_ptr<Request>>& kill_reqs,
             unsigned                               max_infer,
             bool                                   blocking,
             bool&                                  abort,
             int                                    rank)
    {
        infer_reqs.clear();
        kill_reqs.clear();

        queues_[rank]->pop(infer_reqs, kill_reqs, max_infer, blocking, abort);

        if (infer_reqs.size() == max_infer || abort) {
            return;
        }

        while (true) {
            bool success = false;
            for (int i = 0; i < n_ranks_; ++i) {
                int idx = rank + i < n_ranks_ ? rank + i : rank + i - n_ranks_;
                if (std::shared_ptr<Request> r; queues_[idx]->try_pop(r)) {
                    infer_reqs.push_back(std::move(r));
                    success = true;
                    if (infer_reqs.size() == max_infer) {
                        return;
                    }
                }
            }
            if (!success) {
                break;
            }
        }

        if (infer_reqs.empty() && kill_reqs.empty() && blocking) {
            queues_[rank]->pop(infer_reqs, kill_reqs, max_infer, true, abort);
        }
    }

    void cancel(std::shared_ptr<Request> r)
    {
        if (auto rank = seq_map_.find(r->session.id); rank >= 0) {
            queues_[rank]->cancel(std::move(r));
        }
        else {
            /// TOOD: report failure
        }
    }

    void kill(std::shared_ptr<Request> r)
    {
        if (auto rank = seq_map_.find(r->session.id); rank >= 0) {
            queues_[rank]->kill(std::move(r));
        }
        else {
            /// TOOD: report failure
        }
    }

    void notify(std::vector<Signal> signals)
    {
        return signal_buffer_.push(std::move(signals));
    }

    void bind(uint64_t seq_id, int rank)
    {
        seq_map_.insert(seq_id, rank);
    }

    void unbind(uint64_t seq_id)
    {
        seq_map_.erase(seq_id);
    }

private:
    void signal_thread_entry() noexcept;

private:
    const int n_ranks_;

    std::vector<std::unique_ptr<RequestQueueV2>> queues_;

    std::function<std::shared_ptr<void>()> ctx_factory_;

    SignalBuffer signal_buffer_;
    std::thread  signal_thread_;

    SeqMap seq_map_;

    std::atomic<uint32_t> next_;
};

}  // namespace turbomind
