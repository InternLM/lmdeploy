// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "src/turbomind/engine/request.h"
#include "src/turbomind/engine/request_queue.h"
#include "src/turbomind/engine/signal_buffer.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

class SeqId2Rank {
public:
    int find(uint64_t seq_id)
    {
        std::lock_guard lock{mutex_};
        if (auto it = map_.find(seq_id); it != map_.end()) {
            return it->second;
        }
        return -1;
    }

    void bind(const std::vector<uint64_t>& seq_ids, int rank)
    {
        std::lock_guard lock{mutex_};
        for (const auto& x : seq_ids) {
            if (auto [it, success] = map_.emplace(x, rank); !success) {
                TM_LOG_WARNING("[TM][Gateway] Duplicated binding for %lu, %d vs %d", x, rank, it->second);
            }
        }
    }

    void unbind(const std::vector<uint64_t>& seq_ids, int rank)
    {
        std::lock_guard lock{mutex_};
        for (const auto& x : seq_ids) {
            auto it = map_.find(x);
            if (it == map_.end()) {
                TM_LOG_WARNING("[TM][Gateway] No entry found for unbinding %lu, %d", x, rank);
            }
            else if (it->second != rank) {
                TM_LOG_WARNING("[TM][Gateway] Mismatched entry for unbinding %lu, %d vs %d", x, rank, it->second);
            }
            else {
                map_.erase(it);
            }
        }
    }

private:
    std::mutex                        mutex_;
    std::unordered_map<uint64_t, int> map_;
};

class Gateway {
public:
    Gateway(int                                    groups,
            int                                    group_size,
            std::vector<int>                       node_dp_ranks,
            std::function<std::shared_ptr<void>()> ctx_factory);

    void shutdown();

    void push(std::shared_ptr<Request> r)
    {
        int rank = -1;

        if (!r->session.start_flag) {
            // route to corresponding rank
            rank = seqid2rank_.find(r->session.id);
        }
        else if (node_dp_ranks_.size() > 0) {
            rank = next_.fetch_add(1, std::memory_order_relaxed) % node_dp_ranks_.size();
            rank = node_dp_ranks_[rank];
        }

        if (rank >= 0) {
            queues_[rank]->push({std::move(r)});
        }
        else {
            TM_LOG_ERROR("[TM][Gateway] Failed to find a binded queue for %lu", r->session.id);
            notify({[r = std::move(r)] {  //
                UpdateState(*r, Request::kInvalid, 0);
            }});
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

        [&] {
            for (int i = 0; i < size_; ++i) {
                int idx = rank + i < size_ ? rank + i : rank + i - size_;
                if (queues_[idx]->try_pop(infer_reqs, max_infer, idx == rank ? max_infer : 1)) {
                    if (infer_reqs.size() == max_infer) {
                        return;
                    }
                }
            }
        }();

        blocking = blocking && infer_reqs.empty();

        if (queues_[rank]->pop(infer_reqs, kill_reqs, max_infer, blocking, abort)) {
            const int group_id = rank / group_size_;
            // Wake all siblings
            for (int i = group_id * group_size_; i < (group_id + 1) * group_size_; ++i) {
                if (i != rank) {
                    queues_[i]->notify();
                }
            }
        }

        // if (infer_reqs.empty() && kill_reqs.empty()) {
        //     TM_LOG_INFO("[Queue][%d] Wake up with no requests", rank);
        // }

        // Assign a monotonic increasing id for each infer request
        queues_[rank]->assign_unique_ids(infer_reqs);

        // Bind for stateful inference
        std::vector<uint64_t> bind_ids;
        for (const auto& r : infer_reqs) {
            if (r->session.start_flag && !r->session.end_flag) {  // started but not ended
                bind_ids.push_back(r->session.id);
            }
        }
        if (!bind_ids.empty()) {
            seqid2rank_.bind(bind_ids, rank);
        }

        // Unbind for stateful kill
        std::vector<uint64_t> unbind_ids;
        for (const auto& r : kill_reqs) {
            unbind_ids.push_back(r->session.id);
        }
        if (!unbind_ids.empty()) {
            seqid2rank_.unbind(unbind_ids, rank);
        }
    }

    void cancel(std::shared_ptr<Request> r)
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

    void kill(std::shared_ptr<Request> r)
    {
        if (auto rank = seqid2rank_.find(r->session.id); rank >= 0) {
            queues_[rank]->kill(std::move(r));
        }
        else {
            TM_LOG_ERROR("[Gateway] Failed to find a binded queue for %lu", r->session.id);
            notify({[r = std::move(r)] {  //
                UpdateState(*r, Request::kInvalid, 0);
            }});
        }
    }

    void notify(std::vector<Signal> signals)
    {
        return signal_buffer_.push(std::move(signals));
    }

private:
    void signal_thread_entry() noexcept;

private:
    const int size_;
    const int group_size_;

    std::vector<std::unique_ptr<RequestQueue>>          queues_;
    std::vector<std::unique_ptr<std::atomic<uint64_t>>> flags_;
    std::vector<int>                                    node_dp_ranks_;

    std::function<std::shared_ptr<void>()> ctx_factory_;

    SignalBuffer signal_buffer_;
    std::thread  signal_thread_;

    SeqId2Rank seqid2rank_;

    std::atomic<uint32_t> next_;
};

}  // namespace turbomind
