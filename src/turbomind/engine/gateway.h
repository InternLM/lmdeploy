// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/engine/request_queue.h"
#include "src/turbomind/engine/signal_buffer.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

class SequenceBinding {
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
    Gateway(int size, std::function<std::shared_ptr<void>()> ctx_factory);

    void shutdown();

    void push(std::shared_ptr<Request> r);

    void pop(std::vector<std::shared_ptr<Request>>& infer_reqs,
             std::vector<std::shared_ptr<Request>>& kill_reqs,
             unsigned                               max_infer,
             bool                                   blocking,
             bool&                                  abort,
             comm::HostComm&                        dp_group,
             int                                    qid);

    void cancel(std::shared_ptr<Request> r);

    void kill(std::shared_ptr<Request> r);

    void notify(std::vector<Signal> signals, bool pred = true);

    void set_threshold(int value)
    {
        TM_LOG_INFO("set threshold %d -> %d", dp_thr_, value);
        dp_thr_ = value;
    }

private:
    void signal_thread_entry() noexcept;

private:
    const int size_;

    int dp_thr_;

    std::vector<std::unique_ptr<RequestQueue>> queues_;

    std::function<std::shared_ptr<void>()> ctx_factory_;

    SignalBuffer signal_buffer_;
    std::thread  signal_thread_;

    SequenceBinding binding_;

    std::atomic<uint32_t> next_;
};

}  // namespace turbomind
