// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/engine/request_queue.h"
#include "src/turbomind/engine/signal_buffer.h"

namespace turbomind {

class Gateway {
public:
    Gateway(int size, std::function<std::shared_ptr<void>()> ctx_factory);

    void shutdown();

    void push(std::shared_ptr<Request> r);

    void pop(std::vector<std::shared_ptr<Request>>& infer_reqs,
             unsigned                               max_infer,
             bool                                   blocking,
             bool&                                  abort,
             comm::HostComm&                        dp_group,
             int                                    qid);

    void cancel(std::shared_ptr<Request> r);

    void notify(std::vector<Signal> signals, bool pred = true);

    void set_threshold(int value)
    {
        TM_LOG_INFO("set threshold {} -> {}", dp_thr_, value);
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

    std::atomic<uint32_t> next_;
};

}  // namespace turbomind
