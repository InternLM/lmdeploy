// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <memory>
#include <thread>
#include <vector>

#include "src/turbomind/engine/request_queue.h"
#include "src/turbomind/engine/signal_buffer.h"

namespace turbomind {

class Gateway {
public:
    Gateway(std::function<std::shared_ptr<void>()> ctx_factory);

    void shutdown();

    void push(std::vector<std::shared_ptr<Request>> reqs)
    {
        return request_queue_.push(std::move(reqs));
    }

    void pop(std::vector<std::shared_ptr<Request>>& infer_reqs,
             std::vector<std::shared_ptr<Request>>& kill_reqs,
             unsigned                               max_infer_num,
             bool                                   blocking,
             bool&                                  abort)
    {
        return request_queue_.pop(infer_reqs, kill_reqs, max_infer_num, blocking, abort);
    }

    void cancel(std::shared_ptr<Request> req)
    {
        return request_queue_.cancel(std::move(req));
    }

    void kill(std::shared_ptr<Request> req)
    {
        return request_queue_.kill(std::move(req));
    }

    void notify(std::vector<Signal> signals)
    {
        return signal_buffer_.push(std::move(signals));
    }

private:
    void signal_thread_entry() noexcept;

private:
    RequestQueue request_queue_;
    SignalBuffer signal_buffer_;

    std::function<std::shared_ptr<void>()> ctx_factory_;

    std::thread signal_thread_;
};

}  // namespace turbomind
