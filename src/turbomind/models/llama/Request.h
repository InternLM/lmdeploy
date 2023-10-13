// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/utils/Tensor.h"
#include <condition_variable>
#include <cstdint>
#include <future>
#include <limits>
#include <queue>
#include <unordered_map>

namespace turbomind {

struct Request {
    uint64_t id;
    bool     start_flag;
    bool     end_flag;
    bool     stop_flag;

    // per rank inputs/outputs
    std::vector<TensorMap> inputs;
    std::vector<TensorMap> outputs;

    using Callback = std::function<void(std::unordered_map<std::string, Tensor>*)>;
    Callback stream_cb;

    enum
    {
        kInvalid  = 1,
        kConflict = 2,
        kBusy     = 3,
        kInactive = 4,
        kFail     = 5
    };
    std::promise<int> signal;
};

class RequestQueue {
public:
    std::vector<std::future<int>> enqueue(std::vector<std::shared_ptr<Request>> requests)
    {
        std::vector<std::future<int>> futures;
        futures.reserve(requests.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (closed_) {
                throw std::runtime_error("Queue is closed");
            }

            for (auto& r : requests) {
                futures.push_back(r->signal.get_future());
                if (r->stop_flag) {
                    stop_queue_.push(std::move(r));
                }
                else {
                    infer_queue_.push(std::move(r));
                }
            }
        }
        cv_.notify_one();
        return futures;
    }

    void dequeue(std::vector<std::shared_ptr<Request>>& stop_requests,
                 std::vector<std::shared_ptr<Request>>& infer_requests,
                 unsigned                               max_infer_count,
                 bool                                   blocking)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (blocking) {
            cv_.wait(lock, [this] { return !(stop_queue_.empty() && infer_queue_.empty() && closed_ == false); });
        }

        stop_requests.clear();
        while (!stop_queue_.empty()) {
            stop_requests.push_back(std::move(stop_queue_.front()));
            stop_queue_.pop();
        }

        infer_requests.clear();
        while (!infer_queue_.empty() && infer_requests.size() < max_infer_count) {
            infer_requests.push_back(std::move(infer_queue_.front()));
            infer_queue_.pop();
        }
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }

private:
    std::queue<std::shared_ptr<Request>> stop_queue_;
    std::queue<std::shared_ptr<Request>> infer_queue_;
    std::mutex                           mutex_;
    std::condition_variable              cv_;
    bool                                 closed_ = false;
};

}  // namespace turbomind
