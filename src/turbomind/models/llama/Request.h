// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <ostream>
#include <queue>

#include "src/turbomind/utils/Tensor.h"

namespace turbomind {

struct GenerationConfig {
    int max_new_tokens = 0;
    int min_new_tokens = 0;

    int   top_k       = 1;
    float top_p       = 0.f;
    float min_p       = 0.f;
    float temperature = 1.f;

    float repetition_penalty = 1.f;

    uint64_t random_seed = 0;

    int output_logprobs = 0;

    // placeholders that are not implemented yet
    bool output_hidden_states = false;
    bool output_logits        = false;
};

inline std::ostream& operator<<(std::ostream& os, const GenerationConfig& c)
{
    os << "GenerationConfig { ";
    os << "max_new_tokens=" << c.max_new_tokens;
    os << ", min_new_tokens=" << c.min_new_tokens;
    os << ", top_p=" << c.top_p;
    os << ", top_k=" << c.top_k;
    os << ", min_p=" << c.min_p;
    os << ", temperature=" << c.temperature;
    os << ", repetition_penalty=" << c.repetition_penalty;
    os << ", random_seed=" << c.random_seed;
    os << ", output_logprobs=" << c.output_logprobs;
    os << ", output_hidden_states=" << c.output_hidden_states;
    os << ", output_logits=" << c.output_logits;
    os << " }";
    return os;
}

struct SessionParam {
    uint64_t id;

    int step;

    bool start_flag;
    bool end_flag;
    bool stop_flag;
};

struct RequestState {
    int status;
    int seq_len;
};

struct AtomicRequestState {

    std::atomic<RequestState*> data_;

    static_assert(std::atomic<RequestState*>::is_always_lock_free);

    ~AtomicRequestState()
    {
        auto data = exchange(nullptr);
    }

    std::unique_ptr<RequestState> exchange(RequestState* data)
    {
        return std::unique_ptr<RequestState>{data_.exchange(data, std::memory_order_acq_rel)};
    }
};

struct Request {
    uint64_t id;         // sequence id
    uint64_t unique_id;  // monotonic increasing

    SessionParam     session;
    GenerationConfig gen_cfg;

    bool stream_output;

    // reference to IO tensors
    TensorMap inputs;
    TensorMap outputs;

    std::function<void(int)> cancel_cb;
    std::function<void(int)> end_cb;

    std::function<void()> forward_cb;

    std::shared_ptr<AtomicRequestState> state;

    enum {
        kOk       = 0,
        kInvalid  = 1,  // Sequence not exist or both `start` & `stop` (instead of `end`) is set
        kConflict = 2,  // Concurrent requests to the same sequence
        kBusy     = 3,  // Sequence is already running
        kInactive = 4,  // Sequence to `stop` is not active
        kFail     = 5,  // Can't find sequence for `stop` request or internal error during inference
        kTooLong  = 6,  // history + prompt > session_len,
        kFinish   = 7,
    };
};

class RequestQueue {
public:
    void enqueue(std::vector<std::shared_ptr<Request>> requests)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (closed_) {
                throw std::runtime_error("Queue is closed");
            }

            for (auto& r : requests) {
                // futures.push_back(r->signal.get_future());
                if (r->session.stop_flag) {
                    stop_queue_.push(std::move(r));
                }
                else {
                    infer_queue_.push(std::move(r));
                }
            }
        }
        cv_.notify_one();
    }

    void dequeue(std::vector<std::shared_ptr<Request>>& stop_requests,
                 std::vector<std::shared_ptr<Request>>& infer_requests,
                 unsigned                               max_infer_count,
                 bool                                   blocking,
                 bool&                                  abort)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (blocking) {
            cv_.wait(lock, [this] { return !(stop_queue_.empty() && infer_queue_.empty()) || closed_; });
            if (closed_) {
                abort = true;
                return;
            }
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
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    std::queue<std::shared_ptr<Request>> stop_queue_;
    std::queue<std::shared_ptr<Request>> infer_queue_;
    std::mutex                           mutex_;
    std::condition_variable              cv_;
    bool                                 closed_{false};
};

}  // namespace turbomind
