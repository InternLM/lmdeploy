// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>

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

    enum OutType
    {
        kNone       = 0,
        kAll        = 1,
        kGeneration = 2
    };
    int output_last_hidden_state = 0;
    int output_logits            = 0;
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
    os << ", output_hidden_states=" << c.output_last_hidden_state;
    os << ", output_logits=" << c.output_logits;
    os << " }";
    return os;
}

struct SessionParam {
    uint64_t id;

    int step;

    bool start_flag;
    bool end_flag;
    bool kill_flag;
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
    // fast path for accessing common output buffers
    Tensor output_ids;
    Tensor sequence_length;

    std::function<void(int)> end_cb;

    std::atomic<int> cancel_flag;
    bool             is_canceled{};

    std::function<void()> forward_cb;

    std::shared_ptr<AtomicRequestState> state;

    int ec;  // set when disabling conflicting requests

    enum
    {
        kOk       = 0,
        kInvalid  = 1,  // Sequence not exist or both `start` & `stop` (instead of `end`) is set
        kConflict = 2,  // Concurrent requests to the same sequence
        kBusy     = 3,  // Sequence is already running
        kInactive = 4,  // Sequence to `stop` is not active
        kFail     = 5,  // Can't find sequence for `stop` request or internal error during inference
        kTooLong  = 6,  // history + prompt > session_len,
        kFinish   = 7,
        kCancel   = 8,
    };
};

inline void UpdateState(Request& r, int status, int seq_len)
{
    try {
        auto new_state = new RequestState{status, seq_len};
        auto old_state = r.state->exchange(new_state);
        if (!old_state && r.forward_cb) {
            r.forward_cb();
        }
    }
    catch (const std::exception& e) {
        TM_LOG_ERROR("Error invoking callback for (%lu): %s", r.id, e.what());
    }
    catch (...) {
        TM_LOG_ERROR("Unknown error invoking callback for (%lu)", r.id);
    }
}

}  // namespace turbomind
