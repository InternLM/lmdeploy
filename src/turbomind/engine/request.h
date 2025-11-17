// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <ostream>

#include <xgrammar/xgrammar.h>

#include "src/turbomind/core/core.h"
#include "src/turbomind/utils/metrics.h"

namespace turbomind {

struct GenerationConfig {
    int max_new_tokens = 0;
    int min_new_tokens = 0;

    std::vector<int> eos_ids;  // only support single token id

    std::array<std::vector<int>, 2> stop_ids;  // (token_id, offset)
    std::array<std::vector<int>, 2> bad_ids;

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

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    os << "[";
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, ", "));
    if (!vec.empty()) {
        os.seekp(-2, std::ios_base::end);
    }
    os << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const GenerationConfig& c)
{
    os << "GenerationConfig { ";
    os << "max_new_tokens=" << c.max_new_tokens;
    os << ", min_new_tokens=" << c.min_new_tokens;
    os << ", eos_ids=" << c.eos_ids;
    os << ", stop_ids=[" << c.stop_ids[0] << ", " << c.stop_ids[1] << "]";
    os << ", bad_ids=[" << c.bad_ids[0] << ", " << c.bad_ids[1] << "]";
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
    Tensor_<int> output_ids;
    Tensor_<int> sequence_length;

    std::function<void(int)> end_cb;

    std::atomic<int> cancel_flag;

    std::function<void()> forward_cb;

    std::shared_ptr<AtomicRequestState> state;

    std::shared_ptr<RequestMetrics> metrics;

    int ec;  // set when disabling conflicting requests

    enum
    {
        kOk            = 0,
        kInvalid       = 1,  // Sequence not exist or both `start` & `stop` (instead of `end`) is set
        kConflict      = 2,  // Concurrent requests to the same sequence
        kBusy          = 3,  // Sequence is already running
        kInactive      = 4,  // Sequence to `stop` is not active
        kFail          = 5,  // Can't find sequence for `stop` request or internal error during inference
        kTooLong       = 6,  // history + prompt > session_len,
        kFinish        = 7,
        kCancel        = 8,
        kInconsistency = 9,  // Inconsistent request parameters, e.g. prefix caching is not allowed in interactive mode
    };

    std::shared_ptr<xgrammar::GrammarMatcher> matcher;
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
