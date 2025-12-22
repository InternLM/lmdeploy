// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/interval.h"
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

    enum OutType {
        kNone       = 0,
        kAll        = 1,
        kGeneration = 2
    };
    int output_last_hidden_state = 0;
    int output_logits            = 0;
};

std::ostream& operator<<(std::ostream& os, const GenerationConfig& c);

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

    enum {
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
};

void UpdateState(Request& r, int status, int seq_len);

class Sequence;

// Unlike `Request` which is shared by all local TP ranks, each rank has its own `RequestCache`.
struct RequestCache {
    std::shared_ptr<Request> request;

    const Sequence&         sequence;
    const GenerationConfig& gen_cfg;

    RequestCache(std::shared_ptr<Request> r, const Sequence& s):
        request{std::move(r)}, sequence{s}, gen_cfg{request->gen_cfg}
    {
    }

    int status = Request::kOk;

    // These members may be opaque handles from individual modules (pointers to forward declared types), but we tend to
    // keep it simple as long as the complexity is manageable

    int*     token_ids    = nullptr;  // currently the `output_ids` buf of request
    uint8_t* random_state = nullptr;

    int step0       = 0;  // set at request init, constant, first prefill step
    int prompt_len  = 0;  // set at request init, constant, first decode step
    int max_seq_len = 0;  // set at request init, constant

    int hidden_states_offset = 0;  // set at request init, constant
    int logits_offset        = 0;  // set at request init, constant

    int seq_len = 0;  // set at request init, updated per step

    int input_len   = 0;  // set at schedule (set to `seq.input_len`)
    int history_len = 0;  // set at schedule (set to `seq.cache_len`)

    bool is_decoding = 0;  // `seq_len` and `input_ids` taken from the engine
    bool is_generate = 0;  //

    int alpha = 0;  // pending growth of cache_len (draft_len + input_len)
    int beta  = 0;  // pending growth of seq_len (draft_len + {0,1})

    float rope_base = 0.f;

    Interval output_hidden_states;
    Interval output_logits;
};

}  // namespace turbomind
