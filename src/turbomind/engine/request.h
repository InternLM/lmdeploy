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

namespace xgrammar {
class GrammarMatcher;  // forward declaration
class CompiledGrammar;
}  // namespace xgrammar

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

    int ec = 0;  // set when disabling conflicting requests

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
        kInconsistency = 9,   // Inconsistent request parameters, e.g. prefix caching is not allowed in interactive mode
        kNoQueue       = 10,  // No queue available for submitting the request (in current process)
    };

    std::shared_ptr<xgrammar::CompiledGrammar> grammar;
    std::shared_ptr<xgrammar::GrammarMatcher>  matcher;
};

void UpdateState(Request& r, int status, int seq_len);

class Sequence;

// Unlike `Request` which is shared by all local TP ranks, each rank has its own `RequestCache`.
struct RequestCache {
    std::shared_ptr<Request> req;
    const Sequence*          seq;  // May be NULL in `Update` (seq get erased when req is done)
    const GenerationConfig&  gen_cfg;

    RequestCache(std::shared_ptr<Request> r, const Sequence& s): req{std::move(r)}, seq{&s}, gen_cfg{req->gen_cfg} {}

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

    bool autoregres = false;  // set at schedule, `seq_len` and `input_ids` taken from the engine
    bool generating = false;  // set at schedule

    bool done = false;  // set at cancel / update, is the request finished / canceled

    int alpha = 0;  // pending growth of cache_len (draft_len + input_len)
    int beta  = 0;  // pending growth of seq_len (draft_len + {0,1})

    float rope_base = 0.f;

    Interval output_hidden_states;
    Interval output_logits;
};

template<class Archive>
void serdes(Archive& ar, GenerationConfig& g)
{
    // clang-format off
    ar & g.max_new_tokens;
    ar & g.min_new_tokens;
    ar & g.eos_ids;
    ar & g.stop_ids[0];
    ar & g.stop_ids[1];
    ar & g.bad_ids[0];
    ar & g.bad_ids[1];
    ar & g.top_k;
    ar & g.top_p;
    ar & g.min_p;
    ar & g.temperature;
    ar & g.repetition_penalty;
    ar & g.random_seed;
    ar & g.output_logprobs;
    ar & g.output_last_hidden_state;
    ar & g.output_logits;
    // clang-format on
}

template<class Archive>
void save_req_output(Archive& ar, const TensorMap& map)
{
    // clang-format off
    ar & map.size();
    for (const auto& [k, t] : map) {
        TM_CHECK(t.device().type == kCPU);
        ar & k;
        ar & t.layout();
        ar & t.dtype();
    }
    // clang-format on
}

template<class Archive>
void load_req_output(Archive& ar, TensorMap& map)
{
    // clang-format off
    decltype(map.size()) size;
    ar & size;
    for (int i = 0; i < size; ++i) {
        std::string k;
        Layout      layout;
        DataType    dtype;
        ar & k;
        ar & layout;
        ar & dtype;
        map.emplace(std::move(k), Tensor{layout, dtype, kCPU});
    }
    // clang-format on
}

template<class Archive>
void serdes(Archive& ar, Request& r)
{
    // clang-format off
    ar & r.id;
    ar & r.unique_id;
    ar & r.session;
    ar & r.gen_cfg;
    ar & r.stream_output;
    ar & r.inputs;
    if constexpr(Archive::is_loading) {
        load_req_output(ar, r.outputs);
        r.output_ids      = r.outputs.at("output_ids");
        r.sequence_length = r.outputs.at("sequence_length");
    } else {
        save_req_output(ar, r.outputs);
    }
    ar & r.ec;
    // clang-format on
}

}  // namespace turbomind
