// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/interval.h"
#include "src/turbomind/engine/block.h"
#include "src/turbomind/engine/multimodal_input.h"
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

    bool return_ppl = false;

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
    int      step;
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

    int step;  // KV/output offset (replaces SessionParam session; start/end/kill removed)

    GenerationConfig gen_cfg;

    bool stream_output;

    // reference to IO tensors
    TensorMap inputs;
    TensorMap outputs;
    // TODO: update serdes to support multiple nodes inference
    std::shared_ptr<multimodal::Input> mm_inputs;
    // fast path for accessing common output buffers
    Tensor_<int> output_ids;
    Tensor_<int> sequence_length;

    std::atomic<int> cancel_flag;

    std::function<void()> forward_cb;

    std::shared_ptr<AtomicRequestState> state;

    std::shared_ptr<RequestMetrics> metrics;

    int ec = 0;  // set when disabling conflicting requests

    enum
    {
        kOk            = 0,
        kInvalid       = 1,  // Malformed request (e.g. invalid input embeddings) or routing failure
        kConflict      = 2,  // Concurrent requests to the same sequence id
        kFail          = 5,  // Internal error during inference
        kTooLong       = 6,  // history + prompt > session_len
        kFinish        = 7,
        kCancel        = 8,
        kInconsistency = 9,  // Prefix caching incompatible with nonzero step or all-token logits/hidden-state output
        kNoQueue       = 10,
        kOutOfMemory   = 11,
    };

    std::shared_ptr<xgrammar::CompiledGrammar> grammar;
    std::shared_ptr<xgrammar::GrammarMatcher>  matcher;
};

void UpdateState(Request& r, int status, int seq_len);

struct Sequence;

struct MultiModalData;  // defined in models/vision_model.h

// A scheduler-planned device copy between two cache blocks of the same
// category. Resolved to pointers on the engine thread at setup and executed
// as a whole-object copy by the model executor.
struct CacheCopy {
    int src{};
    int dst{};
};

// What set this pass's resume_len. resume_len is a single number, produced by
// whichever mechanism reached the highest skip position in Scheduler::Resume().
// Observability-only; the scheduler stays category-agnostic.
enum class ResumeSource
{
    kNone = 0,    // resume_len == 0, nothing skipped
    kPrefix,      // contiguous valid prefix-category cache (no checkpoint category)
    kFrontier,    // request's own checkpoint frontier (no restore copy)
    kCheckpoint,  // restored a published block checkpoint into the frontier
    kFork,        // extended from a forked sibling's prefix node
};

// Unlike `Request` which is shared by all local TP ranks, each rank has its own `Sequence`.
struct Sequence {

    std::shared_ptr<Request> req;

    const GenerationConfig& gen_cfg;

    explicit Sequence(std::shared_ptr<Request> r): req{std::move(r)}, gen_cfg{req->gen_cfg} {}

    int status = Request::kOk;

    // These members may be opaque handles from individual modules (pointers to forward declared types), but we tend to
    // keep it simple as long as the complexity is manageable

    int* token_ids = nullptr;  // currently the `output_ids` buf of request

    int step0       = 0;  // set at request init, constant, first prefill step
    int prompt_len  = 0;  // set at request init, constant, first decode step
    int max_seq_len = 0;  // set at request init, constant

    int hidden_states_offset = 0;  // set at request init, constant
    int logits_offset        = 0;  // set at request init, constant

    int seq_len = 0;  // set at request init, updated per step

    int input_len   = 0;  // set at schedule
    int history_len = 0;  // set at schedule from `resume_len`

    bool autoregres = false;  // set at schedule, `seq_len` and `input_ids` taken from the engine
    bool generating = false;  // set at schedule

    bool done = false;  // set at cancel / update, is the request finished / canceled

    bool retiring = false;  // finished/canceled; never schedule again
    int  inflight = 0;      // submitted executor batches containing this request

    int generation_token_ids_row    = -1;  // owned by Generation, allocated lazily
    int generation_random_state_row = -1;  // owned by Generation, allocated lazily

    int inflight_input_len  = 0;  // submitted input tokens not yet reflected into filled_len
    int inflight_new_tokens = 0;  // submitted generated tokens not yet reflected into seq_len

    float rope_base = 0.f;

    Interval output_hidden_states;
    Interval output_logits;

    ////////////////////////// Engine-local execution state ///////////////////////////

    std::vector<BlockHandle> block_ids;  // logical (each holds one request ref)

    std::vector<int> alloc_cache_ids;     // cache ids needing allocation this schedule pass
    std::vector<int> involved_cache_ids;  // cache ids stamped for eviction protection (= required alloc set);
                                          // persistent across Continue, rebuilt by Resume

    std::vector<CacheCopy> restore_copies;  // run before BatchOp::kPrepare
    std::vector<CacheCopy> publish_copies;  // run after BatchOp::kUnprep

    int resume_len = 0;  // prefix length every stateful module agrees can be skipped
    int filled_len = 0;  // prefix state actually produced by the latest completed forward

    int readonly_block_num = 0;  // leading block_ids reused read-only (no KV re-write)

    // Prefix-cache logging only; never read by scheduling/admission logic.
    int          matched_blocks = 0;                    // set at Accept: leading prompt blocks found in trie
    bool         resuming       = false;                // transient: planned by Resume() this pass
    ResumeSource resume_source  = ResumeSource::kNone;  // transient: mechanism that set resume_len

    int           frontier_cache_id = 0;        // checkpoint working state for the next forward
    int           frontier_pos      = 0;        // sequence position the frontier corresponds to
    int           publish_cache_id  = 0;        // reserved slot for the next checkpoint publication
    LogicalBlock* publish_target    = nullptr;  // logical block selected for publication this pass
    int           publish_end       = 0;        // sequence position of the pending publication
    int           last_ckpt_pos     = 0;        // end of the last published checkpoint
    bool prompt_boundary_node = false;  // a reusable prompt-boundary partial node exists; when the boundary policy
                                        // admits the publish, the producer clamps its forward to prompt_len-1 to
                                        // populate the node's KV (and publish a checkpoint when the model is recurrent)
    int prompt_boundary_publish = -1;   // cached CacheBoundaryPolicy decision: -1 undecided, 0 no, 1 yes

    std::vector<int> tokens;

    std::vector<Tensor> input_embeds;
    std::vector<int>    input_embeds_offsets;

    // persistent per-sequence vision features (qwen3.5-vit, W1)
    std::vector<std::shared_ptr<MultiModalData>> multimodal_inputs;

    bool is_active   = false;
    bool is_canceled = false;

    // get_ppl / CE-loss (W2)
    Interval       input_ce_loss;
    Buffer_<float> ce_loss;  // device, size 1; rank-0 CE-loss accumulator.
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
    ar & g.return_ppl;
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
    ar & r.step;
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

class Resource {
public:
    virtual ~Resource() = default;

    virtual int  Test(const Sequence& s) const noexcept = 0;
    virtual void Commit(const Sequence& s) noexcept     = 0;
};

class ScheduleResources final: public Resource {
public:
    template<class T, class... Args>
    T& Add(Args&&... args)
    {
        auto  resource = std::make_unique<T>(std::forward<Args>(args)...);
        auto& ref      = *resource;
        resources_.push_back(std::move(resource));
        return ref;
    }

    int Test(const Sequence& s) const noexcept override
    {
        int admitted = std::numeric_limits<int>::max();
        for (const auto& resource : resources_) {
            const int next = resource->Test(s);
            if (next == 0) {
                return 0;
            }
            admitted = std::min(admitted, next);
        }
        return admitted == std::numeric_limits<int>::max() ? 0 : admitted;
    }

    void Commit(const Sequence& s) noexcept override
    {
        for (const auto& resource : resources_) {
            resource->Commit(s);
        }
    }

private:
    std::vector<std::unique_ptr<Resource>> resources_;
};

class ForwardTokenResource final: public Resource {
public:
    explicit ForwardTokenResource(int max_fwd_tokens) noexcept: max_fwd_tokens_{max_fwd_tokens} {}

    int Test(const Sequence& s) const noexcept override
    {
        const int input_len = InputLen(s);
        if (input_len <= 0 || max_fwd_tokens_ <= 0) {
            return 0;
        }
        return std::min(input_len, max_fwd_tokens_);
    }

    void Commit(const Sequence& s) noexcept override
    {
        max_fwd_tokens_ -= s.input_len;
    }

    int remaining_tokens() const noexcept
    {
        return max_fwd_tokens_;
    }

private:
    static int InputLen(const Sequence& s) noexcept
    {
        return s.seq_len + s.inflight_new_tokens - s.inflight_input_len - s.resume_len;
    }

    int max_fwd_tokens_{};
};

}  // namespace turbomind
