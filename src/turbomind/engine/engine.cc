// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <numeric>
#include <thread>
#include <unordered_set>

#include "nvtx3/nvToolsExt.h"

#include "src/turbomind/comm/env.h"
#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/engine/engine.h"
#include "src/turbomind/engine/model_executor.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/engine/scheduler.h"

#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/context_token_resource.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/vision_model.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/metrics.h"

#include "src/turbomind/memory/object.h"
#include "src/turbomind/memory/stats.h"

// #include "dbg.h"

namespace turbomind {

using std::shared_ptr;
using std::unique_ptr;
using std::vector;

struct RequestData {
    vector<shared_ptr<Request>> infer;   // incoming inference request
    vector<int>                 cancel;  // canceled indices in current batch
    bool                        abort;
};

template<class Archive>
void serdes(Archive& ar, RequestData& r)
{
    ar& r.infer;
    ar& r.cancel;
    ar& r.abort;
}

struct Engine::Impl {

    using Requests = vector<shared_ptr<Request>>;
    using Signal   = std::function<void()>;

    struct State;

    Impl(EngineParam                  param,
         ObjectAllocator              alloc,
         CacheRegistry                cache_registry,
         LanguageModel                model,
         std::unique_ptr<VisionModel> vision_model,
         Context&                     ctx,
         Gateway&                     gateway,
         int                          device_id,
         int                          queue_id,
         int                          phases);

    void InternalThreadEntry();

    void Validate(Requests& infer_reqs);

    vector<int> GetCanceled();

    void Cancel(vector<int>& indices, vector<Signal>& signals);

    void Accept(const Requests& rs, vector<Signal>& signals);

    void Interrupt(Sequence& c);

    void Retire(State& s);

    // Allocation of memory / compute resources
    void Schedule();

    // Forward-progress guard: fail the head-of-line request on genuine cache OOM
    void FailStalledHeadOfLine(std::vector<Signal>& signals);

    // Initialize batch data from engine-local sequence state
    void Setup(BatchData& d);

    // Sync vars from batch output to engine-local sequence state
    void Update(BatchData& d, std::vector<Signal>& signals);

    void Run(BatchOp op, int phase, Ref<TensorMap> env)
    {
        // Vision sub-graph runs first so its env outputs (image embeddings,
        // mrope tensors) are visible to the language model in the same pass.
        if (vision_model_) {
            vision_model_->Run(op, phase, env);
        }
        model_.Run(op, phase, env);
    }

    void Start()
    {
        internal_thread_ = std::thread(&Impl::InternalThreadEntry, this);
        executor_.Start();
    }

    void Join()
    {
        if (internal_thread_.joinable()) {
            internal_thread_.join();
        }
    }

    void UpdateScheduleMetrics(bool advance_scheduler = false);

    void MaybeLogCacheStats();

    ~Impl();

    const EngineParam param_;

    Gateway& gateway_;

    comm::HostComm& tp_group_;
    comm::HostComm& dp_group_;

    const int tp_rank_;
    const int dp_rank_;
    const int dp_size_;

    const int device_id_;
    const int queue_id_;

    const int async_;

    int& is_warm_up_;

    ObjectAllocator object_allocator_;
    Scheduler       scheduler_;

    Queue<unique_ptr<BatchData>> inbound_;
    Queue<unique_ptr<BatchData>> outbound_;

    LanguageModel                model_;
    std::unique_ptr<VisionModel> vision_model_;  // null for text-only checkpoints
    ModelExecutor                executor_;

    std::thread internal_thread_;

    // int session_len_trunc_;

    shared_ptr<ScheduleMetrics> metrics_;

    int64_t  scheduler_tick_{};
    uint64_t prefix_query_tokens_{};
    uint64_t prefix_hit_tokens_{};

    int      cache_log_interval_ = GetEnv<CACHE_LOG_INTERVAL>();  // read once (GetEnv caches statically)
    uint64_t schedule_counter_   = 0;

    struct State {
        vector<unique_ptr<Sequence>> rc;

        vector<int> perm;  // current  -> previous

        int bs0     = 0;
        int active  = 0;
        int finish  = 0;
        int swapout = 0;

        int size() const noexcept
        {
            return rc.size();
        }
    };

    vector<State> states_;

    struct Data {
    };
    vector<Data> data_;
};

Engine::Impl::~Impl()
{
    TM_LOG_INFO("{}", __PRETTY_FUNCTION__);
    if (cache_log_interval_ && tp_rank_ == 0) {
        TM_LOG_WARN("dp{} cache stats:\n{}", dp_rank_, FormatMemoryStats(object_allocator_.Stats()));
    }
    inbound_.close();
    outbound_.close();
    // Normally TurboMind joins every engine loop before destroying any engine,
    // so this is a no-op. Keep the fallback for partial construction or
    // exception unwinding: destroying a joinable std::thread calls terminate.
    Join();
    executor_ = {};

    for (auto& state : states_) {
        for (auto& cache : state.rc) {
            if (cache) {
                scheduler_.Release(*cache);
                cache.reset();
            }
        }
    }
}

Engine::Impl::Impl(EngineParam                  param,
                   ObjectAllocator              alloc,
                   CacheRegistry                cache_registry,
                   LanguageModel                model,
                   std::unique_ptr<VisionModel> vision_model,
                   Context&                     ctx,
                   Gateway&                     gateway,
                   int                          device_id,
                   int                          queue_id,
                   int                          phases):
    param_{param},
    gateway_{gateway},
    tp_group_{ctx.comm.h_tp_group},
    dp_group_{ctx.comm.h_dp_group},
    tp_rank_{tp_group_->rank()},
    dp_rank_{dp_group_->rank()},
    dp_size_{dp_group_->n_ranks()},
    device_id_{device_id},
    queue_id_{queue_id},
    async_{phases > 1},
    is_warm_up_{*ctx.is_warm_up},
    object_allocator_{std::move(alloc)},
    scheduler_{object_allocator_,
               std::move(cache_registry),
               param_.cache_block_seq_len * param_.attn_cp_size,
               param_.enable_prefix_caching,
               param_.cache_prompt,
               param_.cache_prompt_boundary_skip,
               param_.cache_generation,
               is_warm_up_},
    model_{std::move(model)},
    vision_model_{std::move(vision_model)}
{
    states_.emplace_back();

    for (int i = 0; i < phases; ++i) {
        data_.emplace_back();
    }

    executor_ = ModelExecutor{model_, vision_model_.get(), ctx, device_id_, outbound_, inbound_};

    UpdateScheduleMetrics();

    if (cache_log_interval_ && tp_rank_ == 0) {
        TM_LOG_WARN("dp{} cache stats:\n{}", dp_rank_, FormatMemoryStats(object_allocator_.Stats()));
    }
}

void Engine::Impl::Validate(Requests& infer_reqs)
{
    std::pmr::monotonic_buffer_resource    mbr;
    std::pmr::unordered_map<uint64_t, int> occur(&mbr);

    for (const auto& s : states_) {
        for (int i = 0; i < s.size(); ++i) {
            if (s.rc[i]) {
                ++occur[s.rc[i]->req->id];
            }
        }
    }
    for (const auto& r : infer_reqs) {
        ++occur[r->id];
    }

    for (const auto& r : infer_reqs) {
        if (occur[r->id] > 1) {
            TM_LOG_ERROR("Skip conflicting infer request for ID {}", r->id);
            r->ec = Request::kConflict;
        }
        if (!r->ec && param_.enable_prefix_caching) {
            if (r->step != 0) {
                TM_LOG_ERROR("Skip inconsistent infer request for ID {} step {}: "
                             "prefix caching is incompatible with a nonzero step",
                             r->id,
                             r->step);
                r->ec = Request::kInconsistency;
            }
            else if (r->gen_cfg.output_logits == GenerationConfig::kAll
                     || r->gen_cfg.output_last_hidden_state == GenerationConfig::kAll || r->gen_cfg.return_ppl) {
                TM_LOG_ERROR("Skip inconsistent infer request for ID {}: prefix caching cannot "
                             "output logits/last_hidden_states for all tokens or ppl",
                             r->id);
                r->ec = Request::kInconsistency;
            }
        }
    }

    for (auto& r : infer_reqs) {
        if (r && r->cancel_flag.load(std::memory_order_acquire) == -1) {
            r->ec = Request::kCancel;
        }
    }
}

vector<int> Engine::Impl::GetCanceled()
{
    auto& s = states_.at(0);

    vector<int> idxs;
    for (int i = 0; i < s.size(); ++i) {  // current batch
        const auto& r = s.rc[i];
        if (r && r->req->cancel_flag.load(std::memory_order_acquire) == -1) {
            idxs.push_back(i);
        }
    }
    return idxs;
}

void Engine::Impl::Interrupt(Sequence& c)
{
    Sequence*          p = &c;
    Buffer_<Sequence*> rs{&p, 1, kCPU};
    Run(BatchOp::kDel, -1, TensorMap{{"requests", rs}});

    scheduler_.Release(c);
}

void Engine::Impl::Retire(State& s)
{
    for (auto& p : s.rc) {
        if (!p || !p->retiring || p->inflight != 0) {
            continue;
        }

        Interrupt(*p);
        p.reset();
        ++s.finish;
    }
}

void Engine::Impl::Cancel(vector<int>& indices, vector<Signal>& signals)
{
    auto& s = states_.at(0);
    for (const auto& i : indices) {
        auto& c = TM_CHECK_NOTNULL(s.rc[i]);
        if (c->retiring) {
            continue;
        }

        c->is_canceled = true;
        c->retiring    = true;
        c->done        = true;
        signals.push_back([r = c->req, l = c->seq_len] { UpdateState(*r, Request::kCancel, l); });
    }
}

void Engine::Impl::Accept(const Requests& rs, vector<Signal>& signals)
{
    auto& s = states_.at(0);

    vector<unique_ptr<Sequence>> incoming;
    incoming.reserve(rs.size());

    for (const auto& r : rs) {

        if (r->ec) {
            signals.push_back([r] { UpdateState(*r, r->ec, 0); });
            continue;
        }

        const auto& input_ids = r->inputs.at("input_ids");
        const int   input_len = input_ids.shape(0);

        if (input_len > param_.session_len) {
            signals.push_back([r] { UpdateState(*r, Request::kTooLong, 0); });
            continue;
        }

        /// TODO: force step after prefix matching

        auto c = std::make_unique<Sequence>(r);

        int* token_ids = c->token_ids = r->output_ids.data();
        /// TODO: move this somewhere else
        token_ids = std::copy_n(input_ids.data<int>(), input_len, token_ids);

        c->prompt_len = c->seq_len = token_ids - c->token_ids;  // all known tokens

        int max_seq_len = c->prompt_len + c->gen_cfg.max_new_tokens;
        if (max_seq_len > param_.session_len) {
            max_seq_len = param_.session_len;
            if (tp_rank_ == 0) {
                const int trunc_output_len = max_seq_len - c->prompt_len;
                // clang-format off
                TM_LOG_WARN("ID {}: total sequence length ({} + {}) exceeds `session_len` ({}), `max_new_tokens` is truncated to {}",
                    r->id, c->prompt_len, c->gen_cfg.max_new_tokens, param_.session_len, trunc_output_len);
                // clang-format on
            }
        }
        c->max_seq_len = max_seq_len;

        incoming.push_back(std::move(c));
    }

    Buffer_<Sequence*> buf(incoming.size(), kCPU);
    for (int i = 0; i < incoming.size(); ++i) {
        buf[i] = incoming[i].get();
    }

    // This includes checks from all modules handling `Add` operation
    Run(BatchOp::kAdd, -1, TensorMap{{"requests", buf}});

    for (auto& x : incoming) {
        if (x->status == 0) {
            scheduler_.AdmitPrompt(*x);
            s.rc.push_back(std::move(x));
        }
        else {
            Interrupt(*x);
            signals.push_back([r = x->req, ec = x->status] {  //
                UpdateState(*r, ec, 0);
            });
        }
    }
}

void Engine::Impl::Schedule()
{
    TM_FUNCTION_SCOPE();
    auto& s = states_.at(0);

    vector<Sequence*> eligible;

    vector<int> was_active;
    vector<int> context_length;
    vector<int> orignal_idxs;
    vector<int> inflight_input_len;

    for (int i = 0; i < s.size(); ++i) {
        auto& p = s.rc[i];
        if (!p) {
            continue;
        }
        auto& c = *p;
        if (!c.retiring) {
            eligible.push_back(&c);
            was_active.push_back(c.is_active);
            context_length.push_back(c.seq_len + c.inflight_new_tokens /* plus draft tokens */);
            inflight_input_len.push_back(c.inflight_input_len);
            orignal_idxs.push_back(i);
            c.input_len = c.history_len = 0;
        }
    }

    ScheduleResources resources;
    resources.Add<ForwardTokenResource>(param_.max_forward_token_num);
    resources.Add<ContextTokenResource>(param_.max_context_token_num);

    scheduler_.Schedule(eligible, resources);

    vector<int> idxs(eligible.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    subrange active{idxs.begin(),
                    std::stable_partition(idxs.begin(), idxs.end(), [&](int i) { return eligible[i]->is_active; })};

    // An empty active batch (cache OOM / resource starvation) is handled by
    // FailStalledHeadOfLine, called after Schedule() returns, where request
    // lifecycle and signal emission live (see README forward-progress).

    if (is_warm_up_) {
        // Avoid extra iteration for warm up request in async mode (force inactivate)
        active = {active.begin(), std::stable_partition(active.begin(), active.end(), [&](int i) {
                      return inflight_input_len[i] == 0;
                  })};
    }

    subrange inactive{active.end(), idxs.end()};

    for (auto i : active) {
        eligible[i]->is_active = true;
    }
    for (auto i : inactive) {
        eligible[i]->is_active   = false;
        eligible[i]->input_len   = 0;
        eligible[i]->history_len = 0;
    }

    subrange existing{active.begin(),
                      std::stable_partition(active.begin(), active.end(), [&](int i) { return was_active[i]; })};

    subrange swap_in{existing.end(), active.end()};

    subrange swap_out{inactive.begin(),
                      std::stable_partition(inactive.begin(), inactive.end(), [&](int i) { return was_active[i]; })};

    // |<-- existing -->|<-- swap-in -->|<- swap-out ->|
    // |<----------- active ----------->|<------- inactive ----->|

    for (auto i : swap_in) {
        eligible[i]->autoregres = {};
        eligible[i]->generating = {};
    }

    for (auto i : swap_in) {
        auto& c = *eligible[i];
        if (!c.first_schedule_recorded) {
            c.first_schedule_recorded = true;

            const int64_t cached_tokens = std::clamp<int64_t>(c.history_len, 0, c.prompt_len);
            if (!is_warm_up_ && param_.enable_prefix_caching) {
                prefix_query_tokens_ += c.prompt_len;
                prefix_hit_tokens_ += cached_tokens;
            }

            if (auto& m = c.req->metrics; TM_LIKELY(m)) {
                m->cached_tokens.store(cached_tokens, std::memory_order_relaxed);
                int64_t expected = 0;
                m->scheduled_time.compare_exchange_strong(
                    expected, RequestMetrics::timestamp(), std::memory_order_relaxed);
            }
        }
    }

    for (auto i : existing) {
        auto& c      = *eligible[i];
        c.autoregres = c.generating && c.input_len == 1;
    }

    for (auto i : active) {
        auto& c      = *eligible[i];
        c.generating = c.resume_len + c.inflight_input_len + c.input_len == c.seq_len + c.inflight_new_tokens;
    }

    // move partially prefilled sequences to the back
    subrange partial{
        std::stable_partition(active.begin(), active.end(), [&](int i) { return eligible[i]->generating; }),
        active.end()};

    // dbg(inv);

    vector<unique_ptr<Sequence>> rc;
    vector<int>                  perm;
    rc.reserve(s.size());
    perm.reserve(s.size());
    for (int i = 0; i < idxs.size(); ++i) {
        perm.push_back(orignal_idxs[idxs[i]]);   // inverse map to original indices (curr -> prev)
        rc.push_back(std::move(s.rc[perm[i]]));  // permute the engine-local sequence state
    }
    // Put done sequences to the back, logical blocks need to be updated.
    for (int i = 0; i < s.size(); ++i) {
        if (auto& p = s.rc[i]) {
            perm.push_back(i);
            rc.push_back(std::move(p));
        }
    }

    s.rc.swap(rc);
    s.perm.swap(perm);

    s.bs0 = std::exchange(s.active, active.size());
    if (cache_log_interval_ && schedule_counter_ % cache_log_interval_ == 0) {
        TM_LOG_INFO("dp{} total: {}, eligible: {}, active: {}", dp_rank_, s.size(), eligible.size(), s.bs0);
    }
    s.swapout = swap_out.size();
    s.finish  = 0;
}

void Engine::Impl::FailStalledHeadOfLine(std::vector<Signal>& signals)
{
    auto& s = states_.at(0);

    if (s.active != 0 || is_warm_up_) {
        return;  // work was admitted, or warm-up legitimately forces empty active
    }

    // Nothing was admitted this pass. If no in-flight work remains, no memory
    // will ever be released, so the highest-priority eligible request cannot
    // make progress even with maximum eviction. Fail it with kOutOfMemory: it
    // retires, releases its held cache, and the next request becomes
    // head-of-line (see README forward-progress).
    Sequence* victim = nullptr;
    for (auto& p : s.rc) {
        if (!p) {
            continue;
        }
        if (p->inflight > 0) {
            return;  // in-flight batch will release memory when it completes (transient drain)
        }
        if (!p->retiring && (!victim || p->req->unique_id < victim->req->unique_id)) {
            victim = p.get();  // smallest unique_id == highest priority == root of the OOM
        }
    }

    if (!victim) {
        return;
    }

    TM_LOG_WARN("dp{} ID {}: cache out of memory, no request can be admitted; failing head-of-line request",
                dp_rank_,
                victim->req->id);

    victim->retiring = true;
    victim->done     = true;
    signals.push_back([r = victim->req] { UpdateState(*r, Request::kOutOfMemory, 0); });
}

void Engine::Impl::Setup(BatchData& d)
{
    TM_FUNCTION_SCOPE();
    auto& s = states_.at(0);

    d.bs0 = s.bs0;
    d.bsz = s.active;

    d.perm = {d.bsz, kCPU};
    std::copy_n(s.perm.data(), d.bsz, d.perm.data());

    BatchCopy copy{};

    Buffer_<Sequence*> rs{s.active, kCPU};
    for (int i = 0; i < s.active; ++i) {
        auto* c = TM_CHECK_NOTNULL(s.rc[i].get());
        ++c->inflight;
        rs[i] = c;
    }

    d.restore_copies.clear();
    d.publish_copies.clear();
    {
        const ObjectAllocator& alloc   = scheduler_.allocator();
        auto                   resolve = [&](std::vector<CacheCopy>& in, std::vector<ResolvedCopy>& out) {
            for (const auto& [src, dst] : in) {
                const CacheBlock& cs = *TM_CHECK_NOTNULL(src);
                const CacheBlock& cd = *TM_CHECK_NOTNULL(dst);
                TM_CHECK_NOTNULL(cs.allocation.a);  // validity (resolved allocation) on both ends
                TM_CHECK_NOTNULL(cd.allocation.a);
                TM_CHECK_EQ(cs.object_id, cd.object_id);        // same object => same part layout
                TM_CHECK_EQ(cs.part_count(), cd.part_count());  // both replay-populated to the same layout
                TM_CHECK_EQ(cs.part_count(), alloc.PartCount(cs.object_id));
                for (int p = 0; p < cs.part_count(); ++p) {
                    out.push_back({cs.base(p), cd.base(p), alloc.PartBytes(cs.object_id, p)});
                }
            }
            in.clear();
        };
        for (int i = 0; i < s.active; ++i) {
            auto& c = *s.rc[i];
            resolve(c.restore_copies, d.restore_copies);
            resolve(c.publish_copies, d.publish_copies);
        }
    }

    TensorMap env{{"requests", rs}, {"batch", d.buf()}, {"copy", copy.buf()}};

    Run(BatchOp::kSetup, d.phase, env);

    // dbg(copy);
    copy.Run();

    d.local_token_num.resize(dp_size_);
    d.local_token_num[dp_rank_] = *env.at("token_num").data<int>();
    if (dp_size_ > 1) {
        AllGather(dp_group_, d.local_token_num.data(), 1);
    }
    d.global_token_num = std::accumulate(d.local_token_num.begin(), d.local_token_num.end(), 0);
}

void Engine::Impl::Update(BatchData& b, std::vector<Signal>& signals)
{
    TM_FUNCTION_SCOPE();
    auto& s = states_.at(0);

    BatchCopy copy;

    TensorMap env{{"batch", b.buf()}, {"copy", copy.buf()}};

    // Copy outputs to host buffers
    Run(BatchOp::kFetch, b.phase, env);

    copy.Run();

    core::Context::stream().Sync();

    //
    Run(BatchOp::kUpdate, b.phase, env);

    Buffer_<bool> finished        = env.at("finished").buffer();
    Buffer_<bool> generating      = env.at("generating").buffer();
    Buffer_<int>  output_ids      = env.at("output_ids").buffer();
    Buffer_<int>  sequence_length = env.at("sequence_length").buffer();

    env = {};

    vector<int> perm(s.size());
    if (async_) {
        perm = s.perm;
    }
    else {
        std::iota(perm.begin(), perm.end(), 0);
    }

    for (int i = 0; i < s.size(); ++i) {
        int j = perm[i];
        if (j < b.bsz) {
            auto& c      = *TM_CHECK_NOTNULL(s.rc[i]);
            c.filled_len = generating[j] ? sequence_length[j] - 1 : sequence_length[j];
            if (c.retiring) {
                continue;
            }
            if (generating[j]) {
                c.token_ids[c.seq_len] = output_ids[j];
                c.seq_len              = sequence_length[j];
                if (int new_tokens = c.seq_len - c.tokens.size(); TM_LIKELY(new_tokens)) {
                    c.tokens.insert(c.tokens.end(), c.token_ids + c.seq_len - new_tokens, c.token_ids + c.seq_len);
                }
                if (TM_UNLIKELY(finished[j])) {
                    if (!c.is_canceled) {
                        scheduler_.Finalize(c);
                    }
                    signals.push_back([r = c.req, l = c.seq_len] { UpdateState(*r, Request::kFinish, l); });
                    c.retiring = true;
                    c.done     = true;
                }
                else if (TM_LIKELY(c.req->stream_output)) {
                    signals.push_back([r = c.req, l = c.seq_len] { UpdateState(*r, Request::kOk, l); });
                }
            }
        }
        else {  // new
        }
    }

    // b.rc.clear();

    if (async_) {
        const int size = s.active + s.swapout;
        for (int i = 0; i < size; ++i) {
            auto& c = *s.rc[i];
            if (i < s.active) {
                c.inflight_input_len  = c.input_len;
                c.inflight_new_tokens = c.generating;
            }
            else {
                // Just got swaped-out
                c.inflight_input_len  = 0;
                c.inflight_new_tokens = 0;
            }
        }
    }

    for (int i = 0; i < s.size(); ++i) {
        const int j = perm[i];
        if (j >= b.bsz) {
            continue;
        }

        auto& c = *TM_CHECK_NOTNULL(s.rc[i]);
        TM_CHECK_GT(c.inflight, 0);
        --c.inflight;
    }
}

void Engine::Impl::InternalThreadEntry()
{
    TM_FUNCTION_SCOPE();
    TM_CUDA_CHECK(cudaSetDevice(device_id_));

    auto stream = Stream::create();

    core::ContextGuard ctx{stream, Allocator(kCPU), Allocator(stream, false)};

    unique_ptr<BatchData> d = std::make_unique<BatchData>(0);

    for (unsigned i = 1; i < data_.size(); ++i) {
        inbound_.push(std::make_unique<BatchData>(i));
    }

    while (true) {

        shared_ptr<RequestData> rs;

        auto& st = states_.at(0);

        if (tp_rank_ == 0) {
            rs = std::make_shared<RequestData>();

            const int  n_free   = param_.max_batch_size - st.size() + st.finish;
            const bool blocking = n_free == param_.max_batch_size;

            gateway_.pop(rs->infer, n_free, blocking, rs->abort, dp_group_, queue_id_);

            Validate(rs->infer);

            rs->cancel = GetCanceled();
        }

        if (st.size() - st.finish == 0 && tp_group_->is_same_process()) {
            // Only thread comm has blocking sync
            tp_group_->Sync(true);
        }

        if (tp_group_->n_ranks() > 1) {
            Broadcast(tp_group_, rs, 0);
        }

        if (rs->abort) {
            TM_LOG_INFO("stop requested.");
            break;
        }

        vector<Signal> signals;

        Accept(rs->infer, signals);

        Cancel(rs->cancel, signals);

        gateway_.notify(std::move(signals), tp_rank_ == 0);

        Retire(st);

        int n_active = st.size() - st.finish;

        TM_CHECK_GE(n_active, 0);

        n_active = AllReduce(dp_group_, n_active, comm::RedOp::kSum);

        if (n_active) {

            Schedule();

            FailStalledHeadOfLine(signals);

            UpdateScheduleMetrics(true);

            MaybeLogCacheStats();

            Setup(*d);

            d->ready.Record(core::Context::stream());

            // auto future = (d->promise = {}).get_future();

            outbound_.push(std::move(d));

            if (!inbound_.pop(d)) {
                break;
            }

            // Must assume `d` is not the same one as above
            TM_CHECK_NOTNULL(d);

            core::Context::stream().Wait(d->done);

            Update(*d, signals);

            Retire(st);

            UpdateScheduleMetrics();

            gateway_.notify(std::move(signals), tp_rank_ == 0);

            // if (future.valid()) {
            //     future.get().Sync();
            // }
        }
        else {
            UpdateScheduleMetrics();
        }

        // dbg("=========================================================================");
    }
}

Engine::~Engine() = default;

Engine::Engine()                  = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

Engine::Engine(EngineParam                  param,
               ObjectAllocator              alloc,
               CacheRegistry                cache_registry,
               LanguageModel                model,
               std::unique_ptr<VisionModel> vision_model,
               Context&                     ctx,
               Gateway&                     gateway,
               int                          device_id,
               int                          dp_rank,
               int                          phases):
    impl_{std::make_unique<Impl>(param,
                                 std::move(alloc),
                                 std::move(cache_registry),
                                 std::move(model),
                                 std::move(vision_model),
                                 ctx,
                                 gateway,
                                 device_id,
                                 dp_rank,
                                 phases)}
{
}

void Engine::Start()
{
    return impl_->Start();
}

void Engine::Join()
{
    if (impl_) {
        impl_->Join();
    }
}

void Engine::Impl::MaybeLogCacheStats()
{
    if (cache_log_interval_ <= 0 || tp_rank_ != 0) {
        return;  // disabled, or non-primary TP rank (avoid duplicate lines)
    }
    if (++schedule_counter_ % static_cast<uint64_t>(cache_log_interval_) != 0) {
        return;
    }
    TM_LOG_WARN("dp{} cache stats:\n{}", dp_rank_, FormatMemoryStats(object_allocator_.Stats()));
}

void Engine::Impl::UpdateScheduleMetrics(bool advance_scheduler)
{
    if (advance_scheduler) {
        ++scheduler_tick_;
    }

    const auto& state = states_.at(0);

    int                                   total_seqs  = 0;
    int                                   active_seqs = 0;
    std::unordered_set<const CacheBlock*> active_blocks;
    for (const auto& p : state.rc) {
        if (!p || p->retiring) {
            continue;
        }
        ++total_seqs;
        if (!p->is_active) {
            continue;
        }
        ++active_seqs;
        for (const CacheBlock* block : p->involved_blocks) {
            if (is_valid(block)) {
                active_blocks.insert(block);
            }
        }
    }

    const MemoryStats memory = object_allocator_.Stats();
    TM_CHECK_LE(active_blocks.size(), memory.live_allocations);

    auto m           = std::make_shared<ScheduleMetrics>();
    m->total_seqs    = total_seqs;
    m->active_seqs   = active_seqs;
    m->waiting_seqs  = total_seqs - active_seqs;
    m->total_blocks  = static_cast<int64_t>(memory.live_allocations);
    m->active_blocks = static_cast<int64_t>(active_blocks.size());
    m->cached_blocks = m->total_blocks - m->active_blocks;
    m->free_blocks   = 0;
    m->cache_usage   = memory.region_bytes ? static_cast<double>(memory.live_bytes) / memory.region_bytes : 0.;
    m->prefix_cache_hit_rate =
        prefix_query_tokens_ ? static_cast<double>(prefix_hit_tokens_) / prefix_query_tokens_ : 0.;
    m->scheduler_tick = scheduler_tick_;

    std::atomic_store_explicit(&metrics_, std::move(m), std::memory_order_release);
}

shared_ptr<ScheduleMetrics> Engine::GetScheduleMetrics()
{
    return std::atomic_load_explicit(&impl_->metrics_, std::memory_order_acquire);
}

}  // namespace turbomind
