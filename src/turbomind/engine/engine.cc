// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>

#include "nvtx3/nvToolsExt.h"

#include "src/turbomind/comm/env.h"
#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/engine/engine.h"
#include "src/turbomind/engine/model.h"
#include "src/turbomind/engine/model_executor.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/engine/scheduler.h"

#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/sampling_topp_kernels.h"
#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/context_token_resource.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/speculative/speculative_model.h"
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

    Impl(EngineParam                       param,
         CacheRegistry                     cache_registry,
         std::unique_ptr<LanguageModel>    model,
         std::unique_ptr<VisionModel>      vision_model,
         std::unique_ptr<SpeculativeModel> spec_model,
         Context&                          ctx,
         Gateway&                          gateway,
         int                               device_id,
         int                               queue_id,
         int                               phases);

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

    // Sync vars from batch output to engine-local sequence state. Host batch
    // operations (add, setup, fetch, update, del) run on this thread through
    // the component container's generic fanout; device operations are the
    // executor's device-bracket steps.
    void Update(BatchData& d, std::vector<Signal>& signals);

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

    Buffer_<uint8_t> symm_buf_;

    // The served model, constructed once here at the composition root. It
    // owns the target, vision, and speculative models; the executor drives it
    // by reference.
    Model model_;

    std::unique_ptr<Scheduler> scheduler_;

    Queue<unique_ptr<BatchData>> inbound_;
    Queue<unique_ptr<BatchData>> outbound_;

    ModelExecutor executor_;

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
                scheduler_->Release(*cache);
                cache.reset();
            }
        }
    }
}

Engine::Impl::Impl(EngineParam                       param,
                   CacheRegistry                     cache_registry,
                   std::unique_ptr<LanguageModel>    model,
                   std::unique_ptr<VisionModel>      vision_model,
                   std::unique_ptr<SpeculativeModel> spec_model,
                   Context&                          ctx,
                   Gateway&                          gateway,
                   int                               device_id,
                   int                               queue_id,
                   int                               phases):
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
    model_{std::move(model), std::move(vision_model), std::move(spec_model), param_, ctx, phases}
{
    const double cache_ratio = param_.cache_max_block_count;
    TM_CHECK_GT(cache_ratio, 0.) << "object-cache path expects 0 < cache_max_block_count < 1";
    TM_CHECK_LT(cache_ratio, 1.) << "object-cache path no longer accepts cache_max_block_count as a block count";

    states_.emplace_back();
    for (int i = 0; i < phases; ++i) {
        data_.emplace_back();
    }

    executor_ = ModelExecutor{model_, param_, ctx, device_id_, outbound_, inbound_};

    const ModelWeight& target_weights             = model_.target->weights();
    const int          max_verification_positions = param_.spec_method.empty() ? 1 : param_.spec_num_draft_tokens + 1;
    const int          max_logits_rows            = param_.max_batch_size * max_verification_positions;

    if (ctx.comm.d_comm) {
        const int model_tp_size = ctx.comm.h_tp_group->n_ranks();
        TM_CHECK(param_.max_forward_token_num % model_tp_size == 0);

        const core::ssize_t bytes = std::max(
            byte_size(target_weights.data_type,
                      core::ssize_t(param_.max_forward_token_num) * param_.attn_dp_size * target_weights.hidden_units),
            byte_size(target_weights.data_type, core::ssize_t(max_logits_rows) * target_weights.vocab_size_padded));

        auto symm_alloc = GetSymmAllocator(ctx.comm.d_comm);
        symm_buf_       = {bytes, symm_alloc};
    }

    core::Context::stream().Sync();

    size_t free_after_workspaces{}, total_bytes{};
    cudaMemGetInfo(&free_after_workspaces, &total_bytes);
    free_after_workspaces = AllReduce(ctx.comm.h_tp_group, free_after_workspaces, comm::RedOp::kMin);

    size_t transient_verification_bytes{};
    if (model_.spec) {
        const size_t vocab_items = static_cast<size_t>(max_logits_rows) * target_weights.vocab_size_padded;
        const size_t target_head_bytes =
            model_.target->logits_use_workspace() ? 0 : byte_size(target_weights.data_type, vocab_items);

        transient_verification_bytes = target_head_bytes + vocab_items * sizeof(float) + vocab_items * sizeof(int)
                                       + GetTopPSortWorkspaceBytes(max_logits_rows,
                                                                   target_weights.vocab_size,
                                                                   target_weights.vocab_size_padded,
                                                                   core::Context::stream().handle());
        transient_verification_bytes +=
            model_.target->SpeculativeStateJournalBytes(param_.max_batch_size, max_verification_positions);
    }

    TM_CHECK_GE(free_after_workspaces, transient_verification_bytes)
        << "insufficient free memory for transient verification storage";
    const size_t cacheable_bytes = free_after_workspaces - transient_verification_bytes;
    const size_t cache_bytes     = static_cast<size_t>(static_cast<double>(cacheable_bytes) * cache_ratio);

    TM_LOG_INFO("Object cache memory: free after model, components, executor, and shared scratch allocation {:.2f} MB, "
                "transient verification reservation {:.2f} MB",
                free_after_workspaces / (1024. * 1024.),
                transient_verification_bytes / (1024. * 1024.));
    TM_LOG_INFO("Object cache budget: {:.2f} MB from cacheable {:.2f} MB and ratio {:.3f}",
                cache_bytes / (1024. * 1024.),
                cacheable_bytes / (1024. * 1024.),
                cache_ratio);

    Buffer cache_region{static_cast<core::ssize_t>(cache_bytes), data_type_v<int8_t>, core::Context::device_alloc()};
    object_allocator_ = ObjectAllocator{std::move(cache_region)};
    cache_registry.RegisterObjectIds(object_allocator_);

    scheduler_ = std::make_unique<Scheduler>(object_allocator_,
                                             std::move(cache_registry),
                                             param_.cache_block_seq_len * param_.attn_cp_size,
                                             param_.enable_prefix_caching,
                                             param_.cache_prompt,
                                             param_.cache_prompt_boundary_skip,
                                             param_.cache_generation,
                                             param_.session_len,
                                             model_.spec ? &model_.spec->policy() : nullptr,
                                             is_warm_up_);

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
    TensorMap          env{{"requests", rs}};
    model_.Run(BatchOp::kDel, -1, env);

    scheduler_->Release(c);
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
        signals.push_back(MakeRequestSignal(c->req, Request::kCancel, c->seq_len));
    }
}

void Engine::Impl::Accept(const Requests& rs, vector<Signal>& signals)
{
    auto& s = states_.at(0);

    vector<unique_ptr<Sequence>> incoming;
    incoming.reserve(rs.size());

    for (const auto& r : rs) {

        if (r->ec) {
            signals.push_back(MakeRequestSignal(r, r->ec, 0));
            continue;
        }

        const auto& input_ids = r->inputs.at("input_ids");
        const int   input_len = input_ids.shape(0);

        if (input_len > param_.session_len) {
            signals.push_back(MakeRequestSignal(r, Request::kTooLong, 0));
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
    TensorMap env{{"requests", buf}};
    model_.Run(BatchOp::kAdd, -1, env);

    for (auto& x : incoming) {
        if (x->status == 0) {
            scheduler_->AdmitPrompt(*x);
            s.rc.push_back(std::move(x));
        }
        else {
            Interrupt(*x);
            signals.push_back(MakeRequestSignal(x->req, x->status, 0));
        }
    }
}

void Engine::Impl::Schedule()
{
    TM_FUNCTION_SCOPE();
    auto& s = states_.at(0);

    vector<Sequence*> eligible;

    vector<int> was_active;
    vector<int> orignal_idxs;

    for (int i = 0; i < s.size(); ++i) {
        auto& p = s.rc[i];
        if (!p) {
            continue;
        }
        auto& c = *p;
        if (!c.retiring) {
            eligible.push_back(&c);
            was_active.push_back(c.is_active);
            orignal_idxs.push_back(i);
        }
    }

    ScheduleResources resources;
    resources.Add<ForwardTokenResource>(param_.max_forward_token_num);
    // A speculative row is charged its absolute key_capacity_end, which runs up
    // to two proposal windows past the last sequence position; keep the context
    // budget above that so near-limit rounds stay admissible.
    const int context_headroom =
        model_.spec ? 2 * model_.spec->policy().Extent(RoundRequest{0, param_.session_len}).query_rows : 0;
    resources.Add<ContextTokenResource>(param_.max_context_token_num + context_headroom);

    scheduler_->Schedule(eligible, resources);

    vector<int> idxs(eligible.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    subrange active{idxs.begin(),
                    std::stable_partition(idxs.begin(), idxs.end(), [&](int i) { return eligible[i]->is_active; })};

    // An empty active batch (cache OOM / resource starvation) is handled by
    // FailStalledHeadOfLine, called after Schedule() returns, where request
    // lifecycle and signal emission live (see README forward-progress).

    subrange inactive{active.end(), idxs.end()};

    for (auto i : active) {
        eligible[i]->is_active = true;
    }
    for (auto i : inactive) {
        Sequence& c = *eligible[i];
        c.is_active = false;
        if (c.inflight == 0) {
            c.submitted.reset();
        }
    }

    subrange existing{active.begin(),
                      std::stable_partition(active.begin(), active.end(), [&](int i) { return was_active[i]; })};

    subrange swap_in{existing.end(), active.end()};

    subrange swap_out{inactive.begin(),
                      std::stable_partition(inactive.begin(), inactive.end(), [&](int i) { return was_active[i]; })};

    // |<-- existing -->|<-- swap-in -->|<- swap-out ->|
    // |<----------- active ----------->|<------- inactive ----->|

    for (auto i : swap_in) {
        auto& c = *eligible[i];
        if (!param_.enable_metrics || c.first_schedule_recorded || !c.req->metrics) {
            continue;
        }
        c.first_schedule_recorded = true;

        const int64_t cached_tokens = std::clamp<int64_t>(c.submitted->history_len, 0, c.prompt_len);
        if (!is_warm_up_ && param_.enable_prefix_caching) {
            prefix_query_tokens_ += c.prompt_len;
            prefix_hit_tokens_ += cached_tokens;
        }

        auto& m = *c.req->metrics;
        m.cached_tokens.store(cached_tokens, std::memory_order_relaxed);
        int64_t expected = 0;
        m.scheduled_time.compare_exchange_strong(expected, RequestMetrics::timestamp(), std::memory_order_relaxed);
    }

    auto extension_end = std::stable_partition(
        active.begin(), active.end(), [&](int i) { return eligible[i]->submitted->is_extension_candidate(); });

    // Speculative rows precede bootstrap rows inside the extension prefix; decoder
    // partitions rely on the speculative rows forming a leading run.
    std::stable_partition(
        active.begin(), extension_end, [&](int i) { return eligible[i]->submitted->is_verification_row(); });

    std::stable_partition(extension_end, active.end(), [&](int i) { return eligible[i]->submitted->generating; });

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
    signals.push_back(MakeRequestSignal(victim->req, Request::kOutOfMemory, 0));
}

void Engine::Impl::Setup(BatchData& d)
{
    TM_FUNCTION_SCOPE();
    auto& s = states_.at(0);

    d.bs0      = s.bs0;
    d.bsz      = s.active;
    d.symm_buf = symm_buf_;

    d.perm = {d.bsz, kCPU};
    std::copy_n(s.perm.data(), d.bsz, d.perm.data());

    BatchCopy copy{};

    Buffer_<Sequence*> rs{s.active, kCPU};
    d.submitted_frontier_reanchor.resize(d.bsz);
    for (int i = 0; i < s.active; ++i) {
        auto* c = TM_CHECK_NOTNULL(s.rc[i].get());
        ++c->inflight;
        rs[i]                            = c;
        d.submitted_frontier_reanchor[i] = c->submitted->frontier_reanchor;
    }

    d.restore_copies.clear();
    d.publish_copies.clear();
    {
        const ObjectAllocator& alloc   = scheduler_->allocator();
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

    model_.Run(BatchOp::kSetup, d.phase, env);

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
    model_.Run(BatchOp::kFetch, b.phase, env);

    copy.Run();

    core::Context::stream().Sync();

    //
    model_.Run(BatchOp::kUpdate, b.phase, env);

    Buffer_<bool> finished        = env.at("finished").buffer();
    Buffer_<bool> generating      = env.at("generating").buffer();
    Buffer_<int>  sequence_length = env.at("sequence_length").buffer();

    Buffer_<int> output_ids;
    Buffer_<int> selected_span_ids;
    Buffer_<int> accept_len;

    const bool speculative_engine = model_.spec != nullptr;
    const int  K                  = speculative_engine ? model_.spec->policy().max_proposals() + 1 : 0;

    if (speculative_engine) {
        selected_span_ids = env.at("selected_span_ids").buffer();
        accept_len        = env.at("accept_len").buffer();
    }
    else {
        output_ids = env.at("output_ids").buffer();
    }

    Buffer_<int> accepted_draft_count;
    if (const Tensor* tensor = env.try_("accepted_draft_count")) {
        accepted_draft_count = Buffer_<int>{tensor->buffer()};
    }

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
            auto& c                                = *TM_CHECK_NOTNULL(s.rc[i]);
            c.filled_len                           = generating[j] ? sequence_length[j] - 1 : sequence_length[j];
            const bool completed_frontier_reanchor = b.submitted_frontier_reanchor[j];
            if (speculative_engine && completed_frontier_reanchor && scheduler_->registry().has_checkpoint()
                && c.inflight == 1) {
                c.frontier_pos = c.filled_len;
            }
            if (c.retiring) {
                continue;
            }
            if (generating[j]) {
                if (speculative_engine) {
                    const int committed = accept_len[j];
                    std::copy_n(selected_span_ids.data() + j * K, committed, c.token_ids + c.seq_len);
                }
                else {
                    c.token_ids[c.seq_len] = output_ids[j];
                }

                c.seq_len = sequence_length[j];

                if (int new_tokens = c.seq_len - c.tokens.size(); TM_LIKELY(new_tokens)) {
                    c.tokens.insert(c.tokens.end(), c.token_ids + c.seq_len - new_tokens, c.token_ids + c.seq_len);
                }

                if (accepted_draft_count) {
                    const int accepted = accepted_draft_count[j];
                    const int k        = model_.spec->policy().max_proposals();

                    if (accepted >= 0 && param_.model_tp_rank == 0 && c.req->metrics) {
                        auto&            metrics = *c.req->metrics;
                        std::scoped_lock lock(metrics.spec_mutex);

                        ++metrics.num_drafts;
                        metrics.num_draft_tokens += k;
                        metrics.num_accepted_tokens += accepted;

                        for (int pos = 0; pos < accepted; ++pos) {
                            ++metrics.num_accepted_tokens_per_pos[pos];
                        }
                    }
                }

                if (TM_UNLIKELY(finished[j])) {
                    if (!c.is_canceled) {
                        scheduler_->Finalize(c);
                    }
                    signals.push_back(MakeRequestSignal(c.req, Request::kFinish, c.seq_len));
                    c.retiring = true;
                    c.done     = true;
                }
                else if (TM_LIKELY(c.req->stream_output)) {
                    signals.push_back(MakeRequestSignal(c.req, Request::kOk, c.seq_len));
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
                const SubmittedRow& row = *c.submitted;
                c.inflight_input_len    = row.inflight_input_delta;
                c.inflight_new_tokens   = row.inflight_new_delta;
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

Engine::Engine(EngineParam                       param,
               CacheRegistry                     cache_registry,
               std::unique_ptr<LanguageModel>    model,
               std::unique_ptr<VisionModel>      vision_model,
               std::unique_ptr<SpeculativeModel> spec_model,
               Context&                          ctx,
               Gateway&                          gateway,
               int                               device_id,
               int                               queue_id,
               int                               phases):
    impl_{std::make_unique<Impl>(param,
                                 std::move(cache_registry),
                                 std::move(model),
                                 std::move(vision_model),
                                 std::move(spec_model),
                                 ctx,
                                 gateway,
                                 device_id,
                                 queue_id,
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

    int total_seqs  = 0;
    int active_seqs = 0;
    for (const auto& p : state.rc) {
        if (!p || p->retiring) {
            continue;
        }
        ++total_seqs;
        if (!p->is_active) {
            continue;
        }
        ++active_seqs;
    }

    const MemoryUsage memory = object_allocator_.Usage();

    auto m          = std::make_shared<ScheduleMetrics>();
    m->total_seqs   = total_seqs;
    m->active_seqs  = active_seqs;
    m->waiting_seqs = total_seqs - active_seqs;
    m->cache_usage  = memory.region_bytes ? static_cast<double>(memory.live_bytes) / memory.region_bytes : 0.;
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
