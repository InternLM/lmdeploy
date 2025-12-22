// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include "nvtx3/nvToolsExt.h"

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/engine/engine.h"
#include "src/turbomind/engine/model_executor.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/core/copy.h"
#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/metrics.h"

// #include "dbg.h"

namespace turbomind {

using std::shared_ptr;
using std::unique_ptr;
using std::vector;

struct RequestData {
    vector<shared_ptr<Request>> infer;  // incoming inference request
    vector<shared_ptr<Request>> kill;   // incoming kill request

    vector<int> cancel;  // canceled indices in current batch
    bool        abort;
};

struct Engine::Impl {

    using Requests = vector<shared_ptr<Request>>;
    using Signal   = std::function<void()>;

    Impl(DataType      dtype,
         EngineParam   param,
         LanguageModel model,
         Context&      ctx,
         Gateway&      gateway,
         int           device_id,
         int           dp_rank,
         int           phases);

    void CreateSequenceManager();

    void InternalThreadEntry();

    void Validate(Requests& infer_rs, Requests& kill_rs);

    void Kill(const Requests& rs, vector<Signal>& signals);

    vector<int> GetCanceled();

    void Cancel(vector<int>& indices, vector<Signal>& signals);

    void Accept(const Requests& rs, vector<Signal>& signals);

    Signal Interrupt(unique_ptr<RequestCache> c, int status);

    // Allocation of memory / compute resources
    void Schedule();

    // intiailize RC from `Sequence`
    void Setup(BatchData& d);

    // Sync vars from batch output to RC
    void Update(const BatchData& d, std::vector<Signal>& signals);

    void Run(BatchOp op, int phase, Ref<TensorMap> env)
    {
        model_.Run(op, phase, env);
    }

    void Start()
    {
        internal_thread_ = std::thread(&Impl::InternalThreadEntry, this);
        executor_.Start();
    }

    ~Impl();

    const DataType    dtype_;
    const EngineParam param_;

    Gateway& gateway_;

    comm::HostComm& tp_group_;
    comm::HostComm& dp_group_;

    const int tp_rank_;
    const int dp_rank_;

    const int device_id_;

    const int async_;

    unique_ptr<SequenceManager> seq_mgr_;

    Queue<unique_ptr<BatchData>> inbound_;
    Queue<unique_ptr<BatchData>> outbound_;

    LanguageModel model_;
    ModelExecutor executor_;

    std::thread internal_thread_;

    int session_len_trunc_;

    ScheduleMetrics metrics_;
    std::mutex      metrics_mutex_;

    struct State {
        vector<unique_ptr<RequestCache>> rc;
        vector<int>                      perm;

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

    struct Data {};
    vector<Data> data_;

    // staging buffers
    Buffer_<void*> block_ptrs_buf_;
    Buffer_<int>   block_ptrs_offsets_buf_;
};

Engine::Impl::~Impl()
{
    TM_LOG_INFO(__PRETTY_FUNCTION__);
    inbound_.close();
    outbound_.close();
    if (internal_thread_.joinable()) {
        internal_thread_.join();
    }
    executor_ = {};
}

Engine::Impl::Impl(DataType      dtype,
                   EngineParam   param,
                   LanguageModel model,
                   Context&      ctx,
                   Gateway&      gateway,
                   int           device_id,
                   int           dp_rank,
                   int           phases):
    dtype_{dtype},
    param_{param},
    gateway_{gateway},
    tp_group_{ctx.comm.h_tp_group},
    dp_group_{ctx.comm.h_dp_group},
    tp_rank_{tp_group_->rank()},
    dp_rank_{dp_group_->rank()},
    device_id_{device_id},
    async_{phases > 1},
    model_{std::move(model)}
{
    states_.emplace_back();

    for (int i = 0; i < phases; ++i) {
        data_.emplace_back();
    }

    executor_ = ModelExecutor{model_, device_id_, outbound_, inbound_};

    CreateSequenceManager();  // initializes `session_len_trunc_`

    const ssize_t max_batch_block_num =
        param.max_batch_size * cdiv(session_len_trunc_, model_.attn_param().cache_block_seq_len);
    block_ptrs_buf_         = {max_batch_block_num, kCPUpinned};
    block_ptrs_offsets_buf_ = {param.max_batch_size + 1, kCPUpinned};
}

void Engine::Impl::CreateSequenceManager()
{
    const auto cache_block_seq_len = model_.attn_param().cache_block_seq_len;

    const int dbits = byte_size(dtype_, 8);

    const auto& model_param = model_.model_param();

    const auto quant_policy = model_param.quant_policy;
    const int  elem_bits    = quant_policy ? quant_policy : dbits;

    SequenceManager::BlockConfig block_config{
        (int)model_param.head_dim,
        (int)model_param.kv_head_num,
        cache_block_seq_len,
        elem_bits == dbits ? 0 : dbits,
        elem_bits,
    };

    const auto get_free_size = [&] {  //
        size_t free{}, total{};
        check_cuda_error(cudaMemGetInfo(&free, &total));
        return AllReduce(tp_group_, free, comm::RedOp::kMin);
    };

    seq_mgr_ = std::make_unique<SequenceManager>(model_param.layer_num,
                                                 block_config,
                                                 param_.cache_max_block_count,
                                                 param_.cache_chunk_size,
                                                 param_.enable_prefix_caching,
                                                 tp_rank_,
                                                 core::Context::alloc(kDEVICE),
                                                 get_free_size);

    const auto max_cached_tokens = seq_mgr_->max_block_count() * (size_t)cache_block_seq_len;
    session_len_trunc_           = std::min(max_cached_tokens, (size_t)param_.session_len);
    TM_LOG_INFO("max cached tokens: %lld", max_cached_tokens);
    if (session_len_trunc_ != param_.session_len) {
        TM_LOG_WARNING("`session_len` truncated to %d due to limited KV cache memory", session_len_trunc_);
    }
}

void Engine::Impl::Validate(Requests& infer_reqs, Requests& kill_reqs)
{
    std::pmr::monotonic_buffer_resource    mbr;
    std::pmr::unordered_map<uint64_t, int> occur(&mbr);

    auto count = [&occur](const auto& reqs) {
        for (const auto& r : reqs) {
            ++occur[r->id];
        }
    };

    auto validate = [&](auto& reqs, const char* type) {
        for (const auto& r : reqs) {
            if (occur[r->id] > 1) {
                TM_LOG_ERROR("Skip conflicting %s request for ID %lu", type, r->id);
                r->ec = Request::kConflict;
            }
            if (param_.enable_prefix_caching) {
                if (r->session.step != 0) {
                    // Prefix caching is incompatible with interactive mode
                    TM_LOG_ERROR("Skip inconsistent %s request for ID %lu step %d", type, r->id, r->session.step);
                    r->ec = Request::kInconsistency;
                }
                else if (r->gen_cfg.output_logits == GenerationConfig::kAll
                         || r->gen_cfg.output_last_hidden_state == GenerationConfig::kAll) {
                    // Prefix caching is incompatible with outputting all tokens' logits or last_hidden_state
                    TM_LOG_ERROR("Skip inconsistent %s request for ID %lu. It cannot output logits or "
                                 "last_hidden_states for all tokens",
                                 type,
                                 r->id);
                    r->ec = Request::kInconsistency;
                }
            }
        }
    };

    for (const auto& s : states_) {
        for (int i = 0; i < s.size(); ++i) {
            if (s.rc[i]) {
                ++occur[s.rc[i]->request->id];
            }
        }
    }

    count(kill_reqs);
    count(infer_reqs);

    validate(kill_reqs, "kill");
    validate(infer_reqs, "infer");

    // New requests that never get a chance to start
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
        if (r && r->request->cancel_flag.load(std::memory_order_acquire) == -1) {
            idxs.push_back(i);
        }
    }
    return idxs;
}

void Engine::Impl::Kill(const Requests& kills, vector<Signal>& signals)
{
    for (auto& r : kills) {
        if (r) {
            int ec = r->ec;
            if (!ec) {
                if (!seq_mgr_->Erase(r->id)) {
                    ec = Request::kInvalid;
                }
            }
            signals.push_back([=] { r->end_cb ? r->end_cb(ec) : void(); });
        }
    }
}

Signal Engine::Impl::Interrupt(unique_ptr<RequestCache> c, int status)
{
    auto& s = TM_CHECK_NOTNULL(c)->sequence;
    if (c->request->session.end_flag) {
        seq_mgr_->CacheGeneration(s);
        TM_CHECK(seq_mgr_->Erase(c->request->id));
    }
    else {
        seq_mgr_->UpdateAndSetUnlock(s);
    }
    return [r = c->request, len = c->seq_len, status] { UpdateState(*r, status, len); };
}

void Engine::Impl::Cancel(vector<int>& indices, vector<Signal>& signals)
{
    auto& s = states_.at(0);
    for (const auto& i : indices) {
        signals.push_back(Interrupt(std::move(s.rc[i]), Request::kCancel));
        s.finish += 1;
    }
}

void Engine::Impl::Accept(const Requests& rs, vector<Signal>& signals)
{
    auto& s = states_.at(0);

    const int offset = s.rc.size();
    int       index  = offset;

    vector<unique_ptr<RequestCache>> incoming;
    incoming.reserve(rs.size());

    for (const auto& r : rs) {

        if (r->ec) {
            signals.push_back([r] { UpdateState(*r, r->ec, 0); });
            continue;
        }

        const int input_len = r->inputs.at("input_ids").shape(0);

        if (input_len > session_len_trunc_) {
            signals.push_back([r] { UpdateState(*r, Request::kTooLong, 0); });
            continue;
        }

        auto ptr = r->session.start_flag ? seq_mgr_->Create(r->id) : seq_mgr_->Get(r->id);
        if (!ptr) {
            signals.push_back([r] { UpdateState(*r, Request::kInvalid, 0); });
            continue;
        }

        const int step = [&] {
            int s = r->session.step;
            if (s < 0) {
                s = ptr->tokens.size();
            }
            else if (s > ptr->tokens.size()) {
                if (tp_rank_ == 0) {
                    TM_LOG_WARNING("[ProcessInferRequests] Skipping invalid step (%d) setting for ID %lu", s, ptr->id);
                }
                s = ptr->tokens.size();
            }
            return s;
        }();

        if (step + input_len > session_len_trunc_) {
            signals.push_back([r] { UpdateState(*r, Request::kTooLong, 0); });
            continue;
        }

        auto& seq = *ptr;

        auto c = std::make_unique<RequestCache>(r, seq);

        if (step < seq.tokens.size()) {
            seq.tokens.resize(step);
            seq.cache_len = std::min(seq.cache_len, step);
        }

        c->step0 = step;

        // const int* input_ids = r->inputs.at("input_ids").data<int>();
        auto& input_ids = r->inputs.at("input_ids");

        int* token_ids = c->token_ids = r->output_ids.data();

        /// TODO: move this somewhere else
        token_ids = std::copy_n(seq.tokens.data(), seq.tokens.size(), token_ids);
        token_ids = std::copy_n(input_ids.data<int>(), input_len, token_ids);

        c->prompt_len = c->seq_len = token_ids - c->token_ids;  // all known tokens

        // dbg(seq.cache_len, seq.tokens.size(), input_len, c->seq_len);

        int max_seq_len = c->prompt_len + c->gen_cfg.max_new_tokens;
        if (max_seq_len > session_len_trunc_) {
            max_seq_len = session_len_trunc_;
            if (tp_rank_ == 0) {
                const int trunc_output_len = max_seq_len - c->prompt_len;
                // clang-format off
                TM_LOG_WARNING("[ProcessInferRequests] [%ld] total sequence length (%d + %d) exceeds `session_len` (%d), `max_new_tokens` is truncated to %d",
                    (long)seq.id, c->prompt_len, c->gen_cfg.max_new_tokens, session_len_trunc_, trunc_output_len);
                // clang-format on
            }
        }
        c->max_seq_len = max_seq_len;

        incoming.push_back(std::move(c));
    }

    Buffer_<RequestCache*> buf(incoming.size(), kCPU);
    for (int i = 0; i < incoming.size(); ++i) {
        buf[i] = incoming[i].get();
    }

    Run(BatchOp::kAdd, -1, TensorMap{{"requests", buf}});

    for (auto& x : incoming) {
        if (x->status == 0) {
            s.rc.push_back(std::move(x));
        }
    }
}

void Engine::Impl::Schedule()
{
    auto& s = states_.at(0);

    vector<const Sequence*>  sequences;
    vector<Sequence::Status> status;
    vector<int>              context_length;
    vector<int>              alpha;
    vector<uint64_t>         priorities;
    vector<RequestCache*>    cache;
    vector<int>              inv;

    for (int i = 0; i < s.size(); ++i) {
        // skip invalid positions
        if (const auto& c = s.rc[i]) {
            cache.push_back(c.get());
            sequences.push_back(&c->sequence);
            status.push_back(c->sequence.status);
            priorities.push_back(c->request->unique_id);
            context_length.push_back(c->seq_len + c->beta /* plus draft tokens */);
            alpha.push_back(c->alpha);
            TM_CHECK(c->sequence.status == Sequence::kActive || c->alpha == 0) << c->sequence.status << " " << c->alpha;
            inv.push_back(i);
            c->input_len = c->history_len = 0;
            // dbg(c->request->id, c->seq_len, c->sequence.cache_len, c->alpha, c->beta, c->is_decoding,
            // c->is_generate);
        }
    }

    // dbg("Schedule");

    auto outcome = seq_mgr_->Materialize(
        sequences, context_length, alpha, priorities, param_.max_forward_token_num, param_.max_context_token_num);

    vector<int> idxs(sequences.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    subrange active{idxs.begin(), std::stable_partition(idxs.begin(), idxs.end(), [&](int i) {
                        return sequences[i]->status == Sequence::kActive;  // IS active
                    })};

    subrange inactive{active.end(), idxs.end()};

    TM_CHECK(sequences.empty() || !active.empty()) << "No enough blocks";

    subrange existing{active.begin(), std::stable_partition(active.begin(), active.end(), [&](int i) {
                          return status[i] == Sequence::kActive;  // WAS active in active
                      })};

    subrange swap_in{existing.end(), active.end()};

    subrange swap_out{inactive.begin(), std::stable_partition(inactive.begin(), inactive.end(), [&](int i) {
                          return status[i] == Sequence::kActive;  // WAS active in inactive
                      })};

    // |<-- existing -->|<-- swap-in -->|<- swap-out ->|
    // |<----------- active ----------->|<------- inactive ----->|

    for (auto i : swap_in) {
        cache[i]->is_decoding = {};
        cache[i]->is_generate = {};
    }

    for (auto i : existing) {
        if (cache[i]->is_generate) {
            cache[i]->is_decoding = true;
        }
    }

    for (auto i : active) {
        auto& s = *sequences[i];
        auto& c = *cache[i];
        if (s.cache_len + c.alpha + s.input_length == c.seq_len + c.beta) {
            c.is_generate = true;
        }
    }

    // move partially prefilled sequences to the back
    subrange partial{std::stable_partition(active.begin(), active.end(), [&](int i) { return cache[i]->is_generate; }),
                     active.end()};
    TM_CHECK_LE(partial.size(), 1);

    // dbg(inv);

    vector<unique_ptr<RequestCache>> rc(idxs.size());
    vector<int>                      perm(idxs.size());
    for (int i = 0; i < idxs.size(); ++i) {
        perm[i] = inv[idxs[i]];              // inverse map to original indices
        rc[i]   = std::move(s.rc[perm[i]]);  // warp the request cache
    }
    s.rc.swap(rc);
    s.perm.swap(perm);

    for (auto& c : s.rc) {
        /// ! input_length not updated for inactive seqs
        c->input_len   = c->sequence.input_length;
        c->history_len = c->sequence.cache_len;
        // dbg(c->request->id,
        //     c->seq_len,
        //     c->history_len,
        //     c->input_len,
        //     c->alpha,
        //     c->beta,
        //     c->is_decoding,
        //     c->is_generate);
    }

    s.bs0     = std::exchange(s.active, active.size());
    s.swapout = swap_out.size();
    s.finish  = 0;
}

void Engine::Impl::Setup(BatchData& d)
{
    auto& st = states_.at(0);

    // dbg(d.phase);

    Buffer_<RequestCache*> rc{st.active, kCPU};
    for (int i = 0; i < st.active; ++i) {
        rc[i] = st.rc[i].get();
    }

    block_ptrs_offsets_buf_[0] = 0;
    auto block_ptrs            = block_ptrs_buf_.data();
    for (int i = 0; i < st.active; ++i) {
        const auto& s                  = st.rc[i]->sequence;
        block_ptrs_offsets_buf_[i + 1] = block_ptrs_offsets_buf_[i] + s.blocks.size();
        block_ptrs = std::transform(s.blocks.cbegin(), s.blocks.cend(), block_ptrs, [&](int block_id) {
            return seq_mgr_->GetBlockPtr(block_id);
        });
    }

    d.bs0 = st.bs0;
    d.bsz = st.active;

    d.perm = {d.bsz, kCPU};
    std::copy_n(st.perm.data(), d.bsz, d.perm.data());

    // dbg(d.bs0, d.bsz, d.perm);

    BatchCopy copy{};

    TensorMap env{{"batch", d.buf()},
                  {"copy", copy.buf()},
                  {"requests", rc},
                  {"block_ptrs_offsets", block_ptrs_offsets_buf_},
                  {"block_ptrs", block_ptrs_buf_}};

    Run(BatchOp::kSetup, d.phase, env);

    // dbg(copy);
    copy.Run();

    /// FIXME: all-gather
    d.local_token_num  = {*env.at("local_token_num").data<int>()};
    d.global_token_num = d.local_token_num[0];
}

void Engine::Impl::Update(const BatchData& b, std::vector<Signal>& signals)
{
    auto& s = states_.at(0);

    Buffer_<bool> finished;
    Buffer_<bool> is_generate;
    Buffer_<int>  output_ids;
    Buffer_<int>  sequence_length;

    {
        BatchCopy copy;
        TensorMap env{{"copy", copy.buf()}};
        Run(BatchOp::kFetch, b.phase, env);
        // dbg(copy);
        copy.Run();

        finished        = env.at("finished").buffer();
        is_generate     = env.at("is_generate").buffer();
        output_ids      = env.at("output_ids").buffer();
        sequence_length = env.at("sequence_length").buffer();
    }

    core::Context::stream().Sync();

    Run(BatchOp::kUpdate, -1, TensorMap{});

    // dbg(b.bs0, b.bsz);
    // dbg(core::to_vector<bool>(finished.slice(0, b.bsz)));

    vector<int> perm;
    if (data_.size() > 1) {
        perm = s.perm;
    }
    else {
        perm.resize(b.bsz);
        std::iota(perm.begin(), perm.end(), 0);
    }

    // dbg("Update");

    const int size = s.active + (async_ ? s.swapout : 0);

    for (int i = 0; i < size; ++i) {
        auto& c = *s.rc[i];

        if (const int j = perm[i]; j < b.bsz) {
            if (auto& seq = c.sequence; is_generate[j]) {
                c.token_ids[c.seq_len] = output_ids[j];
                c.seq_len              = sequence_length[j];
                seq.cache_len          = sequence_length[j] - 1;
                if (const int new_tokens = c.seq_len - seq.tokens.size()) {
                    seq.tokens.insert(seq.tokens.end(), c.token_ids + c.seq_len - new_tokens, c.token_ids + c.seq_len);
                }
                if (c.request->stream_output) {
                    signals.push_back([this, r = c.request, l = sequence_length[j]] {  //
                        UpdateState(*r, Request::kOk, l);
                    });
                }
            }
            else {
                seq.cache_len = sequence_length[j];
            }
        }

        // dbg(c.seq_len, c.sequence.cache_len, c.alpha, c.beta, c.is_decoding, c.is_generate);
    }

    if (async_) {
        for (int i = 0; i < size; ++i) {
            auto& c = *s.rc[i];
            if (i < s.active) {
                c.alpha = c.input_len;
                c.beta  = c.is_generate;
            }
            else {
                c.alpha = c.beta = 0;
            }
        }
    }

    for (int i = 0; i < size; ++i) {
        if (const int j = perm[i]; j < b.bsz && finished[j]) {
            signals.push_back(Interrupt(std::move(s.rc[i]), Request::kFinish));
            s.finish += 1;
        }
    }
}

void Engine::Impl::InternalThreadEntry()
{
    check_cuda_error(cudaSetDevice(device_id_));

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
            gateway_.pop(rs->infer,  //
                         rs->kill,
                         param_.max_batch_size - st.size() + st.finish,
                         st.size() - st.finish == 0,
                         rs->abort,
                         dp_rank_);
            Validate(rs->infer, rs->kill);
            rs->cancel = GetCanceled();
        }

        if (st.size() - st.finish == 0) {
            tp_group_->Sync(true);
        }

        Broadcast(tp_group_, rs, 0);

        if (rs->abort) {
            TM_LOG_INFO("[Engine] stop requested.");
            break;
        }

        vector<Signal> signals;

        Kill(rs->kill, signals);

        Accept(rs->infer, signals);

        Cancel(rs->cancel, signals);

        gateway_.notify(std::move(signals), tp_rank_ == 0);

        TM_CHECK_GE(st.size(), st.finish);

        if (st.size() - st.finish) {

            Schedule();

            Setup(*d);

            d->ready.Record(core::Context::stream());

            // auto future = (d->promise = {}).get_future();

            outbound_.push(std::move(d));

            if (!inbound_.pop(d)) {
                break;
            }

            TM_CHECK_NOTNULL(d);

            core::Context::stream().Wait(d->done);

            Update(*d, signals);

            gateway_.notify(std::move(signals), tp_rank_ == 0);

            // if (future.valid()) {
            //     future.get().Sync();
            // }
        }

        // dbg("=========================================================================");
    }
}

Engine::~Engine() = default;

Engine::Engine()                             = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

Engine::Engine(DataType      dtype,
               EngineParam   param,
               LanguageModel model,
               Context&      ctx,
               Gateway&      gateway,
               int           device_id,
               int           dp_rank,
               int           phases):
    impl_{std::make_unique<Impl>(dtype, param, std::move(model), ctx, gateway, device_id, dp_rank, phases)}
{
}

void Engine::WarmUp() {}

void Engine::Start()
{
    return impl_->Start();
}

ScheduleMetrics Engine::GetScheduleMetrics()
{
    if (!impl_->param_.enable_metrics) {
        return {};
    }
    std::lock_guard lock{impl_->metrics_mutex_};
    auto            metrics = impl_->metrics_;
    return metrics;
}

}  // namespace turbomind