#include "src/turbomind/engine/lmcache.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "src/turbomind/comm/host_comm.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/block.h"
#include "src/turbomind/engine/cache_registry.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/engine/scheduler.h"
#include "src/turbomind/lmcache/lookup.h"
#include "src/turbomind/lmcache/retrieve.h"
#include "src/turbomind/lmcache/store.h"
#include "src/turbomind/memory/object.h"

namespace turbomind {

namespace {

struct PendingRetrieve {
    PendingRetrieve(lmcache::Connector& connector, lmcache::Request request, std::vector<CacheBlock*> pins):
        pins{std::move(pins)}, transfer{connector, std::move(request), core::Context::stream().handle()}
    {
        for (auto* block : this->pins) {
            block->AcquireTransfer();
        }
    }
    ~PendingRetrieve()
    {
        for (auto* block : pins) {
            block->ReleaseTransfer();
        }
    }
    PendingRetrieve(const PendingRetrieve&) = delete;
    PendingRetrieve& operator=(const PendingRetrieve&) = delete;

    std::vector<CacheBlock*> pins;
    lmcache::RetrieveContext transfer;
};

struct RequestSession {
    enum class Phase
    {
        kLookup,
        kRetrieve,
        kReady
    };

    Sequence*                             sequence;
    std::optional<lmcache::LookupContext> lookup;
    std::optional<lmcache::RequestLease>  lease;
    Phase                                 phase{Phase::kLookup};
    std::unique_ptr<PendingRetrieve>      retrieve;
    int                                   next_store_token{};  // Reservation cursor, initially the remote hit length.
    int                                   pending_stores{};

    bool WaitingForRetrieve() const noexcept
    {
        return phase == Phase::kRetrieve && !retrieve;
    }
};

struct PendingStore {
    PendingStore(RequestSession& session, lmcache::Request request, std::vector<CacheBlock*> sources):
        session{session}, request{std::move(request)}, sources{std::move(sources)}
    {
        for (auto* block : this->sources) {
            block->AcquireTransfer();
        }
        ++session.pending_stores;
    }
    ~PendingStore()
    {
        transfer.reset();
        for (auto* block : sources) {
            block->ReleaseTransfer();
        }
        --session.pending_stores;
    }
    PendingStore(const PendingStore&) = delete;
    PendingStore& operator=(const PendingStore&) = delete;

    RequestSession&                      session;
    lmcache::Request                     request;
    std::vector<CacheBlock*>             sources;
    std::optional<lmcache::StoreContext> transfer;
};

using Stores = std::vector<std::unique_ptr<PendingStore>>;

struct StoreBatch {
    Stores stores;
    bool   produced{};
};

}  // namespace

class LmCache::Impl {
public:
    Impl(const std::string& addr,
         const std::string& model_name,
         comm::HostComm&    group,
         const int&         is_warm_up,
         int                block_size,
         int                phases):
        tp_group_{group},
        cache_group_{group->Split(0, group->rank())},
        rank_{group->rank()},
        is_warm_up_{is_warm_up},
        block_size_{block_size},
        batches_(phases)
    {
        lmcache::ConnectorConfig config;
        config.server_addr = addr;
        config.model_name  = model_name;
        config.world_size  = group->n_ranks();
        config.worker_id   = rank_;
        if (rank_ == 0) {
            std::random_device random;
            config.session_id = std::to_string((uint64_t{random()} << 32) | random());
        }
        comm::Broadcast(tp_group_, config.session_id, 0);
        connector_  = std::make_unique<lmcache::Connector>(std::move(config));
        chunk_size_ = connector_->chunk_size();
        TM_CHECK_GT(block_size_, 0);
        TM_CHECK_EQ(chunk_size_ % block_size_, 0) << "LMCache chunk_size must be divisible by the logical block size";
        if (rank_ == 0) {
            TM_LOG_INFO("LMCache enabled: server={}, tp={}, chunk_size={}, block_size={}",
                        addr,
                        group->n_ranks(),
                        chunk_size_,
                        block_size_);
        }
    }

    void Register(void* base, const ObjectAllocator& allocator, const CacheRegistry& registry)
    {
        const int prefix = registry.prefix().object_id();
        TM_CHECK_EQ(allocator.PartCount(prefix), 1);
        pools_ = {{allocator.PartBytes(prefix, 0), block_size_}};
        if (registry.has_checkpoint()) {
            const int checkpoint = registry.checkpoint().object_id();
            for (int part = 0; part < allocator.PartCount(checkpoint); ++part) {
                pools_.push_back({allocator.PartBytes(checkpoint, part), chunk_size_});
            }
        }
        const auto stats = allocator.Stats();
        for (const auto& pool : pools_) {
            TM_CHECK_EQ(stats.page.page_size % pool.part_bytes, 0)
                << "LMCache cache parts must divide the allocator page size";
        }
        const auto storage    = reinterpret_cast<uintptr_t>(stats.page.base);
        const auto allocation = reinterpret_cast<uintptr_t>(base);
        TM_CHECK(base && stats.page.base);
        TM_CHECK_GE(storage, allocation);
        TM_CHECK_LE(storage - allocation, stats.region_bytes);
        lmcache::RegistrationConfig config;
        config.base                 = base;
        config.size                 = stats.region_bytes;
        config.storage_offset_bytes = storage - allocation;
        config.pools                = pools_;
        connector_->Register(config);
    }

    void OnAccepted(Sequence& c)
    {
        // These outputs require prompt forwards; dynamic NTK changes KV for
        // the same tokens according to request length (rope_base).
        if (is_warm_up_ || !c.token_ids || c.rope_base != 0.f || c.gen_cfg.return_ppl
            || c.gen_cfg.output_logits == GenerationConfig::kAll
            || c.gen_cfg.output_last_hidden_state == GenerationConfig::kAll || !c.input_embeds.empty()
            || !c.input_embeds_offsets.empty()
            || std::any_of(c.multimodal_spans.begin(), c.multimodal_spans.end(), [](const auto& span) {
                   return span.fingerprint.empty();
               })) {
            return;
        }
        const auto uid      = c.req->unique_id;
        auto [it, inserted] = sessions_.emplace(uid, RequestSession{&c});
        TM_CHECK(inserted);
        const int end    = Align(c.prompt_len);
        it->second.phase = end == 0 ? RequestSession::Phase::kReady : RequestSession::Phase::kLookup;
        if (end == 0 || rank_ != 0) {
            return;
        }
        lmcache::Request request;
        request.request_id = std::to_string(uid);
        request.end        = end;
        request.token_ids.assign(c.token_ids, c.token_ids + end);
        // Query the whole prefix, including local hits, as LMCacheMPConnector
        // does. Local coverage cannot establish the remote STORE starting point.
        TM_LOG_DEBUG("ID {}: LMCache LOOKUP submitted for tokens [0, {})", c.req->id, end);
        it->second.lookup.emplace(*connector_, std::move(request));
    }

    void Bind(Scheduler& scheduler)
    {
        scheduler_ = &scheduler;
    }

    void PrepareSchedule()
    {
        std::vector<RequestSession*> waiting;
        std::vector<int>             proof;
        for (auto& [uid, session] : sessions_) {
            if (!session.WaitingForRetrieve()) {
                continue;
            }
            auto& c = *session.sequence;
            if (!transfers_healthy_) {
                MakeReady(session);
                continue;
            }
            const int  end   = RetrieveEnd(c);
            const auto local = scheduler_->ProbeResume(c);
            waiting.push_back(&session);
            proof.push_back(local.resume_end >= end);
            proof.push_back(std::min(local.prefix_end, pools_.size() > 1 ? end - chunk_size_ : end));
        }
        if (waiting.empty()) {
            return;
        }
        comm::AllReduce(cache_group_, proof.data(), proof.size(), comm::RedOp::kMin);
        for (size_t i = 0; i < waiting.size(); ++i) {
            auto& session = *waiting[i];
            if (proof[2 * i]) {
                MakeReady(session);
                continue;
            }
            const int preserve = proof[2 * i + 1];
            scheduler_->PrepareRestore(*session.sequence, Align(preserve), RetrieveEnd(*session.sequence), preserve);
        }
    }

    void OnScheduled()
    {
        std::vector<RequestSession*> waiting;
        std::vector<int>             admitted;
        for (auto& [uid, session] : sessions_) {
            auto& c = *session.sequence;
            if (!session.WaitingForRetrieve()) {
                continue;
            }
            waiting.push_back(&session);
            int status = 0;
            if (c.restore_plan->admitted) {
                try {
                    session.retrieve = PrepareRetrieve(c);
                    status           = 1;
                }
                catch (const std::exception& e) {
                    TM_LOG_WARN("LMCache RETRIEVE uid={}, worker={}: preparation failed: {}", uid, rank_, e.what());
                    status = -1;
                }
            }
            admitted.push_back(status);
        }
        if (waiting.empty()) {
            return;
        }
        comm::AllReduce(cache_group_, admitted.data(), admitted.size(), comm::RedOp::kMin);
        for (size_t i = 0; i < waiting.size(); ++i) {
            auto& session = *waiting[i];
            if (admitted[i] == 1) {
                const auto& plan = *session.sequence->restore_plan;
                if (session.lease) {
                    session.lease->DelegateLocks(plan.start, plan.end);
                    session.lease->ReleaseLocks();
                }
                // Submission waits for the next Poll, after the engine's
                // Update/Sync has fenced earlier users of recycled storage.
                session.retrieve->transfer.Activate();
            }
            else if (admitted[i] < 0) {
                MakeReady(session);
            }
            else {
                session.retrieve.reset();  // prepared contexts own no remote locks
                scheduler_->CompleteRestore(*session.sequence, false);
            }
        }
    }

    bool Schedulable(const Sequence& c) const
    {
        auto it = sessions_.find(c.req->unique_id);
        return it == sessions_.end() || it->second.phase == RequestSession::Phase::kReady
               || it->second.WaitingForRetrieve();
    }

    bool Fallback(Sequence& c)
    {
        auto it = sessions_.find(c.req->unique_id);
        if (it == sessions_.end() || !it->second.WaitingForRetrieve()) {
            return false;
        }
        MakeReady(it->second);
        return true;
    }

    void StageStores(int phase, Sequence* const* sequences, int count)
    {
        if (!transfers_healthy_ || count == 0) {
            return;
        }
        auto& batch = batches_[phase];
        TM_CHECK(batch.stores.empty() && !batch.produced);
        struct StoreRange {
            uint64_t uid;
            int      start;
            int      end;
        };
        std::vector<StoreRange> ranges;
        for (int i = 0; i < count; ++i) {
            const auto& c  = *sequences[i];
            auto        it = sessions_.find(c.req->unique_id);
            if (it == sessions_.end()) {
                continue;
            }
            const int end = Align(c.history_len + c.inflight_input_len + c.input_len);
            if (end > it->second.next_store_token) {
                ranges.push_back({c.req->unique_id, it->second.next_store_token, end});
            }
        }
        if (ranges.empty()) {
            return;  // Host ranges and reservation cursors are shared across TP.
        }
        auto common = ranges;
        comm::Broadcast(cache_group_, common, 0);
        TM_CHECK_EQ(ranges.size(), common.size());
        const bool       checkpoints = pools_.size() > 1;
        std::vector<int> available;
        for (size_t i = 0; i < ranges.size(); ++i) {
            const auto& r = ranges[i];
            TM_CHECK(r.uid == common[i].uid && r.start == common[i].start && r.end == common[i].end)
                << "STORE batch ranges differ across TP ranks";
            if (checkpoints) {
                const auto& c = *sessions_.at(r.uid).sequence;
                for (int end = r.start + chunk_size_; end <= r.end; end += chunk_size_) {
                    available.push_back(is_valid(c.block_ids[end / block_size_ - 1]->checkpoint));
                }
            }
        }
        if (checkpoints) {
            comm::AllReduce(cache_group_, available.data(), available.size(), comm::RedOp::kMin);
        }
        int index = 0;
        for (const auto& r : ranges) {
            auto& session = sessions_.at(r.uid);
            int   start   = r.start;
            if (checkpoints) {
                for (int pos = start; pos < r.end; pos += chunk_size_) {
                    if (!available[index++]) {
                        StageStore(batch, session, start, pos);
                        start = pos + chunk_size_;
                    }
                }
            }
            StageStore(batch, session, start, r.end);
            // Missing historical states are unavailable sources. New GDN
            // boundary snapshots are guaranteed by required admission.
            session.next_store_token = r.end;
        }
    }

    void OnBatchComplete(int phase)
    {
        auto& batch = batches_[phase];
        for (auto& store : batch.stores) {
            auto& request = store->request;
            request.end   = std::min<int64_t>(request.end, Align(store->session.sequence->filled_len));
        }
        batch.produced = !batch.stores.empty();
    }

    void Poll()
    {
        PollLookups();
        PollTransfers();
        SubmitStores();
    }

    void Drain()
    {
        // Scheduling has stopped. Discarding unsubmitted descriptors only
        // removes transfer pins; sequences still own their allocations until
        // the executor has stopped. No TP collective or session END runs here.
        batches_.clear();
        for (auto& store : stores_) {
            if (store->transfer) {
                store->transfer->Drain();
            }
        }
        stores_.clear();
        for (auto& [uid, session] : sessions_) {
            if (session.retrieve) {
                session.retrieve->transfer.Drain();
                session.retrieve.reset();
            }
            if (session.sequence->restore_plan) {
                scheduler_->CompleteRestore(*session.sequence, false);
            }
        }
    }

    bool Ready(const Sequence& c) const
    {
        auto it = sessions_.find(c.req->unique_id);
        return it == sessions_.end() || it->second.phase == RequestSession::Phase::kReady;
    }

    bool CanRetire(const Sequence& c) const
    {
        auto it = sessions_.find(c.req->unique_id);
        return it == sessions_.end()
               || (it->second.phase == RequestSession::Phase::kReady && it->second.pending_stores == 0);
    }

    bool HasPendingReleases() const
    {
        return transfers_healthy_ && std::any_of(sessions_.begin(), sessions_.end(), [](const auto& item) {
                   return item.second.pending_stores != 0 || item.second.retrieve != nullptr;
               });
    }

    void OnRetire(Sequence& c)
    {
        auto it = sessions_.find(c.req->unique_id);
        if (it != sessions_.end()) {
            TM_CHECK_EQ(it->second.pending_stores, 0);
            TM_CHECK(!it->second.retrieve);
            sessions_.erase(it);
        }
    }

    int chunk_size() const
    {
        return chunk_size_;
    }

private:
    int RetrieveEnd(const Sequence& c) const
    {
        return std::min(c.lmcache_matched_end, Align(std::max(0, c.prompt_len - 1)));
    }

    bool CanInstall(const Sequence& s) const
    {
        const auto& plan = *s.restore_plan;
        // A native forward may already be writing an allocated, incomplete node.
        // Never replace its allocation, even though our private copy is complete.
        for (int pos = plan.preserve_end; pos < plan.end; pos += block_size_) {
            const auto& node = *s.block_ids[pos / block_size_];
            if (!node.is_valid && is_valid(node.prefix)) {
                return false;
            }
        }
        return true;
    }

    void MakeReady(RequestSession& session, bool install = false)
    {
        session.retrieve.reset();
        if (session.sequence->restore_plan) {
            scheduler_->CompleteRestore(*session.sequence, install);
        }
        if (session.lease) {
            session.lease->ReleaseLocks();
        }
        session.phase = RequestSession::Phase::kReady;
    }

    std::unique_ptr<PendingRetrieve> PrepareRetrieve(Sequence& c)
    {
        const auto&      plan = *c.restore_plan;
        lmcache::Request request;
        request.request_id          = std::to_string(c.req->unique_id);
        request.start               = plan.start;
        request.end                 = plan.end;
        request.skip_first_n_tokens = plan.preserve_end - plan.start;
        request.token_ids.assign(c.token_ids, c.token_ids + plan.end);
        request.block_ids.resize(pools_.size());
        std::vector<CacheBlock*> pins;
        for (int pos = 0; pos < plan.end; pos += block_size_) {
            auto* block = pos < plan.preserve_end ? c.block_ids[pos / block_size_]->prefix.get() :
                                                    plan.prefixes[(pos - plan.preserve_end) / block_size_].get();
            pins.push_back(block);
            if (pos >= plan.start) {
                request.block_ids[0].push_back(connector_->BlockId(0, block->base(0)));
            }
        }
        for (const auto& block : plan.checkpoints) {
            pins.push_back(block.get());
            for (size_t group = 1; group < pools_.size(); ++group) {
                request.block_ids[group].push_back(connector_->BlockId(group, block->base(group - 1)));
            }
        }
        return std::make_unique<PendingRetrieve>(*connector_, std::move(request), std::move(pins));
    }

    void StageStore(StoreBatch& batch, RequestSession& session, int start, int end)
    {
        if (start == end) {
            return;
        }
        const auto&      c = *session.sequence;
        lmcache::Request request;
        request.request_id = std::to_string(c.req->unique_id);
        request.start      = start;
        request.end        = end;
        request.block_ids.resize(pools_.size());
        std::vector<CacheBlock*> sources;
        sources.reserve((end - start) / block_size_ + (end - start) / chunk_size_);

        auto& prefix = request.block_ids[0];
        prefix.reserve((end - start) / block_size_);
        for (int pos = start; pos < end; pos += block_size_) {
            auto* block = c.block_ids[pos / block_size_]->prefix.get();
            prefix.push_back(connector_->BlockId(0, block->base(0)));
            sources.push_back(block);
        }
        if (pools_.size() > 1) {
            for (size_t group = 1; group < pools_.size(); ++group) {
                request.block_ids[group].reserve((end - start) / chunk_size_);
            }
            for (int pos = start + chunk_size_; pos <= end; pos += chunk_size_) {
                auto* block = c.block_ids[pos / block_size_ - 1]->checkpoint.get();
                // All checkpoint parts share one allocation and one source pin.
                sources.push_back(block);
                for (size_t group = 1; group < pools_.size(); ++group) {
                    request.block_ids[group].push_back(connector_->BlockId(group, block->base(group - 1)));
                }
            }
        }
        batch.stores.push_back(std::make_unique<PendingStore>(session, std::move(request), std::move(sources)));
    }

    int Align(int tokens) const
    {
        return tokens / chunk_size_ * chunk_size_;
    }

    void PollLookups()
    {
        if (std::none_of(sessions_.begin(), sessions_.end(), [](const auto& item) {
                return item.second.phase == RequestSession::Phase::kLookup;
            })) {
            return;
        }
        struct LookupCompletion {
            uint64_t uid;
            int      matched;
        };
        std::vector<LookupCompletion> done;
        if (rank_ == 0) {
            for (auto& [uid, session] : sessions_) {
                if (session.phase != RequestSession::Phase::kLookup) {
                    continue;
                }
                const auto result = session.lookup->Poll();
                if (!result) {
                    continue;
                }
                int matched = 0;
                if (result->success) {
                    matched = result->matched_tokens;
                    session.lease.emplace(session.lookup->TakeLease());
                    TM_LOG_DEBUG("LMCache LOOKUP uid={}: matched={} tokens", uid, matched);
                }
                else {
                    TM_LOG_WARN("LMCache LOOKUP uid={} failed, recomputing locally: {}", uid, result->error);
                }
                session.lookup.reset();
                done.push_back({uid, matched});
            }
        }
        comm::Broadcast(tp_group_, done, 0);
        for (const auto& result : done) {
            auto& session                         = sessions_.at(result.uid);
            session.sequence->lmcache_matched_end = result.matched;
            session.next_store_token              = result.matched;
            session.phase =
                RetrieveEnd(*session.sequence) > 0 ? RequestSession::Phase::kRetrieve : RequestSession::Phase::kReady;
            if (session.phase == RequestSession::Phase::kReady) {
                MakeReady(session);
            }
        }
    }

    void SubmitStores()
    {
        for (auto& batch : batches_) {
            if (!batch.produced) {
                continue;
            }
            std::shared_ptr<lmcache::CudaEvent> ready;
            for (auto& store : batch.stores) {
                auto&       session = store->session;
                const auto& c       = *session.sequence;
                auto&       request = store->request;
                if (!transfers_healthy_ || c.is_canceled || request.end <= request.start) {
                    continue;  // Shared decision: no worker submits this operation.
                }
                if (rank_ == 0 && !session.lease) {
                    session.lease.emplace(*connector_, std::to_string(c.req->unique_id));
                }
                try {
                    if (!ready) {
                        ready = lmcache::StoreContext::RecordReady(core::Context::stream().handle());
                    }
                    for (size_t group = 0; group < pools_.size(); ++group) {
                        request.block_ids[group].resize((request.end - request.start) / pools_[group].tokens_per_block);
                    }
                    request.token_ids.assign(c.token_ids, c.token_ids + request.end);
                    store->transfer.emplace(*connector_, std::move(request), ready);
                }
                catch (const std::exception& e) {
                    TM_LOG_WARN("LMCache STORE uid={}, worker={}: failed before submission: {}",
                                c.req->unique_id,
                                rank_,
                                e.what());
                }
                // A local failure keeps its slot for the shared outcome reduction.
                stores_.push_back(std::move(store));
            }
            batch.stores.clear();
            batch.produced = false;
        }
    }

    void PollTransfers()
    {
        const bool produced   = std::any_of(batches_.begin(), batches_.end(), [](const auto& b) { return b.produced; });
        const bool retrieving = std::any_of(sessions_.begin(), sessions_.end(), [](const auto& item) {
            return item.second.phase == RequestSession::Phase::kRetrieve;
        });
        if (sessions_.empty() || (transfers_healthy_ && stores_.empty() && !produced && !retrieving)) {
            return;
        }
        // Entry zero gates new transfers. Other entries are pending=0,
        // finished=1; MIN requires every worker to finish.
        std::vector<int> status{int(connector_->healthy())};
        for (auto& store : stores_) {
            int value = 1;
            if (store->transfer) {
                const auto result = store->transfer->Poll();
                value             = result.has_value();
                if (store->transfer->uncertain()) {
                    status[0] = 0;
                }
            }
            status.push_back(value);
        }
        for (auto& [uid, session] : sessions_) {
            if (session.WaitingForRetrieve() && session.sequence->retiring) {
                MakeReady(session);
            }
            if (!session.retrieve) {
                continue;
            }
            auto& transfer = session.retrieve->transfer;
            if (session.sequence->retiring) {
                transfer.Cancel();
            }
            const auto result = transfer.Poll();
            // pending=0, safely failed=1, successfully installable=2
            const bool install =
                result && result->success && !session.sequence->retiring && CanInstall(*session.sequence);
            status.push_back(!result ? 0 : install ? 2 : 1);
            if (transfer.uncertain()) {
                status[0] = 0;
            }
        }
        comm::AllReduce(cache_group_, status.data(), status.size(), comm::RedOp::kMin);
        transfers_healthy_ = status[0] != 0;
        int index          = 1;
        stores_.erase(std::remove_if(stores_.begin(), stores_.end(), [&](const auto&) { return status[index++] != 0; }),
                      stores_.end());
        for (auto& [uid, session] : sessions_) {
            if (!session.retrieve) {
                continue;
            }
            const int result = status[index++];
            if (result == 0) {
                continue;
            }
            MakeReady(session, result == 2);
        }
    }

    Scheduler*                          scheduler_{};
    comm::HostComm&                     tp_group_;
    comm::HostComm                      cache_group_;
    const int                           rank_;
    const int&                          is_warm_up_;
    const int                           block_size_;
    int                                 chunk_size_;
    bool                                transfers_healthy_{true};
    std::vector<lmcache::CachePool>     pools_;
    std::unique_ptr<lmcache::Connector> connector_;
    std::map<uint64_t, RequestSession>  sessions_;
    std::vector<StoreBatch>             batches_;
    Stores                              stores_;
};

LmCache LmCache::Create(const std::string& addr,
                        const std::string& model_name,
                        comm::HostComm&    group,
                        const int&         warmup,
                        int                block_size,
                        int                phases)
{
    if (addr.empty()) {
        return {};
    }
    return LmCache{std::make_unique<Impl>(addr, model_name, group, warmup, block_size, phases)};
}
LmCache::LmCache()                   = default;
LmCache::~LmCache()                  = default;
LmCache::LmCache(LmCache&&) noexcept = default;
LmCache& LmCache::operator=(LmCache&&) noexcept = default;
LmCache::LmCache(std::unique_ptr<Impl> impl): impl_{std::move(impl)} {}
void LmCache::Register(void* base, const ObjectAllocator& allocator, const CacheRegistry& registry)
{
    if (impl_) {
        impl_->Register(base, allocator, registry);
    }
}
void LmCache::Bind(Scheduler& scheduler)
{
    if (impl_) {
        impl_->Bind(scheduler);
    }
}
void LmCache::PrepareSchedule()
{
    if (impl_) {
        impl_->PrepareSchedule();
    }
}
void LmCache::OnScheduled()
{
    if (impl_) {
        impl_->OnScheduled();
    }
}
bool LmCache::Schedulable(const Sequence& c) const
{
    return !impl_ || impl_->Schedulable(c);
}
bool LmCache::Fallback(Sequence& c)
{
    return impl_ && impl_->Fallback(c);
}
void LmCache::OnAccepted(Sequence& c)
{
    if (impl_) {
        impl_->OnAccepted(c);
    }
}
void LmCache::StageStores(int phase, Sequence* const* c, int count)
{
    if (impl_) {
        impl_->StageStores(phase, c, count);
    }
}
void LmCache::OnBatchComplete(int phase)
{
    if (impl_) {
        impl_->OnBatchComplete(phase);
    }
}
void LmCache::Poll()
{
    if (impl_) {
        impl_->Poll();
    }
}
void LmCache::Drain()
{
    if (impl_) {
        impl_->Drain();
    }
}
bool LmCache::Ready(const Sequence& c) const
{
    return !impl_ || impl_->Ready(c);
}
bool LmCache::CanRetire(const Sequence& c) const
{
    return !impl_ || impl_->CanRetire(c);
}
bool LmCache::HasPendingReleases() const
{
    return impl_ && impl_->HasPendingReleases();
}
int LmCache::chunk_size() const
{
    return impl_ ? impl_->chunk_size() : 0;
}
void LmCache::OnRetire(Sequence& c)
{
    if (impl_) {
        impl_->OnRetire(c);
    }
}

}  // namespace turbomind
