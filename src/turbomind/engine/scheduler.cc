#include "src/turbomind/engine/scheduler.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/engine/cache_mode.h"
#include "src/turbomind/engine/prompt_boundary.h"
#include "src/turbomind/memory/common.h"

namespace turbomind {

namespace {

inline int InitialResumeUpperBound(const Sequence& s)
{
    const int context_len = s.seq_len + s.inflight_new_tokens - s.inflight_input_len;
    return std::max(0, std::min(s.seq_len, context_len - 1));
}

// Clear per-pass planning buffers (alloc, restore, publish); involved_cache_ids persists.
inline void ResetPassBuffers(Sequence& s)
{
    s.alloc_cache_ids.clear();
    s.restore_copies.clear();
    s.publish_copies.clear();
    s.publish_target = nullptr;
    s.publish_end    = 0;
}

// Full rebuild: per-pass buffers plus involved_cache_ids (Resume only).
inline void ResetPlanBuffers(Sequence& s)
{
    ResetPassBuffers(s);
    s.involved_cache_ids.clear();
}

struct AllocReplay {
    int cache_id;
};

struct EvictReplay {
    int cache_id;
};

using Replay = std::vector<std::variant<AllocReplay, EvictReplay>>;

class EvictingIterator {
public:
    EvictingIterator(const std::vector<int>& cache_ids, const CacheBlockPool& cache):
        cache_ids_{&cache_ids}, cache_{&cache}
    {
    }

    EvictingIterator(std::vector<int>&&, const CacheBlockPool&) = delete;

    EvictingIterator(const EvictingIterator& base, uint64_t cutoff):
        cache_ids_{base.cache_ids_}, pos_{base.pos_}, cache_{base.cache_}, cutoff_{cutoff}
    {
    }

    EvictingIterator(const EvictingIterator&) noexcept = default;
    EvictingIterator& operator=(const EvictingIterator&) noexcept = default;

    explicit operator bool() const noexcept
    {
        return pos_ < cache_ids_->size() && (*cache_)[(*cache_ids_)[pos_]].timestamp < cutoff_;
    }

    uint64_t Evict(ScratchAllocator& scratch, Replay& replay)
    {
        const int   cache_id = (*cache_ids_)[pos_++];
        const auto& cache    = (*cache_)[cache_id];
        scratch.Evict(cache.object_id, cache.allocation.a);
        replay.push_back(EvictReplay{cache_id});
        return cache.timestamp;
    }

    size_t pos() const noexcept
    {
        return pos_;
    }

    void SeekTo(size_t pos) noexcept
    {
        pos_ = pos;
    }

private:
    const std::vector<int>* cache_ids_;
    size_t                  pos_{};
    const CacheBlockPool*   cache_;
    uint64_t                cutoff_{std::numeric_limits<uint64_t>::max()};
};

class AllocatingIterator {
public:
    AllocatingIterator(const std::vector<int>& cache_ids, const CacheBlockPool& cache):
        iter_{cache_ids.begin()}, end_{cache_ids.end()}, cache_{cache}
    {
    }

    AllocatingIterator(std::vector<int>&&, const CacheBlockPool&) = delete;

    AllocatingIterator(const AllocatingIterator&) = delete;
    AllocatingIterator& operator=(const AllocatingIterator&) = delete;

    explicit operator bool() const noexcept
    {
        return iter_ != end_;
    }

    // Idempotent: ids already allocated for real (cached alloc set), or
    // planned by an earlier request in this pass, are skipped.
    bool
    Allocate(ScratchAllocator& scratch, std::unordered_set<int>& planned, std::vector<int>& planned_now, Replay& replay)
    {
        const int   cache_id = *iter_;
        const auto& cache    = cache_[cache_id];
        if (cache.valid() || planned.count(cache_id)) {
            ++iter_;
            return true;
        }
        if (scratch.Allocate(cache.object_id)) {
            ++iter_;
            planned.insert(cache_id);
            planned_now.push_back(cache_id);
            replay.push_back(AllocReplay{cache_id});
            return true;
        }
        return false;
    }

private:
    std::vector<int>::const_iterator iter_;
    std::vector<int>::const_iterator end_;
    const CacheBlockPool&            cache_;
};

const char* ResumeSourceName(ResumeSource src)
{
    switch (src) {
        case ResumeSource::kPrefix:
            return "prefix";
        case ResumeSource::kFrontier:
            return "frontier";
        case ResumeSource::kCheckpoint:
            return "checkpoint";
        case ResumeSource::kFork:
            return "fork";
        default:
            return "none";
    }
}

// Collect start-fingerprints of images whose start token lies in [lo, hi), with
// their block-relative start positions. multimodal_spans is prompt-ordered
// ascending by interval.begin().
void CollectStartFps(const Sequence&           s,
                     int                       lo,
                     int                       hi,
                     std::vector<Fingerprint>& fps,
                     std::vector<int>*         pos = nullptr)
{
    for (const auto& sp : s.multimodal_spans) {
        const int b = sp.interval.begin();
        if (b < lo) {
            continue;
        }
        if (b >= hi) {
            break;
        }
        fps.push_back(sp.fingerprint);
        if (pos) {
            pos->push_back(b - lo);
        }
    }
}

enum class CollisionSite
{
    kAccept,
    kPromptBoundary,
    kPublish
};

// Finalize-event record, filled by PublishGeneration's index loop.
struct GenStat {
    int  first_offset  = 0;  // offset of first newly-indexed generated block (token)
    int  indexed       = 0;  // generated blocks newly inserted into the trie
    int  last_size     = 0;  // filled tokens of the last inserted block
    bool terminal_ckpt = false;
    int  dropped       = 0;  // redundant full-block checkpoints dropped on terminal adoption
};

// Prefix-cache log helpers (definitions at the bottom of this file). Each opens
// with an isolated level gate so its formatting is skipped when level > INFO.
// Rule: derive what survives the pass from `s`; pass a record for ephemeral
// within-pass facts. `bs` only where range math needs it.
void LogAccept(const Sequence& s, int bs);
void LogResume(const Sequence& s);
void LogDeferred(const Sequence& s, int bs, const Scheduler::ProducerConflict& c);
void LogPublished(const Sequence& s, int bs, const Scheduler::PublishStat& p);
void LogFinalized(const Sequence& s, int bs, const GenStat& g);
void LogCollision(const Sequence& s, CollisionSite site, int begin, int end);

}  // namespace

// True if any multimodal span overlaps [lo, hi). Interval is the absolute token
// span [begin, end); a partial prompt block "contains image tokens" when a span
// intersects it, even one that started in an earlier (full) block and extends
// in. multimodal_spans is prompt-ordered ascending by interval.begin().
bool Scheduler::HasMultimodalOverlap(const Sequence& s, int lo, int hi)
{
    for (const auto& sp : s.multimodal_spans) {
        if (sp.interval.begin() >= hi) {
            break;  // ascending; no later span can overlap
        }
        if (sp.interval.end() > lo) {
            return true;
        }
    }
    return false;
}

static PerformanceCounter make_perf_counter()
{
    constexpr int kSchedPerfCounters = 32;
    return PerformanceCounter{kSchedPerfCounters};
}

struct Scheduler::ScheduleState {
    std::vector<Sequence*> requests;

    std::vector<uint64_t>      cutoff;                    // per-request eviction cutoff stamps
    uint64_t                   floor{};                   // pass-start timestamp; inactive < floor <= cutoff[i]
    Replay                     replay;                    // alloc/evict ops of the current phase
    size_t                     committed_replay_size{0};  // replay prefix from committed requests (phase 1)
    std::vector<bool>          committed;
    std::vector<LogicalBlock*> pending_fork;          // fork_to node per request, nullptr = none
    std::vector<PublishPlan>   pending_publish;       // checkpoint publication intent per request
    bool                       has_optionals{false};  // any optional intent recorded => run phase 2
    std::vector<int>           evict_ids;             // SortedIndices() snapshot, shared by both phases
    size_t                     evict_pos{0};          // oldest-first eviction cursor shared by both phases
    std::unordered_set<int>    planned;               // cache ids planned/reserved for allocation
};

bool Scheduler::PrefixEligible(const Sequence& s) const noexcept
{
    // Native VLM (multimodal_spans) is eligible: image identity is carried by the
    // per-image fingerprint folded into the prefix key. The legacy Python-embedding
    // path (input_embeds) stays excluded -- out of scope for this change.
    return enable_prefix_caching_ && !is_warm_up_ && s.input_embeds.empty() && s.input_embeds_offsets.empty()
           && s.token_ids != nullptr;
}

TokenSpan Scheduler::TokenSegment(const Sequence& s, int offset, int size) const
{
    TM_CHECK_NOTNULL(s.token_ids);
    TM_CHECK_GE(offset, 0);
    TM_CHECK_GE(size, 0);
    TM_CHECK_LE(offset + size, s.seq_len);
    return MakeTokenSpan(s.token_ids + offset, size);
}

Scheduler::Scheduler(ObjectAllocator&                     alloc,
                     CacheRegistry                        registry,
                     int                                  cache_block_seq_len,
                     bool                                 enable_prefix_caching,
                     const std::string&                   cache_prompt,
                     int                                  cache_prompt_boundary_skip,
                     const std::string&                   cache_generation,
                     const int&                           is_warm_up):
    enable_prefix_caching_{enable_prefix_caching},
    prompt_cache_mode_{ParseCacheMode(cache_prompt)},
    cache_prompt_boundary_skip_{cache_prompt_boundary_skip < 1 ? 1 : cache_prompt_boundary_skip},
    generation_cache_mode_{ParseCacheMode(cache_generation)},
    is_warm_up_{is_warm_up},
    alloc_{alloc},
    registry_{std::move(registry)},
    logical_{cache_, cache_block_seq_len},
    trie_{cache_block_seq_len},
    accum_{make_perf_counter()},
    interv_{make_perf_counter()}
{
    logical_.set_recycle_hook([this](LogicalBlock& b) { trie_.Erase(b); });
}

Scheduler::~Scheduler()
{
    if (interv_) {
        accum_ += interv_;
        interv_ = {};
    }
    LogProfile(accum_);

    // Drain all live allocations so allocation-held refs are released and the
    // remaining trie nodes recycle before the pools are destroyed.
    // SortedIndices() returns exactly the allocated blocks (alloc set).
    for (const int id : cache_.SortedIndices()) {
        cache_.Deallocate(alloc_, id);
        if (LogicalBlock* o = cache_[id].owner) {
            logical_.Drop(o);
        }
    }
}

void Scheduler::EnsureBlocks(Sequence& s)
{
    const int bs     = logical_.block_size();
    const int length = s.seq_len + s.inflight_new_tokens;
    const int needed = (length + bs - 1) / bs;
    while (static_cast<int>(s.block_ids.size()) < needed) {
        const int   i = static_cast<int>(s.block_ids.size());
        BlockHandle h = logical_.Create(i);
        h->prefix_id  = cache_.Create(registry_.prefix().object_id(), h.get());  // owner = node
        s.block_ids.push_back(std::move(h));                                     // request ref
    }
}

struct Scheduler::AcceptState {
    const LogicalBlock* parent{};  // trie node reached so far (nullptr = root)
    PrefixKey           key{};

    int                 miss{};         // first block index not matched in the trie
    const LogicalBlock* miss_parent{};  // trie position at the miss, for fork_from
    PrefixKey           miss_key{};

    size_t next_fp = 0;  // monotonic cursor into Sequence::multimodal_spans
};

void Scheduler::Accept(Sequence& s)
{
    TM_CHECK(s.block_ids.empty());
    if (!PrefixEligible(s)) {
        return;  // blocks are created lazily by EnsureBlocks
    }
    AcceptState st{};            // parent defaults to nullptr (root)
    MatchPrompt(s, st);          // match full blocks to the first miss
    s.matched_blocks = st.miss;  // leading prompt blocks found in the trie
    CreateMissingBlocks(s, st);  // create + index the remaining prompt blocks
    SetupForks(s, st);           // fork_from (partial match) + fork_to (prompt boundary)
    LogAccept(s, logical_.block_size());
}

void Scheduler::MatchPrompt(Sequence& s, AcceptState& st)
{
    const int bs          = logical_.block_size();
    const int full_blocks = s.prompt_len / bs;

    int i = 0;
    for (; i < full_blocks; ++i) {
        const int                offset = i * bs;
        size_t                   cur    = st.next_fp;  // working copy; do not commit on a miss
        std::vector<Fingerprint> fps;
        while (cur < s.multimodal_spans.size() && s.multimodal_spans[cur].interval.begin() < offset + bs) {
            fps.push_back(s.multimodal_spans[cur].fingerprint);
            ++cur;
        }
        const auto tokens = TokenSegment(s, offset, bs);
        const auto next   = ExtendPrefixKey(st.key, tokens, fps);
        if (LogicalBlock* b = trie_.Find(st.parent, next, tokens, fps)) {
            s.block_ids.emplace_back(b);  // retain via BlockHandle copy
            st.parent  = b;
            st.key     = next;
            st.next_fp = cur;  // commit advance only on a match
        }
        else {
            break;  // cursor still at the miss block's first span
        }
    }

    st.miss        = i;
    st.miss_parent = st.parent;
    st.miss_key    = st.key;
}

void Scheduler::CreateMissingBlocks(Sequence& s, AcceptState& st)
{
    const int bs     = logical_.block_size();
    const int prompt = s.prompt_len;

    const int all_blocks = (prompt + bs - 1) / bs;

    for (int i = st.miss; i < all_blocks; ++i) {
        const int                offset = i * bs;
        const int                size   = std::min(prompt - offset, bs);
        std::vector<Fingerprint> fps;
        while (st.next_fp < s.multimodal_spans.size()
               && s.multimodal_spans[st.next_fp].interval.begin() < offset + size) {
            fps.push_back(s.multimodal_spans[st.next_fp].fingerprint);
            ++st.next_fp;
        }
        const auto    tokens = TokenSegment(s, offset, size);
        BlockHandle   h      = logical_.Create(i);
        LogicalBlock& x      = *h;
        x.prefix_id          = cache_.Create(registry_.prefix().object_id(), h.get());
        if (size == bs) {
            const auto next = ExtendPrefixKey(st.key, tokens, fps);
            x.parent        = st.parent;
            x.key           = next;
            x.size          = size;
            x.tokens.assign(tokens.begin(), tokens.end());
            x.image_fps     = fps;  // usually empty
            if (!trie_.Insert(x)) {
                LogCollision(s, CollisionSite::kAccept, offset, offset + size);
                // Stays un-indexed; treated as a private block from here on.
                x.parent = nullptr;
                x.key    = {};
                x.size   = 0;
                x.tokens.clear();
                x.image_fps.clear();
            }
            else {
                st.parent = h.get();
                st.key    = next;
            }
        }
        // The partial last block stays private; parent/key do not advance.
        s.block_ids.push_back(std::move(h));  // request ref
    }
}

void Scheduler::SetupForks(Sequence& s, AcceptState& st)
{
    const int bs     = logical_.block_size();
    const int prompt = s.prompt_len;

    const int all_blocks = (prompt + bs - 1) / bs;

    // fork_from (read side) is always armed: any prior request may have published
    // a prompt partial node (cache_prompt in {all, auto}) or a generation
    // terminal partial ('all'), so the read edge must always try to match.
    if (st.miss < all_blocks) {
        LogicalBlock& x      = *s.block_ids[st.miss];
        const int     offset = st.miss * bs;
        const int     size   = std::min(prompt - offset, bs);
        PrefixKey     k      = st.miss_key;

        std::vector<Fingerprint> fps;
        std::vector<int>         fp_pos;
        CollectStartFps(s, offset, offset + size, fps, &fp_pos);

        if (LogicalBlock* v = trie_.Search(st.miss_parent, k, TokenSegment(s, offset, size), fps, fp_pos)) {
            x.fork_from = BlockHandle{v};  // edge ref
        }
    }

    // Prompt-boundary publish point (fork_to). B = prompt_len - K (K =
    // cache_prompt_boundary_skip). 'all' publishes a partial node whenever B is
    // mid-block and arms the checkpoint clamp when B is block-aligned. 'auto'
    // publishes the partial node only when its own token range [j*bs, B) overlaps
    // a multimodal span (including a span that began in an earlier block and
    // extends into this range), and never arms the block-aligned clamp.
    const auto plan = PlanPromptBoundary(prompt, bs, cache_prompt_boundary_skip_, st.miss);
    if (plan.valid) {
        const bool need_image = plan.partial && prompt_cache_mode_ == CacheMode::kAuto;
        const bool has_image  = need_image && HasMultimodalOverlap(s, plan.block * bs, plan.pos);

        if (DecidePromptBoundaryPublish(prompt_cache_mode_, plan.partial, has_image)) {
            bool have_target = true;

            if (plan.partial) {
                const int     j      = plan.block;  // j >= 1 (guaranteed by the planner)
                LogicalBlock& x      = *s.block_ids[j];
                const auto    tokens = TokenSegment(s, j * bs, plan.node_size);

                std::vector<Fingerprint> fps;
                CollectStartFps(s, j * bs, j * bs + plan.node_size, fps);

                const auto    next = ExtendPrefixKey(s.block_ids[j - 1]->key, tokens, fps);
                BlockHandle   vh   = logical_.Create(j);
                LogicalBlock& y    = *vh;
                y.parent           = s.block_ids[j - 1].get();
                y.key              = next;
                y.size             = plan.node_size;
                y.tokens.assign(tokens.begin(), tokens.end());
                y.image_fps        = fps;
                y.prefix_id        = cache_.Create(registry_.prefix().object_id(), vh.get());
                if (trie_.Insert(y)) {
                    x.fork_to = std::move(vh);  // edge holds the only ref
                }
                else {
                    LogCollision(s, CollisionSite::kPromptBoundary, j * bs, j * bs + plan.node_size);
                    have_target = false;  // undiscoverable: vh drops at scope end -> recycle
                }
            }

            if (have_target) {
                s.prompt_boundary_node = true;
                s.prompt_boundary_pos  = plan.pos;  // clamp the producer's prefill to B
            }
        }
    }
}

void Scheduler::Resume(Sequence& s)
{
    TM_CHECK(!s.is_active);

    s.resuming = true;

    EnsureBlocks(s);

    ResetPlanBuffers(s);

    const bool ckpt  = registry_.has_checkpoint();
    const int  upper = InitialResumeUpperBound(s);
    const int  bs    = logical_.block_size();

    if (ckpt) {
        if (s.frontier_cache_id == 0) {
            s.frontier_cache_id = cache_.Create(registry_.checkpoint().object_id());
            s.frontier_pos      = 0;
        }
        if (s.publish_cache_id == 0) {
            s.publish_cache_id = cache_.Create(registry_.checkpoint().object_id());
        }
    }

    // 1. Contiguous reusable prefix end (token level). Indexed nodes carry
    //    their own content extent (size); private blocks are capped by what
    //    this request has proven produced (filled_len).
    int prefix_end         = 0;
    int readonly_block_num = 0;
    for (const BlockHandle& h : s.block_ids) {
        const LogicalBlock& x = *h;
        if (!x.is_valid || !ValidAlloc(x.prefix_id)) {
            break;
        }
        const int extent = x.key ? x.size : std::min(std::max(s.filled_len - x.offset, 0), x.capacity);
        if (extent <= 0) {
            break;
        }
        prefix_end = x.offset + extent;
        if (extent < x.capacity) {
            break;  // partial / own-frontier: first writable block
        }
        ++readonly_block_num;  // fully-valid whole block: read-only reusable
    }
    prefix_end           = std::min(prefix_end, upper);  // resume bound, unchanged
    s.readonly_block_num = readonly_block_num;

    // 2. Resume step selection
    int           step         = prefix_end;  // without checkpointing, KV grants per-token resume
    ResumeSource  source       = prefix_end > 0 ? ResumeSource::kPrefix : ResumeSource::kNone;
    LogicalBlock* fork_dst     = nullptr;
    LogicalBlock* fork_src     = nullptr;
    int           restore_ckpt = 0;  // checkpoint cache id to copy into the frontier

    if (ckpt) {
        step   = 0;
        source = ResumeSource::kNone;

        // Frontier fast path (no copy needed)
        const int fpos = s.frontier_pos - s.inflight_input_len;
        if (ValidAlloc(s.frontier_cache_id) && 0 < fpos && fpos <= prefix_end) {
            step   = fpos;
            source = ResumeSource::kFrontier;
        }

        // Latest block checkpoint within the reusable prefix
        if (step < prefix_end) {
            for (int i = std::min<int>(s.block_ids.size(), (prefix_end + bs - 1) / bs); i > 0; --i) {
                const LogicalBlock& x = *s.block_ids[i - 1];
                const int           e = x.key ? x.offset + x.size : x.offset + x.capacity;
                if (e <= step) {
                    break;
                }
                if (e <= prefix_end && ValidAlloc(x.checkpoint_id)) {
                    step         = e;
                    source       = ResumeSource::kCheckpoint;
                    restore_ckpt = x.checkpoint_id;
                    break;
                }
            }
        }
    }

    // 3. Fork extension: an indexed partial node can beat the current step by
    //    copying its content into our private block at the boundary.
    if (prefix_end % bs == 0 && prefix_end / bs < static_cast<int>(s.block_ids.size())) {
        LogicalBlock& x = *s.block_ids[prefix_end / bs];
        if (x.fork_from) {
            const LogicalBlock& y = *x.fork_from;
            const int           e = y.offset + y.size;
            if (y.is_valid && e <= upper && e > step && ValidAlloc(y.prefix_id)
                && (!ckpt || ValidAlloc(y.checkpoint_id))) {
                step         = e;
                source       = ResumeSource::kFork;
                fork_dst     = &x;
                fork_src     = x.fork_from.get();
                restore_ckpt = ckpt ? y.checkpoint_id : 0;
            }
        }
    }

    s.resume_len = step;
    // source is kNone exactly when step == 0, so no extra guard is needed.
    s.resume_source = source;

    // 4. Restore copy plans (cache ids; resolved to pointers at setup)
    if (fork_dst) {
        s.restore_copies.push_back({fork_src->prefix_id, fork_dst->prefix_id});
    }
    if (ckpt && step > 0 && restore_ckpt) {
        s.restore_copies.push_back({restore_ckpt, s.frontier_cache_id});
    }
    // step == 0 with checkpointing: GDN recognizes a forward starting at
    // position 0 (history_len + inflight_input_len == 0) and resets.

    // 5. Allocation set and eviction-protection set. Protect only what is
    //    needed to run the forward: the prefix blocks (read-only context + the
    //    written tail) and the single frontier. Published checkpoints are
    //    resume-time optimizations, not run-time state — they stay out of the
    //    protected set so they remain evictable (a long/high-priority sequence
    //    can reclaim its own prior checkpoints to run). The one checkpoint (or
    //    fork source) actually restored this pass is protected separately via
    //    its restore_copies entry (stamped in PlanRequests, Section 4).
    for (const BlockHandle& h : s.block_ids) {
        const LogicalBlock& x = *h;
        s.involved_cache_ids.push_back(x.prefix_id);
        if (!ValidAlloc(x.prefix_id)) {
            s.alloc_cache_ids.push_back(x.prefix_id);
        }
    }
    if (ckpt) {
        s.involved_cache_ids.push_back(s.frontier_cache_id);
        if (!ValidAlloc(s.frontier_cache_id)) {
            s.alloc_cache_ids.push_back(s.frontier_cache_id);
        }
    }
}

void Scheduler::Continue(Sequence& s)
{
    TM_CHECK(s.is_active);

    s.resume_len         = s.filled_len;
    s.readonly_block_num = 0;  // decode writes only the new token (past the boundary)
    s.resuming           = false;

    const int first_new = static_cast<int>(s.block_ids.size());
    EnsureBlocks(s);

    ResetPassBuffers(s);  // per-pass buffers only; involved_cache_ids persists

    const bool ckpt = registry_.has_checkpoint();

    if (ckpt && s.publish_cache_id == 0) {
        s.publish_cache_id = cache_.Create(registry_.checkpoint().object_id());
    }

    // Active-request invariant: a request that committed last pass kept every
    // involved cache id (none were evicted) and allocated its whole required
    // set, so the persistent involved set is still valid. Only the blocks
    // appended by EnsureBlocks since the last plan are new, and being freshly
    // created they are unallocated. Published checkpoints are deliberately not
    // tracked here: they are not needed to run and must stay evictable so the
    // sequence can run with just its prefix blocks and frontier.
    for (int i = first_new; i < static_cast<int>(s.block_ids.size()); ++i) {
        const int p = s.block_ids[i]->prefix_id;
        s.involved_cache_ids.push_back(p);
        s.alloc_cache_ids.push_back(p);
    }

    // The frontier was added to involved_cache_ids by the activating Resume and
    // stays valid while active (it is in the protected set); nothing to re-add.
    if (ckpt) {
        TM_CHECK(ValidAlloc(s.frontier_cache_id));
    }
}

void Scheduler::SetProducers(Sequence& s, int t0, int end)
{
    const int bs = logical_.block_size();
    for (int i = t0 / bs; i < (end + bs - 1) / bs; ++i) {
        s.block_ids[i]->producer = s.req->unique_id;
    }
}

Scheduler::ProducerConflict Scheduler::CheckProducers(const Sequence& s, int t0, int end) const
{
    const int bs = logical_.block_size();
    for (int i = t0 / bs; i < (end + bs - 1) / bs; ++i) {
        const LogicalBlock& x = *s.block_ids[i];
        if (x.producer && x.producer != s.req->unique_id) {
            return {x.producer, i};
        }
    }
    return {};
}

Scheduler::PublishStat Scheduler::Publish(Sequence& s, int t0, int end)
{
    const int   bs   = logical_.block_size();
    const int   last = std::min<int>((end + bs - 1) / bs, s.block_ids.size());
    PublishStat stat{};
    for (int i = 0; i < last; ++i) {
        LogicalBlock& x = *s.block_ids[i];
        if (x.producer == s.req->unique_id) {
            x.producer = 0;
        }
        if (x.key) {
            // Indexed nodes become valid only when fully covered
            if (x.offset + x.size <= end) {
                if (!x.is_valid) {
                    if (stat.reusable_blocks == 0) {
                        stat.start = x.offset;
                    }
                    ++stat.reusable_blocks;
                    stat.end = x.offset + x.size;
                }
                x.is_valid = true;
            }
        }
        else {
            // Private blocks: content extent is tracked via filled_len
            x.is_valid = true;
        }
    }
    return stat;
}

void Scheduler::ReleaseCacheId(int cache_id)
{
    if (cache_id == 0) {
        return;
    }
    auto& c = cache_[cache_id];
    if (c.object_id >= 0) {
        TM_CHECK(c.owner == nullptr);  // request-owned ids only (frontier/publish)
        if (c.valid()) {
            cache_.Deallocate(alloc_, cache_id);
        }
        cache_.Invalidate(cache_id);
    }
}

void Scheduler::Release(Sequence& s)
{
    for (const BlockHandle& h : s.block_ids) {
        LogicalBlock& x = *h;
        if (!x.indexed) {
            // Private blocks are undiscoverable: drop their allocations now so
            // the allocation-held refs go away and the block can recycle.
            for (const int c : {x.prefix_id, x.checkpoint_id}) {
                if (ValidAlloc(c)) {
                    cache_.Deallocate(alloc_, c);
                    logical_.Drop(&x);  // the allocation's ref (request ref still pins x)
                }
            }
        }
    }
    s.block_ids.clear();  // request refs -> recycles unreferenced blocks

    ReleaseCacheId(std::exchange(s.frontier_cache_id, 0));
    ReleaseCacheId(std::exchange(s.publish_cache_id, 0));

    s.frontier_pos   = 0;
    s.last_ckpt_pos  = 0;
    s.publish_target = nullptr;
    s.publish_end    = 0;
    s.alloc_cache_ids.clear();
    s.involved_cache_ids.clear();
    s.restore_copies.clear();
    s.publish_copies.clear();
    s.resume_len         = 0;
    s.filled_len         = 0;
    s.readonly_block_num = 0;
    s.input_len          = 0;
    s.history_len        = 0;
}

void Scheduler::PublishGeneration(Sequence& s)
{
    if (!PrefixEligible(s) || s.filled_len <= 0) {
        return;
    }
    if (generation_cache_mode_ == CacheMode::kNone) {
        return;  // index no generated blocks at all
    }

    // 'all' indexes the terminal partial block + adopts the terminal recurrent
    // frontier checkpoint; 'auto' indexes full generated blocks only.
    const bool publish_generation_boundary = (generation_cache_mode_ == CacheMode::kAll);

    const LogicalBlock* parent = nullptr;
    PrefixKey           key{};

    GenStat gen{};  // index-loop summary for the finalized log

    for (size_t i = 0; i < s.block_ids.size(); ++i) {
        LogicalBlock* up = s.block_ids[i].get();
        LogicalBlock& x  = *up;
        if (x.offset >= s.filled_len) {
            break;
        }
        if (x.indexed) {
            if (x.offset + x.size > s.filled_len) {
                break;
            }
            parent = up;
            key    = x.key;
            continue;
        }
        const int size = std::min(s.filled_len - x.offset, x.capacity);
        if (!x.is_valid || !ValidAlloc(x.prefix_id)) {
            break;
        }
        // The terminal partial generated block is the generation-boundary partial
        // node; index it only when generation_cache_mode_ is kAll
        // (publish_generation_boundary). It carries the partial block's KV for
        // every model; a recurrent model additionally adopts the terminal frontier
        // checkpoint below (guarded by a valid frontier id). Full generated blocks
        // always index. It ends at filled_len, so nothing follows.
        if (size < x.capacity && !publish_generation_boundary) {
            break;
        }
        const auto               tokens = TokenSegment(s, x.offset, size);
        std::vector<Fingerprint> fps;
        if (x.offset < s.prompt_len) {
            // Only the prompt-tail block (private until now) can hold an image start;
            // generated positions never do. Fold + store so this node's identity
            // matches what a future request's MatchPrompt rebuilds.
            CollectStartFps(s, x.offset, x.offset + size, fps);
        }
        const auto next = ExtendPrefixKey(key, tokens, fps);
        x.parent        = parent;
        x.key           = next;
        x.size          = size;
        x.tokens.assign(tokens.begin(), tokens.end());
        x.image_fps     = fps;  // usually empty
        if (!trie_.Insert(x)) {
            LogCollision(s, CollisionSite::kPublish, x.offset, x.offset + size);
            x.parent = nullptr;
            x.key    = {};
            x.size   = 0;
            x.tokens.clear();
            x.image_fps.clear();
            break;
        }
        if (gen.indexed == 0) {
            gen.first_offset = x.offset;
        }
        ++gen.indexed;
        gen.last_size = size;
        // Adopt the frontier as the terminal checkpoint of the last block.
        // Gated by publish_generation_boundary (this is the only partial-block
        // generation checkpoint; full-block ones at boundaries stay always-on).
        //
        // The live recurrent buffer is guaranteed to correspond to filled_len
        // here: the finishing pass stored its state at filled_len, and the GDN
        // recurrence kernel bypasses its state write-back whenever the device
        // finished mask is set, so any async over-shoot pass leaves the buffer
        // untouched. We deliberately do NOT test s.frontier_pos: that field is
        // resume-fast-path bookkeeping, committed speculatively as the scheduled
        // forward end (CommitResults), so async lookahead over-counts it past
        // filled_len and it would spuriously block this (safe) adoption.
        // A valid frontier id implies checkpoints are registered (created only
        // under has_checkpoint()), so no separate has_checkpoint() gate here.
        if (publish_generation_boundary && x.offset + size == s.filled_len && ValidAlloc(s.frontier_cache_id)
            && x.checkpoint_id == 0) {

            const int interval = registry_.checkpoint_min_interval();

            // Classify in-window checkpoints below filled_len. A checkpoint on a
            // block being indexed in *this* call (pos > prompt_len, still
            // private until now -> no consumer ref) is droppable; one on an
            // already-shared block (pos <= prompt_len) is a blocker we must not
            // touch, so we skip adoption to preserve min_interval spacing.
            bool blocked = false;
            for (int j = static_cast<int>(i); j-- > 0;) {
                const LogicalBlock& p   = *s.block_ids[j];
                const int           pos = p.offset + p.size;
                if (s.filled_len - pos >= interval) {
                    break;  // outside the window
                }
                if (const int c = p.checkpoint_id; ValidAlloc(c) && pos <= s.prompt_len) {
                    blocked = true;
                    break;
                }
            }

            if (!blocked) {
                const int f     = std::exchange(s.frontier_cache_id, 0);
                x.checkpoint_id = f;
                cache_[f].owner = up;
                logical_.Retain(up);  // ref held by the live allocation
                gen.terminal_ckpt = true;

                // Drop droppable redundant full-block checkpoints in the window;
                // the terminal checkpoint supersedes them. Mirror eviction
                // exactly: free memory + drop the allocation's logical ref.
                for (int j = static_cast<int>(i); j-- > 0;) {
                    LogicalBlock& p   = *s.block_ids[j];
                    const int     pos = p.offset + p.size;
                    if (s.filled_len - pos >= interval) {
                        break;  // outside the window; spacing already satisfies min_interval
                    }
                    if (pos > s.prompt_len) {
                        if (const int c = p.checkpoint_id; ValidAlloc(c)) {
                            cache_.Deallocate(alloc_, c);  // free memory + drop the alloc ref
                            logical_.Drop(&p);  // block stays (request + index refs); slot left as evicted leftover
                            ++gen.dropped;      // observability only (LogFinalized)
                        }
                    }
                }
            }
        }
        parent = up;
        key    = next;
    }

    LogFinalized(s, logical_.block_size(), gen);
}

// When this pass reaches the prompt boundary, plan the device copy that
// populates the indexed prompt-end partial node (fork_to). Returns the
// fork_to node when a copy is planned, nullptr otherwise.
LogicalBlock* Scheduler::PlanForkToPopulation(Sequence& s, int end, std::unordered_set<int>& planned)
{
    const int bs = logical_.block_size();

    const LogicalBlock& x = *s.block_ids[(end - 1) / bs];
    if (!x.fork_to) {
        return nullptr;
    }
    const LogicalBlock& y       = *x.fork_to;
    const int           y_cache = y.prefix_id;
    if (y.offset + y.size != end || y.is_valid || ValidAlloc(y_cache) || planned.count(y_cache)) {
        return nullptr;  // boundary not reached, or another request already covers it
    }
    // Reserve the node so a later request sharing it does not also plan to
    // populate it. The slot itself is allocated in the optional phase (from
    // inactive memory); this reservation only dedups intent within the pass. A
    // fork-to node is a distinct logical block from any request's required
    // prefix blocks, so it never collides with a required allocation id.
    planned.insert(y_cache);
    return x.fork_to.get();
}

// Prompt-boundary group (caller guarantees end == B == prompt_boundary_pos): fork_to KV copy +
// checkpoint, both partial-block, bypassing the min-interval.
void Scheduler::PlanPromptBoundaryPublication(ScheduleState& pass, int i, Sequence& s, int end)
{
    // (a) copy the request's partial KV into the shared fork_to node.
    if (LogicalBlock* node = PlanForkToPopulation(s, end, pass.planned)) {
        pass.pending_fork[i] = node;
        pass.has_optionals   = true;
    }

    // (b) checkpoint onto the fork_to node, or the block itself when
    // block-aligned B is a block boundary.
    if (s.publish_cache_id) {
        LogicalBlock& x          = *s.block_ids[(end - 1) / logical_.block_size()];
        const bool    at_block   = x.offset + x.capacity == end;
        const bool    at_fork_to = x.fork_to && x.fork_to->offset + x.fork_to->size == end;
        LogicalBlock* target     = at_block ? &x : (at_fork_to ? x.fork_to.get() : nullptr);
        if (target && !ValidAlloc(target->checkpoint_id)) {
            pass.pending_publish[i] = {target, end, s.publish_cache_id};
            pass.has_optionals      = true;
        }
    }
}

// Full-block group: coverage-driven checkpoint, published iff a full block ends
// exactly at `end` (subject to min-interval); no prompt-boundary mode involved.
// The full block's prefix is published in place by Publish() (no KV copy).
void Scheduler::PlanFullBlockPublication(ScheduleState& pass, int i, Sequence& s, int end)
{
    if (s.publish_cache_id == 0) {
        return;
    }
    LogicalBlock& x = *s.block_ids[(end - 1) / logical_.block_size()];
    if (x.offset + x.capacity != end) {
        return;  // partial block — nothing to publish
    }
    const int interval = registry_.checkpoint_min_interval();
    if (end - s.last_ckpt_pos >= interval && !ValidAlloc(x.checkpoint_id)) {
        pass.pending_publish[i] = {&x, end, s.publish_cache_id};
        pass.has_optionals      = true;
    }
}

void Scheduler::Schedule(std::vector<Sequence*> requests, Resource& resource)
{
    counter_ = make_perf_counter();

    counter_.tick(0);

    ScheduleState pass{std::move(requests)};
    PlanRequests(pass);  // Resume/Continue, sort, stamp involved + restore srcs, pass.floor

    counter_.tick(1);

    RunRequiredAdmission(pass, resource);  // phase 1: required scratch alloc + eviction; collect intents

    counter_.tick(2);

    ReplayMemory(pass);  // commit phase 1; clears pass.replay

    counter_.tick(3);

    if (pass.has_optionals) {
        counter_.tick(10);

        RunOptionalAdmission(pass);  // allocate intents from inactive memory; fill pass.replay

        counter_.tick(11);

        ReplayMemory(pass);  // commit phase 2; clears pass.replay

        counter_.tick(12);
    }
    counter_.tick(4);

    CommitResults(pass);  // publication attach, fork_to populate, Publish

    counter_.tick(5);

    interv_ += counter_;

#if TM_SCHED_PROFILE
    if (int n = GetEnv<CACHE_LOG_INTERVAL>(); n > 0 && interv_.passes[0] == n) {
        LogProfile(interv_);
        accum_ += interv_;
        interv_ = make_perf_counter();
    }
#endif
}

void Scheduler::PlanRequests(ScheduleState& pass)
{
    for (Sequence* sp : pass.requests) {
        if (sp->is_active) {
            Continue(*sp);
        }
        else {
            Resume(*sp);
        }
    }

    std::sort(pass.requests.begin(), pass.requests.end(), [](Sequence* a, Sequence* b) {
        return a->req->unique_id < b->req->unique_id;
    });

    pass.cutoff.resize(pass.requests.size());
    const int n = static_cast<int>(pass.requests.size());
    for (int i = n; i > 0; --i) {
        Sequence&      s   = *pass.requests[i - 1];
        const uint64_t pre = cache_.Stamp(s.involved_cache_ids);  // pre-stamp value = cutoff
        pass.cutoff[i - 1] = pre;
        if (i == n) {
            pass.floor = pre;  // pass-start timestamp: the inactive/active boundary
        }
        // Protect the sources read by this pass's restore copies (a restored
        // checkpoint and/or a fork source). They are foreign blocks not in this
        // request's involved set, but must survive eviction until the restore
        // copy runs before kPrepare. They land just above this request's cutoff,
        // in the same band the old code gave them when they lived in involved.
        // restore_copies is empty on the Continue path, so this adds nothing there.
        for (const CacheCopy& c : s.restore_copies) {
            cache_.Stamp(c.src);
        }
    }

    pass.committed.assign(pass.requests.size(), false);
    pass.pending_fork.assign(pass.requests.size(), nullptr);
    pass.pending_publish.assign(pass.requests.size(), PublishPlan{});
}

void Scheduler::RunRequiredAdmission(ScheduleState& pass, Resource& resource)
{
    counter_.tick(20);

    pass.evict_ids = cache_.SortedIndices();

    counter_.tick(21);

    EvictingIterator evict_pos{pass.evict_ids, cache_};

    uint64_t max_evict_ts = 0;

    ScratchAllocator scratch{alloc_};

    counter_.tick(22);

    const int bs = logical_.block_size();

    // Required admission loop: place every forward that fits (prefix blocks +
    // frontier), evicting up to each request's cutoff. Optional optimizations
    // (publication, fork-to population) are only decided here; their slots are
    // allocated later, in RunOptionalAdmission, from inactive memory.
    for (int i = 0; i < static_cast<int>(pass.requests.size()); ++i) {
        auto& s = *pass.requests[i];

        if (max_evict_ts >= pass.cutoff[i]) {
            break;  // would run on memory evicted from a higher-priority request
        }

        const int admitted = resource.Test(s);
        if (admitted == 0) {
            TM_LOG_INFO("hit resource limit at {}/{}", i, pass.requests.size());
            break;
        }

        s.history_len = s.resume_len;

        // Land the forward end on a checkpoint candidate. The prompt-boundary
        // clamp (forward ends exactly at B) takes precedence;
        // otherwise truncate partial prefill chunks to a block boundary.
        const int begin   = s.resume_len + s.inflight_input_len;
        const int ctx_end = s.seq_len + s.inflight_new_tokens;  // == prompt_len for a fresh prefill
        int       desired = begin + admitted;

        const int prompt_boundary_pos = s.prompt_boundary_pos;

        // The publish decision is finalized in SetupForks (prompt_boundary_node);
        // the clamp fires on the pass that can reach B (>= so an exact landing
        // isn't truncated away).
        const bool publish_prompt =
            s.prompt_boundary_node && begin < prompt_boundary_pos && desired >= prompt_boundary_pos;

        if (publish_prompt) {
            desired = prompt_boundary_pos;  // land exactly on B
        }
        else if (desired < ctx_end) {  // partial chunk: truncate to a block boundary
            desired = desired / bs * bs;
        }

        const int len = desired - begin;
        if (len <= 0) {
            continue;  // nothing admitted this pass; CommitResults leaves it inactive
        }
        s.input_len = len;

        const int end = begin + s.input_len;

        if (const ProducerConflict conflict = CheckProducers(s, begin, end); conflict.producer) {
            LogDeferred(s, bs, conflict);
            continue;  // deferred; CommitResults leaves it inactive
        }

        EvictingIterator   evicting{evict_pos, pass.cutoff[i]};
        AllocatingIterator allocating{s.alloc_cache_ids, cache_};

        uint64_t         evict_ts = 0;
        std::vector<int> planned_now;

        bool ok = true;
        while (allocating) {
            bool success = allocating.Allocate(scratch, pass.planned, planned_now, pass.replay);
            while (!success && evicting) {
                evict_ts = evicting.Evict(scratch, pass.replay);
                success  = allocating.Allocate(scratch, pass.planned, planned_now, pass.replay);
            }
            if (!success) {
                ok = false;
                break;
            }
        }

        if (!ok) {  // out of memory: roll back this request's planning, stop the pass
            for (const int id : planned_now) {
                pass.planned.erase(id);
            }
            TM_LOG_INFO("out of memory at {}/{}", i, pass.requests.size());
            break;  // CommitResults leaves this and all later requests inactive
        }

        resource.Commit(s);
        s.is_active                = true;
        pass.committed[i]          = true;
        pass.committed_replay_size = pass.replay.size();
        max_evict_ts               = std::max(max_evict_ts, evict_ts);
        evict_pos                  = evicting;

        // Optional optimizations (allocated later, from inactive memory). One
        // checkpoint per forward, routed by its end; publish_prompt is false when
        // prompt_boundary_node was not set in SetupForks, or when this forward's
        // geometry does not reach B, so nothing prompt-boundary is allocated.
        // PlanPromptBoundaryPublication reserves the fork-to id in pass.planned for
        // cross-request intent dedup.
        if (publish_prompt) {
            PlanPromptBoundaryPublication(pass, i, s, end);  // fork_to KV + prompt-boundary checkpoint
        }
        else {
            PlanFullBlockPublication(pass, i, s, end);  // full-block checkpoint (coverage only)
        }

        SetProducers(s, begin, end);
        LogResume(s);  // emit here so a producer's resume precedes any later consumer's defer log
    }

    counter_.tick(23);

    scratch = {};

    counter_.tick(24);

    pass.replay.resize(pass.committed_replay_size);
    pass.evict_pos = evict_pos.pos();  // hand the oldest-first cursor to the optional phase
}

void Scheduler::RunOptionalAdmission(ScheduleState& pass)
{
    ScratchAllocator opt{alloc_};

    // Continue the monotonic oldest-first sweep from where phase 1 stopped, but
    // reach only INACTIVE slots (timestamp < pass.floor) of any category.
    // Phase-1-evicted candidates lie strictly before evict_pos; phase-1-allocated
    // blocks are not in the pass-start snapshot; surviving candidates are still
    // allocated (a slot stays allocated unless evicted, and its block stays alive
    // via its own allocation ref or a sequence/fork ref), so they are valid to
    // evict here. opt is a ScratchAllocator over the committed (post-phase-1)
    // state: it holds a capacity (MemoryState) copy and borrows the live
    // allocator's object registry, so no ObjectAllocator is cloned. The
    // recorded replay is applied to the real allocator afterward.
    EvictingIterator base{pass.evict_ids, cache_};
    base.SeekTo(pass.evict_pos);
    EvictingIterator evicting{base, pass.floor};

    // Skip only on real, committed memory (c.valid()). A fork-to id reserved in
    // pass.planned during phase 1 still needs its slot allocated here, so we must
    // NOT treat membership in pass.planned as "already allocated".
    auto try_optional = [&](int cache_id) -> bool {
        const auto& c = cache_[cache_id];
        if (c.valid()) {
            return true;
        }
        bool ok = opt.Allocate(c.object_id);
        while (!ok && evicting) {
            evicting.Evict(opt, pass.replay);
            ok = opt.Allocate(c.object_id);
        }
        if (!ok) {
            return false;
        }
        pass.planned.insert(cache_id);
        pass.replay.push_back(AllocReplay{cache_id});
        return true;
    };

    for (int i = 0; i < static_cast<int>(pass.requests.size()); ++i) {
        if (!pass.committed[i]) {
            continue;
        }
        Sequence& s = *pass.requests[i];

        // fork-to population (prefix reuse for future forks)
        if (LogicalBlock* node = pass.pending_fork[i]) {
            if (!try_optional(node->prefix_id)) {
                pass.pending_fork[i] = nullptr;  // dropped; CommitResults won't populate it
            }
        }
        // checkpoint publication
        if (const PublishPlan& pub = pass.pending_publish[i]; pub.cache_id) {
            if (try_optional(pub.cache_id)) {
                s.publish_target = pub.target;  // confirmed; CommitResults attaches it
                s.publish_end    = pub.end;
            }
            // else: skip publication this pass; publish_cache_id stays reserved
        }
    }
}

void Scheduler::ReplayMemory(ScheduleState& pass)
{
    // Memory replay: the ONLY place where actual allocation/deallocation
    // happens during a scheduling pass.
    for (const auto& op : pass.replay) {
        std::visit(
            [&](const auto& item) {
                using T = std::decay_t<decltype(item)>;
                auto& c = cache_[item.cache_id];
                if constexpr (std::is_same_v<T, EvictReplay>) {
                    const bool is_prefix = c.object_id == registry_.prefix().object_id_or_negative();
                    cache_.Deallocate(alloc_, item.cache_id);  // clears allocation; owner persists
                    if (LogicalBlock* o = c.owner) {
                        if (is_prefix) {
                            o->is_valid = false;
                        }
                        logical_.Drop(o);  // may recycle the block and free this slot
                    }
                }
                else {
                    c.allocation = alloc_.Allocate(c.object_id);  // single-object; {nullptr} on OOM
                    TM_CHECK(c.allocation.a);                     // admission guarantees capacity
                    c.alloc_key = c.allocation->key;              // snapshot for stale detection
                    logical_.Retain(c.owner);                     // no-op when owner == nullptr (request-owned)
                }
            },
            op);
    }
    pass.replay.clear();  // each phase materializes only its own segment
}

void Scheduler::CommitResults(ScheduleState& pass)
{
    const int bs = logical_.block_size();

    // Post-replay commit: publication attach, fork_to population, frontier
    // metadata, and publication of produced ranges.
    for (int i = 0; i < static_cast<int>(pass.requests.size()); ++i) {
        auto& s = *pass.requests[i];

        // README scheduler-inactive: reset every uncommitted request here only.
        // RunRequiredAdmission reject paths rely on committed[i] == false reaching this
        // branch; do not zero these fields at reject sites.
        if (!pass.committed[i]) {
            s.is_active      = false;
            s.input_len      = 0;
            s.history_len    = 0;
            s.publish_target = nullptr;
            s.publish_end    = 0;
            s.alloc_cache_ids.clear();
            s.restore_copies.clear();
            s.publish_copies.clear();
            continue;
        }

        // A resuming request was inactive, so its prior `filled_len` predates the
        // prefix it now reuses read-only / restores from a checkpoint. Reconcile it
        // to the resume point: `filled_len` is the context currently established, not
        // just KV this request's own forward produced. The `[resume_len, end)` span
        // the in-flight resume forward rebuilds is carried by `inflight_input_len`.
        // Safe to write here: the request is inactive, so this never races Update()
        // of the previous batch.
        if (s.resuming) {
            s.filled_len = s.resume_len;
        }

        const int begin = s.history_len + s.inflight_input_len;
        const int end   = begin + s.input_len;

        bool ckpt_published = false;

        if (LogicalBlock* v = pass.pending_fork[i]) {
            LogicalBlock& y = *v;
            y.is_valid      = true;  // content arrives via the device-ordered copy below
            s.publish_copies.push_back({s.block_ids[(end - 1) / bs]->prefix_id, y.prefix_id});
            // Allocated outside the stamped involved sets: stamp now so the
            // freshly populated node is not the top eviction candidate.
            cache_.Stamp(y.prefix_id);
        }

        if (s.publish_target) {
            LogicalBlock& t = *s.publish_target;
            if (ValidAlloc(t.checkpoint_id)) {
                // Another request in this pass already published this node
                ReleaseCacheId(std::exchange(s.publish_cache_id, 0));
            }
            else {
                if (const int stale = t.checkpoint_id) {
                    cache_.Invalidate(stale);  // evicted leftover slot
                }
                const int id     = std::exchange(s.publish_cache_id, 0);
                t.checkpoint_id  = id;
                cache_[id].owner = s.publish_target;
                logical_.Retain(s.publish_target);  // ref held by the live allocation
                s.last_ckpt_pos = s.publish_end;
                ckpt_published  = true;
                s.publish_copies.push_back({s.frontier_cache_id, id});
                // Allocated outside the stamped involved sets: stamp now so
                // the fresh checkpoint is not the top eviction candidate.
                cache_.Stamp(id);
            }
            s.publish_target = nullptr;
            s.publish_end    = 0;
        }

        if (registry_.has_checkpoint()) {
            s.frontier_pos = end;
        }

        // Content is guaranteed to be produced by this iteration (device
        // execution is in submission order); no point deferring to Update().
        PublishStat pub = Publish(s, begin, end);
        pub.forked      = pass.pending_fork[i] != nullptr;
        pub.ckpt        = ckpt_published;
        LogPublished(s, bs, pub);
    }
}

void Scheduler::LogProfile(const PerformanceCounter& counter) const
{
#if TM_SCHED_PROFILE
    // TODO: Gate TP rank

    fmt::memory_buffer buf;

    fmt::format_to(std::back_inserter(buf),
                   "\n[sched] total {:.2f}, plan {:.2f}, required {:.2f}, replay {:.2f}, commit {:.2f}",
                   counter.dist(0, 5),
                   counter.dist(0, 1),
                   counter.dist(1, 2),
                   counter.dist(2, 3),
                   counter.dist(4, 5));

    if (counter.passes[10]) {
        fmt::format_to(std::back_inserter(buf),
                       "\n[sched]   optional {:.2f}, replay {:.2f}",
                       counter.dist(10, 11),
                       counter.dist(11, 12));
    }

    fmt::format_to(std::back_inserter(buf),
                   "\n[sched]   sort {:.2f}, scratch {:.2f}, loop {:.2f}, ~scratch {:.2f}",
                   counter.dist(20, 21),
                   counter.dist(21, 22),
                   counter.dist(22, 23),
                   counter.dist(23, 24));

    TM_LOG_WARN("sched stats:{}", fmt::to_string(buf));
#endif
}

namespace {

using turbomind::core::Logger;

void LogAccept(const Sequence& s, int bs)
{
    if (Logger::Instance().get_level() > Logger::Level::kInfo) {
        return;
    }
    const int   prompt = s.prompt_len, full = prompt / bs, all = (prompt + bs - 1) / bs;
    const int   matched = s.matched_blocks, M = matched * bs;
    std::string mtail, clast, ctail;
    if (matched < (int)s.block_ids.size() && s.block_ids[matched]->fork_from) {
        const LogicalBlock& y = *s.block_ids[matched]->fork_from;
        mtail                 = fmt::format(", fork_from@{}", y.offset + y.size);  // matched-side partial reuse
    }
    if (all - matched > 0 && prompt % bs) {
        clast = fmt::format(", last {}/{}", prompt - full * bs, bs);  // created-side partial tail
    }
    if (s.prompt_boundary_pos > 0) {
        const int j = (s.prompt_boundary_pos - 1) / bs;  // block holding B (matches PlanPromptBoundary)
        if (j >= 0 && j < (int)s.block_ids.size() && s.block_ids[j]->fork_to) {
            const LogicalBlock& ft = *s.block_ids[j]->fork_to;
            ctail                  = fmt::format(", fork_to@{}", ft.offset + ft.size);  // created-side publish node end
        }
    }
    TM_LOG_INFO("req {} (uid {}) matched [0,{}) ({} blk){} | created [{},{}) ({} blk{}){}",
                s.req->id,
                s.req->unique_id,
                M,
                matched,
                mtail,
                M,
                prompt,
                all - matched,
                clast,
                ctail);
}

void LogResume(const Sequence& s)
{
    if (Logger::Instance().get_level() > Logger::Level::kInfo) {
        return;
    }
    if (!s.resuming) {
        return;
    }
    const int begin = s.history_len + s.inflight_input_len;
    const int end   = begin + s.input_len;
    const int total = s.seq_len + s.inflight_new_tokens;
    const int pct   = total > 0 ? 100 * s.history_len / total : 0;
    TM_LOG_INFO("req {} (uid {}) resume [0,{}) {} blk ro ({}%) source={} | computed [{},{}) {} tok",
                s.req->id,
                s.req->unique_id,
                s.history_len,
                s.readonly_block_num,
                pct,
                ResumeSourceName(s.resume_source),
                begin,
                end,
                s.input_len);
}

void LogDeferred(const Sequence& s, int bs, const Scheduler::ProducerConflict& c)
{
    if (Logger::Instance().get_level() > Logger::Level::kInfo) {
        return;
    }
    const int begin = s.resume_len + s.inflight_input_len;
    const int end   = begin + s.input_len;
    const int b0    = std::max(begin, c.block * bs);
    const int b1    = std::min(end, (c.block + 1) * bs);
    TM_LOG_INFO("req {} (uid {}) deferred: tok [{},{}) held by producer uid {}",
                s.req->id,
                s.req->unique_id,
                b0,
                b1,
                c.producer);
}

void LogPublished(const Sequence& s, int bs, const Scheduler::PublishStat& p)
{
    if (Logger::Instance().get_level() > Logger::Level::kInfo) {
        return;
    }
    if (!(p.reusable_blocks > 0 || p.forked || p.ckpt)) {
        return;
    }
    const int   end = s.history_len + s.inflight_input_len + s.input_len;  // forward end this pass
    std::string body;
    auto        add = [&](std::string c) { body += body.empty() ? c : ", " + c; };
    if (p.reusable_blocks > 0) {
        add(fmt::format("prefix [{},{}) ({} blk)", p.start, p.end, p.reusable_blocks));
    }
    if (p.forked) {
        const int b0 = (end - 1) / bs * bs;
        add(fmt::format("boundary [{},{}) ({} tok)", b0, end, end - b0));
    }
    if (p.ckpt) {
        add(fmt::format("ckpt@{}", s.last_ckpt_pos));
    }
    TM_LOG_INFO("req {} (uid {}) published {}", s.req->id, s.req->unique_id, body);
}

void LogFinalized(const Sequence& s, int bs, const GenStat& g)
{
    if (Logger::Instance().get_level() > Logger::Level::kInfo) {
        return;
    }
    if (!(g.indexed > 0 || g.terminal_ckpt)) {
        return;
    }  // terminal_ckpt implies indexed > 0
    std::string tail = (g.last_size < bs) ? fmt::format(", last {}/{}", g.last_size, bs) : "";
    std::string ckpt =
        g.terminal_ckpt ? (g.dropped ? fmt::format(", terminal ckpt (dropped {})", g.dropped) : ", terminal ckpt") : "";
    TM_LOG_INFO("req {} (uid {}) finalized gen [{},{}) ({} blk{}){}",
                s.req->id,
                s.req->unique_id,
                g.first_offset,
                s.filled_len,
                g.indexed,
                tail,
                ckpt);
}

void LogCollision(const Sequence& s, CollisionSite site, int begin, int end)
{
    if (Logger::Instance().get_level() > Logger::Level::kInfo) {
        return;
    }
    const char* where = "";
    const char* note  = "";
    switch (site) {
        case CollisionSite::kAccept:
            where = "accept";
            note  = "";
            break;
        case CollisionSite::kPromptBoundary:
            where = "prompt boundary";
            note  = " (no fork_to)";
            break;
        case CollisionSite::kPublish:
            where = "publish";
            note  = ", stop";
            break;
    }
    TM_LOG_INFO("req {} (uid {}) collision at {}: tok [{},{}) → private{}",
                s.req->id,
                s.req->unique_id,
                where,
                begin,
                end,
                note);
}

}  // namespace

}  // namespace turbomind
