#pragma once

#include <chrono>
#include <utility>
#include <vector>

#include "src/turbomind/comm/env.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/engine/block.h"
#include "src/turbomind/engine/cache_mode.h"
#include "src/turbomind/engine/cache_registry.h"
#include "src/turbomind/engine/prefix_trie.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/memory/object.h"

#define TM_SCHED_PROFILE 0

namespace turbomind {

#if TM_SCHED_PROFILE
struct PerformanceCounter {
    using time_point = std::chrono::high_resolution_clock::time_point;
    std::vector<time_point> timestamps;
    std::vector<int>        passes;

    PerformanceCounter() = default;
    explicit PerformanceCounter(int capacity): timestamps(capacity), passes(capacity) {}

    float dist(int i, int j) const noexcept
    {
        if (passes[i]) {
            // TM_CHECK_EQ(passes[i], passes[j]);
            return static_cast<float>(
                       std::chrono::duration_cast<std::chrono::nanoseconds>(timestamps[j] - timestamps[i]).count())
                   * 0.001f / passes[i];
        }
        else {
            return 0.f;
        }
    }

    void tick(int index)
    {
        // TM_CHECK(passes[index] == 0);
        timestamps[index] = std::chrono::high_resolution_clock::now();
        passes[index]     = 1;
    }

    PerformanceCounter& operator+=(const PerformanceCounter& other)
    {
        for (size_t i = 0; i < timestamps.size(); ++i) {
            if (other.passes[i]) {
                timestamps[i] += (other.timestamps[i] - other.timestamps[0]);
                passes[i] += other.passes[i];
            }
        }
        return *this;
    }

    explicit operator bool() const noexcept
    {
        return passes[0] > 0;
    }
};
#else
struct PerformanceCounter {
    PerformanceCounter() = default;
    explicit PerformanceCounter(int) {}
    void                tick(int) {}
    PerformanceCounter& operator+=(const PerformanceCounter&)
    {
        return *this;
    }
    float dist(int, int) const noexcept
    {
        return 0.f;
    }
    explicit operator bool() const noexcept
    {
        return false;
    }
};
#endif

TM_ENV_VAR(CACHE, LOG_INTERVAL, 0);  // 0 = disabled; N = log every N Schedule() passes

class Scheduler {
public:
    Scheduler(ObjectAllocator&   alloc,
              CacheRegistry      registry,
              int                cache_block_seq_len,
              bool               enable_prefix_caching,
              const std::string& cache_prompt,
              int                cache_prompt_boundary_skip,
              const std::string& cache_generation,
              const int&         is_warm_up);

    ~Scheduler();

    const LogicalBlockPool& logical() const noexcept
    {
        return logical_;
    }

    const CacheRegistry& registry() const noexcept
    {
        return registry_;
    }

    const ObjectAllocator& allocator() const noexcept
    {
        return alloc_;
    }

    bool prefix_enabled() const noexcept
    {
        return enable_prefix_caching_;
    }

    // True if any multimodal span overlaps [lo, hi). Pure; used by SetupPartialSiblings to
    // gate the 'auto' prompt-boundary publish. Public so it can be unit-tested.
    static bool HasMultimodalOverlap(const Sequence& s, int lo, int hi);

    // Match the prompt against the prefix trie; create missing blocks; set up
    // the partial sibling edge (matcher bind + prompt-boundary node creation).
    void AdmitPrompt(Sequence& s);

    // Commit step: per-request planning (PlanResume/PlanContinue), admission with
    // scratch allocation + eviction, memory replay, publication, MarkProduced.
    void Schedule(std::vector<Sequence*> requests, Resource& resource);

    // Index generated blocks into the trie; adopt the frontier into the last
    // partial block. Called on normal finish.
    void Finalize(Sequence& s);

    // Drop the request's references; pool recycling does the rest.
    void Release(Sequence& s);

    // Observability-only records consumed by the file-local prefix-cache log
    // helpers in scheduler.cc. Public so those file-local helpers can name them.
    struct PublishStat {
        int  start           = 0;      // first newly-valid prefix block offset (token); MarkProduced()
        int  reusable_blocks = 0;      // indexed nodes whose is_valid flipped true this pass; MarkProduced()
        int  end             = 0;      // highest published prefix position (token); MarkProduced()
        bool forked          = false;  // a partial sibling populated this pass; set by CommitResults()
        bool ckpt            = false;  // a checkpoint published this pass; set by CommitResults()
    };
    struct ProducerConflict {
        uint64_t producer = 0;   // blocking request's unique_id; 0 = none
        int      block    = -1;  // first conflicting block index
    };

private:
    // Shared state of one Schedule pass; defined in scheduler.cc so the
    // replay/admission types stay file-local.
    struct ScheduleState;

    // Optional checkpoint-publication intent allocated in the optional admission
    // phase. slot is the target's own (block-owned) checkpoint slot; nullptr => nothing.
    struct PublishPlan {
        LogicalBlock* target{};
        int           end{};
        CacheBlock*   slot{};
    };

    // Schedule phases, called in order; see Schedule's body.
    void PlanRequests(ScheduleState& pass);
    void RunRequiredAdmission(ScheduleState& pass, Resource& resource);
    void RunOptionalAdmission(ScheduleState& pass);
    void ReplayMemory(ScheduleState& pass);
    void CommitResults(ScheduleState& pass);

    // Trie cursor threaded through the AdmitPrompt phases; defined in scheduler.cc.
    struct AcceptState;

    void MatchPrompt(Sequence& s, AcceptState& cur);
    void IndexMissingBlocks(Sequence& s, AcceptState& cur);
    void SetupPartialSiblings(Sequence& s, AcceptState& cur);

    // Per-request planning for inactive sequences: find the latest feasible
    // resume step, emit restore copy plans, fill resume_len/alloc/involved.
    void PlanResume(Sequence& s);

    // Per-request planning for sequences active in the last iteration.
    void PlanContinue(Sequence& s);

    // Clear producer marks and mark produced blocks valid for [t0, end). Returns
    // the indexed blocks that became cross-request reusable this pass.
    PublishStat MarkProduced(Sequence& s, int t0, int end);

    void             SetProducers(Sequence& s, int t0, int end);
    ProducerConflict CheckProducers(const Sequence& s, int t0, int end) const;

    // Land the forward end on a boundary candidate; a result <= begin means
    // nothing runs this pass. Precedence documented at the definition.
    int ClampForwardEnd(const Sequence& s, int begin, int desired, int ctx_end) const;

    // Publication planning for a committed forward ending at `end`, routed by
    // whether this is the prompt-boundary pass (end == B). Finds the node
    // ending exactly at `end` (the block itself when block-aligned, else its
    // partial sibling), then decides partial sibling KV population and the
    // checkpoint. Only records intent; slots are allocated in the optional
    // admission phase. Reserves the sibling's prefix slot in pass.planned to
    // dedup intent across requests.
    void PlanPublication(ScheduleState& pass, int i, Sequence& s, int end, bool at_prompt_boundary);

    void EnsureBlocks(Sequence& s);

    bool      PrefixEligible(const Sequence& s) const noexcept;
    bool      CheckpointPublicationEligible() const noexcept;
    TokenSpan TokenSegment(const Sequence& s, int offset, int size) const;

    void LogProfile(const PerformanceCounter& counter) const;

    bool             enable_prefix_caching_{false};
    CacheMode        prompt_cache_mode_{CacheMode::kAuto};
    int              cache_prompt_boundary_skip_{1};
    CacheMode        generation_cache_mode_{CacheMode::kAuto};
    const int&       is_warm_up_;
    ObjectAllocator& alloc_;     // owned by Engine; also used outside the scheduler
    CacheRegistry    registry_;  // owned: registration is closed before construction
    CacheBlockPool   cache_;
    LogicalBlockPool logical_;
    PrefixTrie       trie_;

    PerformanceCounter counter_;
    PerformanceCounter interv_;
    PerformanceCounter accum_;
};

}  // namespace turbomind
