#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <utility>
#include <vector>

#include "src/turbomind/core/check.h"
#include "src/turbomind/engine/fingerprint.h"
#include "src/turbomind/engine/prefix_key.h"
#include "src/turbomind/memory/common.h"
#include "src/turbomind/memory/object.h"

namespace turbomind {

struct LogicalBlock;
class LogicalBlockPool;
struct CacheBlock;
class CacheBlockPool;

// Intrusive, non-atomic strong handle to a logical block (engine-thread only).
// Holds one ref for its lifetime; copy retains, destruction drops.
class LogicalBlockPtr {
    LogicalBlock* p_{};

public:
    LogicalBlockPtr() = default;
    explicit LogicalBlockPtr(LogicalBlock* p);
    LogicalBlockPtr(const LogicalBlockPtr& o);
    LogicalBlockPtr(LogicalBlockPtr&& o) noexcept: p_{std::exchange(o.p_, nullptr)} {}
    LogicalBlockPtr& operator=(LogicalBlockPtr o) noexcept
    {
        std::swap(p_, o.p_);
        return *this;
    }
    ~LogicalBlockPtr();

    LogicalBlock& operator*() const noexcept;  // defined after LogicalBlock is complete
    LogicalBlock* operator->() const noexcept
    {
        return p_;
    }
    LogicalBlock* get() const noexcept
    {
        return p_;
    }
    explicit operator bool() const noexcept
    {
        return p_ != nullptr;
    }
    friend bool operator==(const LogicalBlockPtr& a, const LogicalBlockPtr& b) noexcept
    {
        return a.p_ == b.p_;
    }
};

// Unique, non-atomic owning handle to a cache slot (engine-thread only).
// Destruction invalidates the slot. Precondition at destruction: the slot's
// allocation is already gone (memory release is a separate concern; the
// normal path is ReplayMemory, the release paths call Deallocate first).
class CacheBlockPtr {
    CacheBlock* p_{};

public:
    CacheBlockPtr() = default;
    explicit CacheBlockPtr(CacheBlock* p) noexcept: p_{p} {}
    CacheBlockPtr(const CacheBlockPtr&) = delete;
    CacheBlockPtr(CacheBlockPtr&& o) noexcept: p_{std::exchange(o.p_, nullptr)} {}
    CacheBlockPtr& operator=(CacheBlockPtr o) noexcept
    {
        std::swap(p_, o.p_);
        return *this;
    }
    ~CacheBlockPtr();

    CacheBlock& operator*() const noexcept;
    CacheBlock* operator->() const noexcept
    {
        return p_;
    }
    CacheBlock* get() const noexcept
    {
        return p_;
    }
    explicit operator bool() const noexcept
    {
        return p_ != nullptr;
    }
};

struct CacheBlock {
    uint64_t       timestamp{};    // eviction priority; zero means highest
    int            object_id{-1};  // ObjectAllocator registration id
    object_alloc_t allocation{};   // {const Allocation*}; .a == nullptr => no live allocation
    uint64_t       alloc_key{};    // snapshot of allocation->key at replay (ABA stale check)

    // Slot -> owning logical block (weak identity). Set at Create; persists
    // across evict/realloc. nullptr = sequence-owned (frontier).
    LogicalBlock* owner{};
    // Non-empty iff allocation is valid and owner is not null.
    LogicalBlockPtr pin;
    CacheBlockPool* mgr{};

    // Base of part `p`; `part` indexes the resolved Allocation.
    char* base(int part) const
    {
        return allocation->base(part);
    }
    int part_count() const
    {
        return allocation->part_count();
    }
    bool valid() const noexcept
    {
        return allocation.a != nullptr;
    }

    // Deallocates the backing object and clears the slot back to "no
    // allocation" state (the owner identity persists). Pre-condition: the
    // slot has a live allocation. Dropping the pin may recycle the owner and
    // invalidate this slot, so callers must not touch the slot after return.
    void Deallocate(ObjectAllocator& alloc);

    // Demote to evict-first priority: timestamp 0 sorts first in
    // SortedBlocks() and is below every eviction cutoff and pass floor.
    // Stamp never hands out 0 (next_timestamp_ starts at 1).
    void Demote() noexcept
    {
        TM_CHECK(valid());
        timestamp = 0;
    }
};

// nullptr replaces the old index-0 sentinel ("no slot").
inline bool is_valid(const CacheBlock* b) noexcept
{
    return b != nullptr && b->valid();
}

inline bool is_valid(const CacheBlockPtr& b) noexcept
{
    return is_valid(b.get());
}

class CacheBlockPool {
public:
    CacheBlockPtr Create(int object_id, LogicalBlock* owner = nullptr);

    // Eviction candidates: exactly the currently allocated blocks. The cached
    // allocation handle is the validity flag; the timestamp only orders the candidates.
    std::vector<CacheBlock*> SortedBlocks();

    uint64_t Stamp(const std::vector<CacheBlock*>& blocks);
    uint64_t Stamp(CacheBlock* b);

    size_t size() const noexcept
    {
        return blocks_.size() - free_.size();
    }

private:
    friend class CacheBlockPtr;

    // Owner destroyed; reset the slot and return it for reuse.
    void Invalidate(CacheBlock* b);

    uint64_t next_timestamp_{1};

    std::deque<CacheBlock>   blocks_;  // stable addresses; never shrinks
    std::vector<CacheBlock*> free_;
};

struct LogicalBlock {
    // Position and content extent within the sequence
    int offset{-1};
    int capacity{0};
    int size{0};  // filled tokens of an indexed node; 0 for private blocks

    // Intrusive strong refcount (requests + partial sibling edges + valid allocations)
    int               refs{0};
    LogicalBlockPool* mgr{};  // set at Create; used by handle / Retain / Drop

    // Cache slots, one per category; empty = not created. Destroying a handle
    // invalidates the slot.
    CacheBlockPtr prefix;
    CacheBlockPtr checkpoint;

    // Prefix trie node state (mutated only via the trie methods)
    const LogicalBlock*      parent{};  // nullptr = root; non-owning identity
    PrefixKey                key{};     // empty => not (yet) a prefix node
    std::vector<int>         tokens;
    std::vector<Fingerprint> image_fps;       // start-fingerprints of images beginning in this block (usually empty)
    bool                     indexed{false};  // present in the trie index

    // First-known indexed partial sibling at this block index: an identity-
    // verified node with the same parent and a strict token-prefix of this
    // block's content. Every edge points to a sibling with strictly smaller
    // `size` (a carrier indexed later by Finalize only grows), so
    // size strictly decreases along edge paths and the graph is acyclic.
    // First-wins: bound at most once, at AdmitPrompt, on a block created in the
    // same pass (mirrors trie first-wins insertion). Strong, RAII.
    LogicalBlockPtr partial;

    bool     is_valid{false};  // content proven produced; cleared on prefix evict
    uint64_t producer{0};      // request currently writing this range; 0 = none
};

// Owns logical block lifetime via an intrusive refcount. Nodes live in a
// deque with a free list (stable addresses, never shrinks), so a
// LogicalBlock* is a stable identity. When refs reaches 0 the node is
// recycled: a recycle hook removes it from the PrefixTrie index, every
// attached cache slot's allocation is already invalid (valid allocations hold
// refs through CacheBlock::pin), then destroying prefix/checkpoint handles
// invalidates their slots.
class LogicalBlockPool {
public:
    explicit LogicalBlockPool(int block_size = 0): block_size_{block_size} {}

    ~LogicalBlockPool();

    void ResetBlockSize(int block_size)
    {
        TM_CHECK_GT(block_size, 0);
        TM_CHECK_EQ(live_, 0);
        block_size_ = block_size;
    }

    int block_size() const noexcept
    {
        return block_size_;
    }

    void set_recycle_hook(std::function<void(LogicalBlock&)> h)
    {
        on_recycle_ = std::move(h);
    }

    LogicalBlockPtr Create(int logical_index);

    void Retain(LogicalBlock* p) noexcept
    {
        if (p) {
            ++p->refs;
        }
    }

    void Drop(LogicalBlock* p)  // sole decrement funnel
    {
        if (p) {
            TM_CHECK_GT(p->refs, 0);
            if (--p->refs == 0) {
                Recycle(p);
            }
        }
    }

    size_t size() const noexcept
    {
        return live_;
    }

private:
    void Recycle(LogicalBlock* p);  // sole place that frees a node

    int block_size_{};
    int live_{};

    std::deque<LogicalBlock>   nodes_;  // stable addresses; never shrinks
    std::vector<LogicalBlock*> free_;

    std::function<void(LogicalBlock&)> on_recycle_;
};

inline LogicalBlockPtr::LogicalBlockPtr(LogicalBlock* p): p_{p}
{
    if (p_) {
        p_->mgr->Retain(p_);
    }
}

inline LogicalBlockPtr::LogicalBlockPtr(const LogicalBlockPtr& o): p_{o.p_}
{
    if (p_) {
        p_->mgr->Retain(p_);
    }
}

inline LogicalBlockPtr::~LogicalBlockPtr()
{
    if (p_) {
        p_->mgr->Drop(p_);
    }
}

inline LogicalBlock& LogicalBlockPtr::operator*() const noexcept
{
    return *p_;
}

inline CacheBlockPtr::~CacheBlockPtr()
{
    if (p_) {
        TM_CHECK(!p_->valid());
        p_->mgr->Invalidate(p_);
    }
}

inline CacheBlock& CacheBlockPtr::operator*() const noexcept
{
    return *p_;
}

}  // namespace turbomind
