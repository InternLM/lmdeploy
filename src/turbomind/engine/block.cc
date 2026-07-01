#include "src/turbomind/engine/block.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace turbomind {

void CacheBlockPool::Invalidate(int index)
{
    TM_CHECK_GT(index, 0);
    TM_CHECK_LT(index, static_cast<int>(blocks_.size()));
    TM_CHECK_GE(blocks_[index].object_id, 0);
    blocks_[index] = {};
    free_list_.push_back(index);
}

int CacheBlockPool::Create(int object_id, LogicalBlock* owner)
{
    TM_CHECK_GE(object_id, 0);
    if (TM_UNLIKELY(free_list_.empty())) {
        free_list_.push_back(static_cast<int>(blocks_.size()));
        blocks_.emplace_back();
    }

    const int idx = free_list_.back();
    free_list_.pop_back();
    blocks_[idx]           = {};
    blocks_[idx].object_id = object_id;
    blocks_[idx].owner     = owner;
    return idx;
}

void CacheBlockPool::Deallocate(ObjectAllocator& alloc, int cache_id)
{
    auto& c = blocks_[cache_id];
    TM_CHECK(c.valid());
    alloc.Deallocate(c.object_id, c.allocation);
    c.allocation = {};
    c.alloc_key  = 0;
    c.timestamp  = 0;
}

std::vector<int> CacheBlockPool::SortedIndices() const
{
    std::vector<int> idxs;
    idxs.reserve(blocks_.size());
    for (int i = 1; i < static_cast<int>(blocks_.size()); ++i) {
        if (blocks_[i].valid()) {
            idxs.push_back(i);
        }
    }
    std::sort(idxs.begin(), idxs.end(), [this](int i, int j) { return blocks_[i].timestamp < blocks_[j].timestamp; });
    return idxs;
}

uint64_t CacheBlockPool::Stamp(const std::vector<int>& cache_ids)
{
    const auto ret = next_timestamp_;
    for (auto it = cache_ids.rbegin(); it != cache_ids.rend(); ++it) {
        if (*it) {
            Stamp(*it);
        }
    }
    return ret;
}

uint64_t CacheBlockPool::Stamp(int cache_id)
{
    TM_CHECK_GT(cache_id, 0);
    TM_CHECK_LT(cache_id, static_cast<int>(blocks_.size()));
    TM_CHECK_GE(blocks_[cache_id].object_id, 0);
    blocks_[cache_id].timestamp = next_timestamp_++;
    return blocks_[cache_id].timestamp;
}

LogicalBlockPool::~LogicalBlockPool()
{
    if (live_ != 0) {
        TM_LOG_ERROR("leaked {} logical blocks", live_);
    }
}

BlockHandle LogicalBlockPool::Create(int logical_index)
{
    TM_CHECK_GT(block_size_, 0);
    TM_CHECK_GE(logical_index, 0);

    LogicalBlock* p = alloc_.allocate(1);
    std::allocator_traits<NodeAlloc>::construct(alloc_, p);
    p->mgr      = this;
    p->offset   = logical_index * block_size_;
    p->capacity = block_size_;
    ++live_;
    return BlockHandle{p};  // refs 0 -> 1
}

void LogicalBlockPool::Recycle(LogicalBlock* p)
{
    if (on_recycle_) {
        on_recycle_(*p);  // PrefixTrie::Erase (pool stays prefix-agnostic)
    }
    if (const int c = p->prefix_id) {
        cache_.Invalidate(c);  // allocation already gone (see class comment)
    }
    if (const int c = p->checkpoint_id) {
        cache_.Invalidate(c);
    }
    std::allocator_traits<NodeAlloc>::destroy(alloc_, p);  // ~LogicalBlock drops fork edges, frees tokens
    alloc_.deallocate(p, 1);                               // back to the pmr pool
    --live_;
}

}  // namespace turbomind
