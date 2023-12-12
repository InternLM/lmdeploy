// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/debug_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/string_utils.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace turbomind {

BlockManager::BlockManager(size_t block_size, double block_count, int chunk_size, IAllocator* allocator):
    block_size_(block_size), allocator_(allocator)
{
    if (block_count < 1.) {
        max_block_count_ = GetBlockCount(block_size, block_count);
    }
    else {
        max_block_count_ = block_count;
    }

    if (chunk_size == 0) {
        chunk_size_ = static_cast<int>(std::sqrt(max_block_count_));
    }
    else if (chunk_size < 0) {
        chunk_size_ = max_block_count_;
    }
    else {
        chunk_size_ = chunk_size;
    }

    TM_LOG_INFO("[BlockManager] block_size = %lu MB", (unsigned long)block_size_ >> 20);
    TM_LOG_INFO("[BlockManager] max_block_count = %d", max_block_count_);
    TM_LOG_INFO("[BlockManager] chunk_size = %d", chunk_size_);

    blocks_.reserve(max_block_count_);

    active_ids_.reserve(max_block_count_);
    cached_ids_.reserve(max_block_count_);
    free_ids_.reserve(max_block_count_);

    // pre-allocate first chunk
    Malloc();
    dbg(free_ids_);
}

BlockManager::~BlockManager()
{
    for (auto& chunk : chunks_) {
        allocator_->free(&chunk);
    }
}

bool BlockManager::Malloc()
{
    auto chunk_size = std::min<int>(chunk_size_, max_block_count_ - blocks_.size());

    if (!chunk_size) {
        return false;
    }

    auto ptr = (std::byte*)allocator_->malloc(block_size_ * chunk_size);
    if (!ptr) {
        return false;
    }

    chunks_.push_back(ptr);

    for (int i = 0; i < chunk_size; ++i, ptr += block_size_) {
        auto& block     = blocks_.emplace_back();
        block.use_count = 0;
        block.id        = (int)blocks_.size() - 1;
        block.timestamp = 0;
        block.data      = ptr;

        free_ids_.push_back(block.id);
    }

    return true;
}

size_t BlockManager::GetBlockCount(size_t block_size, double ratio)
{
    size_t free{};
    size_t total{};
    check_cuda_error(cudaMemGetInfo(&free, &total));
    return static_cast<size_t>(total * ratio) / block_size;
}

void BlockManager::Move(std::vector<int>& src, const std::vector<int>& delta, std::vector<int>& dst)
{
    FT_CHECK(src.size() >= delta.size());
    std::vector<int> src1(src.size() - delta.size());
    {
        auto end = std::set_difference(src.begin(), src.end(), delta.begin(), delta.end(), src1.begin());
        FT_CHECK(end == src1.end());
    }
    src.swap(src1);

    std::vector<int> dst1(dst.size() + delta.size());
    {
        auto end = std::set_union(dst.begin(), dst.end(), delta.begin(), delta.end(), dst1.begin());
        FT_CHECK(end == dst1.end());
    }
    dst.swap(dst1);
}

auto BlockManager::Allocate(int count) -> std::pair<BlockIds, UniqueIds>
{
    while (free_ids_.size() < count) {
        if (!Malloc()) {
            throw std::runtime_error("out of memory");
        }
    }

    BlockIds  block_ids(count);
    UniqueIds unique_ids(count);

    for (int i = 0; i < count; ++i) {
        int   idx = free_ids_[i];
        auto& b   = blocks_[idx];
        FT_CHECK(is_free(b));  // pre-condition: uc == 0 && ts == 0
        b.use_count = 1;
        b.unique_id = unique_id_++;
        FT_CHECK(is_active(b));  // post-condition
        block_ids[i]  = idx;
        unique_ids[i] = b.unique_id;
    }

    Move(free_ids_, block_ids, active_ids_);

    dbg(free_ids_, active_ids_);

    return {block_ids, unique_ids};
}

void BlockManager::Evict(int count)
{
    FT_CHECK(count <= cached_ids_.size());
    std::vector<int> idxs(cached_ids_);
    // get first `count` cached ids according to timestamp
    std::nth_element(idxs.begin(), idxs.begin() + count, idxs.end(), [&](int i, int j) {
        return blocks_[i].timestamp < blocks_[j].timestamp;
    });
    idxs.resize(count);

    // sort the retrieved ids
    std::sort(idxs.begin(), idxs.end());

    // set as free
    for (const auto& idx : idxs) {
        auto& b = blocks_[idx];
        FT_CHECK(is_cached(b));
        b.unique_id = 0;
        b.timestamp = 0;
        FT_CHECK(is_free(b));
    }

    Move(cached_ids_, idxs, free_ids_);

    dbg(cached_ids_, free_ids_);
}

void BlockManager::Free(BlockIds ids)
{
    std::sort(ids.begin(), ids.end());

    for (const auto& i : ids) {
        auto& b = blocks_[i];
        FT_CHECK(is_cached(b));  // uc == 0 && ts != 0
        b.unique_id = 0;
        b.timestamp = 0;
        FT_CHECK(is_free(b));
    }

    Move(cached_ids_, ids, free_ids_);
}

int BlockManager::Unlock(const BlockIds& ids)
{
    BlockIds unlock;
    unlock.reserve(ids.size());

    for (const auto& i : ids) {
        auto& b = blocks_[i];
        FT_CHECK(is_active(b));  // pre-condition: uc > 0
        if (--b.use_count == 0) {
            unlock.push_back(b.id);
            FT_CHECK(is_cached(b));  // post-condition
        }
    }

    std::sort(unlock.begin(), unlock.end());

    Move(active_ids_, unlock, cached_ids_);

    dbg(active_ids_, cached_ids_);
    return unlock.size();
}

int BlockManager::Lock(const BlockIds& ids)
{
    BlockIds lock;
    lock.reserve(ids.size());

    for (const auto& i : ids) {
        auto& b = blocks_[i];
        FT_CHECK_WITH_INFO(is_cached(b), to_string(b));
        if (++b.use_count == 1) {
            lock.push_back(i);
            FT_CHECK(is_active(b));
        }
    }

    std::sort(lock.begin(), lock.end());

    Move(cached_ids_, lock, active_ids_);

    // dbg(cached_ids_, active_ids_);

    return lock.size();
}

void BlockManager::Touch(const BlockIds& ids)
{
    std::for_each(ids.crbegin(), ids.crend(), [this](int i) {
        FT_CHECK(is_active(blocks_[i]));
        blocks_[i].timestamp = timestamp_++;
    });
}

int BlockManager::Verify(const std::vector<int>& block_ids, const std::vector<uint64_t>& unique_ids)
{
    FT_CHECK(block_ids.size() == unique_ids.size());
    int valid = block_ids.size();
    for (int i = 0; i < block_ids.size(); ++i) {
        if (unique_id(block_ids[i]) != unique_ids[i]) {
            valid = i;
            break;
        }
    }
    int miss = 0;
    for (int i = valid; i < block_ids.size(); ++i) {
        miss += (unique_id(block_ids[i]) != unique_ids[i]);
    }
    // All later blocks should have been invalidated
    FT_CHECK_WITH_INFO(miss == (int)block_ids.size() - valid,
                       fmtstr("count = %d, valid = %d, miss = %d", (int)block_ids.size(), valid, miss));
    return valid;
}

Snapshot BlockManager::TakeSnapshot()
{
    std::vector<int> use_count(blocks_.size());
    for (const auto& idx : active_ids_) {
        use_count[idx] = blocks_[idx].use_count;
    }
    return {active_count(), cached_count(), free_count(), std::move(use_count)};
}

std::ostream& operator<<(std::ostream& os, const BlockManager& manager)
{
    os << "block_size: " << manager.block_size_ << ", ";
    os << "max_block_count: " << manager.max_block_count_ << ", ";
    os << "chunk_size: " << manager.chunk_size_ << ", ";
    os << "chunks: " << manager.chunks_.size() << ", ";
    os << "active_ids: " << manager.active_ids_.size() << ", ";
    os << "cached_ids: " << manager.cached_ids_.size() << ", ";
    os << "free_ids: " << manager.free_ids_.size() << ", ";
    os << "blocks: " << manager.blocks_.size() << ", ";
    os << "unique_id: " << manager.unique_id_ << ", ";
    os << "timestamp: " << manager.timestamp_ << ", ";
    os << "allocator: " << manager.allocator_;
    return os;
}

std::ostream& operator<<(std::ostream& os, const Block& block)
{
    os << "id=" << block.id << ", use_count=" << block.use_count << ", unique_id=" << block.unique_id
       << ", timestamp=" << block.timestamp << ", data=" << block.data;
    return os;
}

}  // namespace turbomind
