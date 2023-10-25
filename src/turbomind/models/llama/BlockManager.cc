// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/debug_utils.h"
#include "src/turbomind/utils/logger.h"
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
        block.ref_count = 0;
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
    std::vector<int> src1(src.size() - delta.size());
    std::set_difference(src.begin(), src.end(), delta.begin(), delta.end(), src1.begin());
    src.swap(src1);

    std::vector<int> dst1(dst.size() + delta.size());
    std::set_union(dst.begin(), dst.end(), delta.begin(), delta.end(), dst1.begin());
    dst.swap(dst1);
}

std::vector<const Block*> BlockManager::Allocate(int count)
{
    while (free_ids_.size() < count) {
        if (!Malloc()) {
            throw std::runtime_error("out of memory");
        }
    }

    std::vector<const Block*> ret;

    std::vector<int> idxs(count);

    for (int i = 0; i < count; ++i) {
        int idx     = free_ids_[i];
        idxs[i]     = idx;
        auto& block = blocks_[idx];
        FT_CHECK(is_free(block));
        block.ref_count = 1;
        block.use_count = 1;
        block.unique_id = unique_id_++;
        ret.push_back(&block);
    }

    Move(free_ids_, idxs, active_ids_);

    dbg(free_ids_, active_ids_);

    return ret;
}

void BlockManager::Evict(int count)
{
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
        b.ref_count = 0;
        b.unique_id = 0;
        b.timestamp = 0;
    }

    Move(cached_ids_, idxs, free_ids_);

    dbg(cached_ids_, free_ids_);
}

int BlockManager::Free(const std::vector<const Block*>& bs)
{
    std::vector<int> idxs;

    for (const auto& p : bs) {
        auto& b = blocks_[p->id];
        FT_CHECK(is_cached(b));
        if (--b.ref_count == 0) {
            b.unique_id = 0;
            b.timestamp = 0;
            idxs.push_back(b.id);
        }
    }

    std::sort(idxs.begin(), idxs.end());

    Move(cached_ids_, idxs, free_ids_);

    dbg(cached_ids_, free_ids_);

    return idxs.size();
}

int BlockManager::Unlock(const std::vector<const Block*>& bs)
{
    std::vector<int> idxs;

    for (const auto& p : bs) {
        auto& block = blocks_[p->id];
        FT_CHECK(is_active(block));
        if (--block.use_count == 0) {
            idxs.push_back(block.id);
        }
    }

    std::sort(idxs.begin(), idxs.end());

    Move(active_ids_, idxs, cached_ids_);

    dbg(active_ids_, cached_ids_);

    return idxs.size();
}

int BlockManager::Lock(const std::vector<const Block*>& bs)
{
    std::vector<int> idxs;

    for (const auto& p : bs) {
        auto& block = blocks_[p->id];
        FT_CHECK(is_cached(block));
        if (++block.use_count == 1) {
            idxs.push_back(p->id);
        }
    }

    std::sort(idxs.begin(), idxs.end());

    Move(cached_ids_, idxs, active_ids_);

    dbg(cached_ids_, active_ids_);

    return idxs.size();
}

void BlockManager::Touch(const std::vector<const Block*>& bs)
{
    std::for_each(bs.crbegin(), bs.crend(), [this](const Block* p) {
        FT_CHECK(is_active(*p));
        const_cast<Block*>(p)->timestamp = timestamp_++;
    });
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