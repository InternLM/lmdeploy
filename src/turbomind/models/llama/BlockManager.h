// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <iterator>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

namespace turbomind {

// [L, H, S, D]

// [L, S/x, H, x, D]

struct Block {
    int      id;  // fixed linear id in the pool
    int      ref_count;
    uint64_t unique_id;  // unique for every block allocation
    uint64_t timestamp;
    void*    data;

    friend std::ostream& operator<<(std::ostream& os, const Block& block);
};

inline bool is_active(const Block& block)
{
    return block.ref_count > 0;
}

inline bool is_cached(const Block& block)
{
    return block.ref_count == 0 && block.timestamp > 0;
}

inline bool is_free(const Block& block)
{
    return block.ref_count == 0 && block.timestamp == 0;
}

struct Snapshot {
    int              active;
    int              cached;
    int              free;
    std::vector<int> ref_count;
};

class BlockManager {
public:
    explicit BlockManager(size_t block_size, double block_count, int chunk_size, IAllocator* allocator);

    ~BlockManager();

    // free -> active
    [[nodiscard]] std::vector<const Block*> Allocate(int count);

    // decrease ref count
    // active -> cached
    [[maybe_unused]] int Release(const std::vector<const Block*>& bs);

    // increase ref count
    // cached -> active
    void Retain(const std::vector<const Block*>& bs);

    // cached -> free
    void Evict(int count);

    // increase timestamp in reversed order
    void Touch(const std::vector<const Block*>& bs);

    Snapshot TakeSnapshot();

    int max_block_count() const noexcept
    {
        return max_block_count_;
    }

    int active_count() const noexcept
    {
        return active_ids_.size();
    }

    int cached_count() const noexcept
    {
        return cached_ids_.size();
    }

    int free_count() const noexcept
    {
        return (max_block_count_ - blocks_.size()) + free_ids_.size();
    }

    friend std::ostream& operator<<(std::ostream& os, const BlockManager&);

private:
    static size_t GetBlockCount(size_t block_size, double ratio);

    // move indices between sets
    static void Move(std::vector<int>& src, const std::vector<int>& delta, std::vector<int>& dst);

    // allocate a chunk of blocks
    bool Malloc();

private:
    size_t      block_size_;
    int         max_block_count_{};
    int         chunk_size_{};
    IAllocator* allocator_;

    std::vector<void*> chunks_;

    std::vector<int> active_ids_;
    std::vector<int> cached_ids_;
    std::vector<int> free_ids_;

    std::vector<Block> blocks_;  // < 100k

    // uint64_t unique_id_{1UL << 63};
    uint64_t unique_id_{0};
    uint64_t timestamp_{1};
};

}  // namespace turbomind