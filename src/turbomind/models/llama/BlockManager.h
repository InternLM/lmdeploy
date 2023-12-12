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
#include <sstream>
#include <unordered_map>
#include <vector>

namespace turbomind {

// [L, H, S, D]

// [L, S/x, H, x, D]

struct Block {
    int      id;         // fixed linear id in the pool
    int      use_count;  // active sequences using the block
    uint64_t unique_id;  // unique for every block allocation
    uint64_t timestamp;
    void*    data;

    friend std::ostream& operator<<(std::ostream& os, const Block& block);
    friend std::string   to_string(const Block& b)
    {
        std::stringstream ss;
        ss << b;
        return ss.str();
    }
};

using BlockIds  = std::vector<int>;
using UniqueIds = std::vector<uint64_t>;

inline bool is_active(const Block& block)
{
    // timestamp may be 0 for newly allocated block that has not been written
    return block.use_count > 0;
}

inline bool is_cached(const Block& block)
{
    return block.use_count == 0 && block.timestamp != 0;
}

inline bool is_free(const Block& block)
{
    return block.use_count == 0 && block.timestamp == 0;
}

struct Snapshot {
    int              active;
    int              cached;
    int              free;
    std::vector<int> use_count;
};

class BlockManager {
public:
    explicit BlockManager(size_t block_size, double block_count, int chunk_size, IAllocator* allocator);

    ~BlockManager();

    // free -> active (use_count = 1, ref_count = 1)
    [[nodiscard]] std::pair<BlockIds, UniqueIds> Allocate(int count);

    // cached -> active (use_count += 1)
    [[maybe_unused]] int Lock(const BlockIds& ids);

    // active -> cached (use_count -= 1)
    [[maybe_unused]] int Unlock(const BlockIds& ids);

    // cached -> free (ref_count = 0)
    void Evict(int count);

    // cached -> free (ref_count -= 1)
    void Free(BlockIds bs);

    // increase timestamp in reversed order
    void Touch(const BlockIds& bs);

    [[nodiscard]] int Verify(const BlockIds& block_ids, const UniqueIds& unique_ids);

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

    Block& block(int idx)
    {
        return blocks_[idx];
    }

    int unique_id(int idx)
    {
        return blocks_[idx].unique_id;
    }

    friend std::ostream& operator<<(std::ostream& os, const BlockManager&);

private:
    static size_t GetBlockCount(size_t block_size, double ratio);

    // move indices between sets
    static void Move(BlockIds& src, const BlockIds& delta, BlockIds& dst);

    // allocate a chunk of blocks
    bool Malloc();

private:
    size_t      block_size_;
    int         max_block_count_{};
    int         chunk_size_{};
    IAllocator* allocator_;

    std::vector<void*> chunks_;

    BlockIds active_ids_;
    BlockIds cached_ids_;
    BlockIds free_ids_;

    std::vector<Block> blocks_;  // < 100k

    uint64_t unique_id_{1};
    uint64_t timestamp_{1};
};

}  // namespace turbomind
