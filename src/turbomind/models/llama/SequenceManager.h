#pragma once

#include "src/turbomind/models/llama/BlockManager.h"

namespace turbomind {

// |<-- active -->|<-- pending -->|<-- new -->|

struct Sequence {

    enum Status {
        kCached = 0,
        kLocked,
        kActive
    };

    uint64_t id;
    Status   status;

    std::vector<const Block*> blocks;
    std::vector<uint64_t>     block_unique_ids;

    mutable std::vector<int> tokens;  // update by user

    mutable int cache_len;

    // additional data kept round-to-round
    mutable std::vector<std::byte> random_state;  // update by user

    friend std::ostream& operator<<(std::ostream& os, const Sequence& seq);
};

class SequenceManager {
public:
    // allocate slack blocks to reduce block manager overhead
    static constexpr int kSlackBlockNum = 1;

    explicit SequenceManager(size_t      layer_num,
                             size_t      head_num,
                             size_t      head_dim,
                             size_t      block_len,
                             double      block_count,
                             int         chunk_size,
                             size_t      elem_bits,
                             int         rank,
                             IAllocator* allocator);

    const Sequence* Create(uint64_t id);

    const Sequence* Fetch(uint64_t id);

    void Update(const Sequence& seq);

    bool Erase(uint64_t id);

    bool Contains(uint64_t id);

    bool Materialize(const std::vector<const Sequence*>& sequences,
                     const std::vector<int>&             context_lengths,
                     const std::vector<uint64_t>&        priorities,
                     int                                 step_length);

    void* OffsetKey(void* block_ptr)
    {
        return block_ptr;
    }

    void* OffsetVal(void* block_ptr)
    {
        return (std::byte*)block_ptr + val_offset_;
    }

    int max_block_count() const noexcept
    {
        return block_manager_->max_block_count();
    }

private:
    void VerifyBlocks(Sequence& seq);

private:
    int    block_len_;
    int    rank_;
    size_t val_offset_{};

    bool need_verification_{};

    // Use `std::map` to avoid reference invalidation
    std::map<uint64_t, Sequence> sequences_;

    std::unique_ptr<BlockManager> block_manager_;

    std::vector<const Block*> released_;
};

}  // namespace turbomind