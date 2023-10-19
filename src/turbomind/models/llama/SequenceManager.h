// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/BlockManager.h"

namespace turbomind {

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
    explicit SequenceManager(size_t      layer_num,
                             size_t      head_num,
                             size_t      head_dim,
                             size_t      block_seq_len,
                             double      block_count,
                             int         chunk_size,
                             size_t      elem_bits,
                             int         rank,
                             IAllocator* allocator);

    SequenceManager(const SequenceManager&)     = delete;
    SequenceManager(SequenceManager&&) noexcept = default;

    const Sequence* Create(uint64_t id);

    const Sequence* Get(uint64_t id);

    bool Contains(uint64_t id);

    bool Erase(uint64_t id);

    void Release(const Sequence& seq);

    struct Outcome {
        int allocation;
        int swap_in;
        int swap_out;
    };

    Outcome Materialize(const std::vector<const Sequence*>& sequences,
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
    void Verify(Sequence& seq, std::vector<const Block*>& retain);

private:
    int    block_seq_len_;
    int    rank_;
    size_t val_offset_{};

    bool need_verification_{};

    // Use `std::map` to avoid reference invalidation
    std::map<uint64_t, Sequence> sequences_;

    std::unique_ptr<BlockManager> block_manager_;

    std::vector<const Block*> released_;
};

inline std::ostream& operator<<(std::ostream& os, const SequenceManager::Outcome& oc)
{
    os << "allocation: " << oc.allocation << ", swap-in: " << oc.swap_in << ", swap-out: " << oc.swap_out;
    return os;
}

}  // namespace turbomind