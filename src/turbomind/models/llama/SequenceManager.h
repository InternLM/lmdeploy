// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/BlockManager.h"

namespace turbomind {

struct Sequence {

    enum Status
    {
        kCached = 0,
        kLocked,
        kActive
    };

    uint64_t id;
    Status   status = kCached;

    std::vector<const Block*> blocks;
    std::vector<uint64_t>     block_unique_ids;

    int input_length = 0;

    mutable std::vector<int> tokens;  // update by user

    mutable int cache_len = 0;

    // additional data kept round-to-round
    mutable std::vector<std::byte> random_state;  // update by user

    mutable float rope_theta = 0.f;

    Sequence(uint64_t _id): id(_id) {}

    friend std::ostream& operator<<(std::ostream& os, const Sequence& seq);
};

using Sequences = std::vector<const Sequence*>;

inline std::ostream& operator<<(std::ostream& os, const Sequence& seq)
{
    os << "id=" << seq.id << ", status=" << seq.status << ", token_count=" << seq.tokens.size()
       << ", block_count=" << seq.blocks.size() << ", cache_len=" << seq.cache_len
       << ", random_state_size=" << seq.random_state.size();
    return os;
}

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

    [[nodiscard]] const Sequence* Create(uint64_t id);

    [[nodiscard]] const Sequence* Get(uint64_t id);

    [[nodiscard]] bool Contains(uint64_t id);

    [[nodiscard]] bool Erase(uint64_t id);

    void UpdateAndSetUnlock(const Sequence& seq);

    struct Outcome {
        int allocation;
        int swap_in;
        int swap_out;
    };

    using AdjustInputCount = std::function<std::pair<int, int>(const Sequences&, const std::vector<int>&)>;

    [[nodiscard]] Outcome Materialize(Sequences                    sequences,
                                      std::vector<int>             context_lengths,
                                      const std::vector<uint64_t>& priorities,
                                      int                          step_length,
                                      AdjustInputCount             adjust);

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
    void CommitUnlockAndFree();

    void VerifyAndLockCached(const Sequences& sequences);

    std::vector<int> CountRequiredBlocks(const Sequences&        sequences,  //
                                         const std::vector<int>& context_lengths,
                                         int                     step_length);

    static void SortByPriority(Sequences&                   sequences,  //
                               std::vector<int>&            context_lengths,
                               const std::vector<uint64_t>& priorities);

    static void AssignAndActivate(const Sequences&                 sequences,  //
                                  const std::vector<int>&          block_counts,
                                  const std::vector<const Block*>& blocks);

private:
    int    block_seq_len_;
    int    rank_;
    size_t val_offset_{};

    bool need_verify_{};

    // Use `std::map` to avoid reference invalidation
    std::map<uint64_t, Sequence> sequences_;

    std::unique_ptr<BlockManager> block_manager_;

    std::vector<const Block*> unlocked_;
    std::vector<const Block*> freed_;
};

inline std::ostream& operator<<(std::ostream& os, const SequenceManager::Outcome& oc)
{
    os << "allocation: " << oc.allocation << ", swap-in: " << oc.swap_in << ", swap-out: " << oc.swap_out;
    return os;
}

}  // namespace turbomind
