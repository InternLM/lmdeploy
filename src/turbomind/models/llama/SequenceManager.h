// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <functional>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/core.h"

#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/models/llama/BlockTrie.h"

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

    BlockIds  blocks;
    UniqueIds block_unique_ids;

    int input_length = 0;  // the number of tokens to be processed in each forward iter

    mutable std::vector<int> prompt;

    mutable std::vector<int> tokens;  // update by user or when the sequence is finished

    mutable int cache_len = 0;

    // additional data kept round-to-round
    mutable std::vector<std::byte> random_state;  // update by user

    mutable float rope_theta = 0.f;

    // embedding data
    mutable std::vector<Tensor> input_embeds;
    mutable std::vector<int>    input_embeds_offsets;

    explicit Sequence(uint64_t _id): id(_id) {}

    friend std::ostream& operator<<(std::ostream& os, const Sequence& seq);
};

using Sequences = std::vector<const Sequence*>;

inline std::ostream& operator<<(std::ostream& os, const Sequence& seq)
{
    os << "id=" << seq.id << ", status=" << seq.status << ", token_count=" << seq.tokens.size()
       << ", block_count=" << seq.blocks.size() << ", cache_len=" << seq.cache_len
       << ", random_state_size=" << seq.random_state.size() << ", input_length=" << seq.input_length;
    return os;
}

class SequenceManager {
public:
    // clang-format off
    struct BlockConfig {
        int head_dim_;
        int head_num_;
        int block_len_;
        int t_bits_;
        int q_bits_;
        int t_bits() const { return t_bits_; }
        int q_bits() const { return q_bits_; }
        int head_dim() const { return head_dim_; }
        int head_num() const { return head_num_; }
        int block_len() const { return block_len_; }
    };
    // clang-format on

    explicit SequenceManager(size_t             layer_num,
                             const BlockConfig& block_config,
                             double             block_count,
                             int                chunk_size,
                             bool               enable_prefix_caching,
                             int                rank,
                             int                attn_cp_size,
                             core::Allocator    allocator,
                             GetFreeMemSize     get_free_size);

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

    using AdjustInputCount = std::function<int(const Sequences&, const std::vector<int>&)>;

    //                50       1       0       50
    //    context = seq_len + beta = cache + alpha + input
    //     alpha' = input
    //      beta' = int(is_gen)
    //  -----------------------------------
    //   seq_len += output
    //     cache += input + output - 1  or  cache = seq_len - 1

    [[maybe_unused]] Outcome Materialize(Sequences             sequences,
                                         std::vector<int>      context_length,
                                         std::vector<int>      alpha,
                                         std::vector<uint64_t> priorities,
                                         int                   max_fwd_tokens,
                                         int                   max_tmp_tokens);

    /** @brief cache the input prompt tokens of each seq in sequences[0:active_size-1]
     *
     * @param sequences The sequence list
     * @param active_size the number of active sequences in the list
     */
    void CachePrompt(const Sequences& sequences, int active_size);

    /** @brief cache the generated tokens of a given sequence
     *
     * @param sequence the given sequence
     *
     * @note This function can only be called after the sequence finish generation
     * and all tokens including the prompt tokens and generated tokens have been put to
     * `seq.tokens`
     */
    void CacheGeneration(const Sequence& sequence);

    [[nodiscard]] void* GetBlockPtr(int block_id)
    {
        return block_manager_->block(block_id).data;
    }

    int max_block_count() const noexcept
    {
        return block_manager_->max_block_count();
    }

    int total_count() const noexcept
    {
        return block_manager_->total_count();
    }

    int active_count() const noexcept
    {
        return block_manager_->active_count();
    }

    int free_count() const noexcept
    {
        return block_manager_->free_count();
    }

    int cached_count() const noexcept
    {
        return block_manager_->cached_count();
    }

    // return #total_seq, #active_seq, #cached_seq
    std::tuple<int, int, int> seq_stats() const noexcept;

private:
    void Erase(std::map<uint64_t, Sequence>::iterator& it);

    void CommitUnlockAndFree();

    void VerifyAndLockCached(const Sequences& sequences);

    std::vector<int> CountRequiredBlocks(const Sequences&        sequences,  //
                                         const std::vector<int>& context_length);

    static void AssignAndActivate(const Sequences&        sequences,  //
                                  const std::vector<int>& counts,
                                  const BlockIds&         blocks,
                                  const UniqueIds&        unique_ids);

    void PrefixMatch(Sequences& sequences, const std::vector<int>& alpha);

private:
    int block_seq_len_;
    int rank_;
    int attn_cp_size_;

    // Use `std::map` to avoid reference invalidation
    std::map<uint64_t, Sequence> sequences_;

    std::shared_ptr<BlockManager> block_manager_;
    std::shared_ptr<BlockTrie>    block_trie_;

    BlockIds unlocked_;
    BlockIds freed_;
};

inline std::ostream& operator<<(std::ostream& os, const SequenceManager::Outcome& oc)
{
    os << "allocation: " << oc.allocation << ", swap-in: " << oc.swap_in << ", swap-out: " << oc.swap_out;
    return os;
}

}  // namespace turbomind
