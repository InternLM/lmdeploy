// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <functional>
#include <tuple>
#include <unordered_map>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/core.h"

#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/models/llama/BlockTrie.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

struct RequestCache;

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

    // Gated DeltaNet linear attention persistent states (e.g. Qwen3.5-MoE).
    // Allocated on first request, preserved across requests for the same session,
    // and freed automatically when the sequence is erased from the SequenceManager.
    //   conv_states:      (num_linear_layers, conv_dim, d_conv) — per-channel rolling conv history
    //   recurrent_states: (num_linear_layers, num_v_heads, key_head_dim, value_head_dim) — SSM state
    mutable Tensor   conv_states;
    mutable Tensor   recurrent_states;
    mutable bool     linear_states_need_reset          = false;
    mutable int      linear_restore_snapshot_slot      = -1;
    mutable uint64_t linear_restore_snapshot_unique_id = 0;

    // Per-forward staged snapshots for newly completed reusable cache blocks.
    // Leading dim is this pass's block_count; row r holds absolute block (staged_linear_block_begin + r).
    // Layouts:
    //   staged_conv_snapshots:      (block_count, num_linear_layers, d_conv, conv_dim)
    //   staged_recurrent_snapshots: (block_count, num_linear_layers, num_v_heads, key_head_dim, value_head_dim)
    mutable Tensor               staged_conv_snapshots;
    mutable Tensor               staged_recurrent_snapshots;
    mutable std::vector<uint8_t> staged_linear_block_valid;
    mutable int                  staged_linear_block_begin = 0;
    mutable int                  staged_linear_block_count = 0;

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
        bool share_kv_;
        int t_bits() const { return t_bits_; }
        int q_bits() const { return q_bits_; }
        int head_dim() const { return head_dim_; }
        int head_num() const { return head_num_; }
        int block_len() const { return block_len_; }
        bool is_share_kv() const { return share_kv_; }
    };
    // clang-format on

    explicit SequenceManager(const ModelParam& model_param,
                             DataType          runtime_dtype,
                             int               cache_block_seq_len,
                             int               attn_tp_size,
                             int               max_batch_size,
                             double            block_count,
                             int               chunk_size,
                             bool              enable_prefix_caching,
                             int               rank,
                             int               attn_cp_size,
                             core::Allocator   allocator,
                             GetFreeMemSize    get_free_size);

    SequenceManager(const SequenceManager&)     = delete;
    SequenceManager(SequenceManager&&) noexcept = default;

    [[nodiscard]] const Sequence* Create(uint64_t id);

    [[nodiscard]] const Sequence* Get(uint64_t id);

    [[nodiscard]] bool Contains(uint64_t id);

    [[nodiscard]] bool Erase(uint64_t id);

    void PrepareLinearCheckpointStaging(RequestCache& cache);

    /** @brief Best-effort prefix-cache counters: publish_ok, publish_miss, publish_pool_exhausted, alpha_skip,
     * linear_restore. */
    [[nodiscard]] static std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> LinearPrefixCacheStats() noexcept;

    void AcquireLinearStateSlot(const Sequence& seq);

    void ReleaseLinearStateSlot(const Sequence& seq);

    void InvalidateStatesAndCache(const Sequence& seq);

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

    int block_seq_len() const noexcept
    {
        return block_seq_len_;
    }

    // return #total_seq, #active_seq, #cached_seq
    std::tuple<int, int, int> seq_stats() const noexcept;

private:
    void Erase(std::map<uint64_t, Sequence>::iterator& it);

    void CommitUnlockAndFree();

    void InvalidateStatesAndCache(const Sequence& seq, BlockIds& freed_blocks);

    void ClearLinearSnapshotStaging(const Sequence& seq);

    [[nodiscard]] bool IsLinearSnapshotValid(int slot, uint64_t unique_id) const;

    [[nodiscard]] std::pair<int, uint64_t> PublishLinearSnapshot(const Sequence& seq, int block_idx, int slot_hint);

    void ReleaseLinearSnapshot(int slot, uint64_t unique_id);

    /// Copy trie-referenced linear snapshot into the sequence. Returns false if the slot is stale or tensors missing.
    bool RestoreLinearSnapshot(const Sequence& seq, int slot, uint64_t unique_id);

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

    // Runtime working-state pool: one slot per active sequence.
    Tensor                            pooled_conv_states_;
    Tensor                            pooled_recurrent_states_;
    std::vector<int>                  free_linear_state_slots_;
    std::unordered_map<uint64_t, int> seq_to_linear_state_slot_;

    // Cached prefix-state snapshot pool: one slot per reusable cache block/trie node.
    Tensor                pooled_prefix_conv_snapshots_;
    Tensor                pooled_prefix_recurrent_snapshots_;
    std::vector<int>      free_linear_snapshot_slots_;
    std::vector<uint64_t> linear_snapshot_unique_ids_;
    uint64_t              next_linear_snapshot_unique_id_ = 1;

    int      num_linear_layers_               = 0;
    int      d_conv_                          = 0;
    int      conv_dim_                        = 0;
    int      num_v_heads_                     = 0;
    int      key_head_dim_                    = 0;
    int      value_head_dim_                  = 0;
    DataType linear_conv_dtype_               = {};
    DataType linear_state_dtype_              = {};
    size_t   linear_active_pool_bytes_        = 0;
    size_t   linear_snapshot_bytes_per_block_ = 0;

    BlockIds unlocked_;
    BlockIds freed_;
};

inline std::ostream& operator<<(std::ostream& os, const SequenceManager::Outcome& oc)
{
    os << "allocation: " << oc.allocation << ", swap-in: " << oc.swap_in << ", swap-out: " << oc.swap_out;
    return os;
}

}  // namespace turbomind
