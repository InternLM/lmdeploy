// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/BlockManager.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace turbomind {

struct Sequence;

struct TrieNode {
    std::unordered_map<size_t, std::shared_ptr<TrieNode>> children;
    size_t                                                hash_key;
    std::vector<int>                                      tokens;
    int                                                   block_id;
    uint64_t                                              block_unique_id;
    int                                                   num_matched;
    int                                                   linear_state_slot = -1;
};

struct BlockTrieMatch {
    BlockIds  block_ids;
    UniqueIds unique_ids;
    int       linear_cache_len = 0;
    Tensor    conv_states;
    Tensor    recurrent_states;
};

class BlockTrie {
public:
    explicit BlockTrie(size_t                        block_len,
                       std::shared_ptr<BlockManager> block_manager,
                       int                           linear_prefix_cache_interval_blocks = 0,
                       int                           linear_state_slot_capacity          = 0,
                       std::vector<ssize_t>          conv_state_shape                    = {},
                       DataType                      conv_state_dtype                    = {},
                       std::vector<ssize_t>          recurrent_state_shape               = {},
                       DataType                      recurrent_state_dtype               = {});

    /**
     * @brief Attempt to match cached key-value (KV) blocks for a given sequence.
     *
     * This function iterates the tokens of the sequence and attempts
     * to match them with the cached KV blocks. If the max prefix match is found,
     * it returns the IDs, unique IDs of the matched blocks.
     *
     * @param seq The sequence whose tokens are to be matched against the cached KV blocks.
     * @return A tuple containing the following:
     *         - BlockIds: A list of IDs of the matched blocks.
     *         - UniqueIds: A list of unique IDs of the matched blocks.
     *
     * @note If no blocks are matched, all containers in the returned tuple will be empty.
     */
    BlockTrieMatch Match(const Sequence& seq);

    /**
     * @brief Cache the key-value (KV) blocks of a given sequence.
     *
     * This function caches the KV blocks of the specified sequence. Only valid blocks
     * of a sequence whose status is NOT `Sequence::kCached` are considered
     * to be cached
     *
     * @param seq The sequence whose KV blocks are to be cached.
     * @param tokens The token list corresponding to the KV blocks
     * @return A tuple containing the following:
     *         - BlockIds: A list of IDs of the cached blocks.
     *         - UniqueIds: A list of unique IDs of the cached blocks.
     */
    std::tuple<BlockIds, UniqueIds> Cache(const Sequence& seq, const std::vector<int>& tokens);

    /**
     * @brief remove invalid nodes
     */
    void Verify();

private:
    void   DFS(std::shared_ptr<TrieNode>& node);
    void   ReleaseLinearPrefixState(std::shared_ptr<TrieNode>& node);
    bool   IsLinearCheckpointNode(int num_matched) const;
    int    AcquireLinearStateSlot();
    void   ReleaseLinearStateSlot(int slot);
    Tensor LinearConvState(int slot) const;
    Tensor LinearRecurrentState(int slot) const;

private:
    size_t block_seq_len_;

    std::shared_ptr<BlockManager> block_manager_;

    std::shared_ptr<TrieNode> root_;

    int                  linear_prefix_cache_interval_blocks_{};
    int                  linear_prefix_cache_interval_tokens_{};
    int                  linear_state_slot_capacity_{};
    std::vector<ssize_t> conv_state_shape_;
    std::vector<ssize_t> recurrent_state_shape_;
    DataType             conv_state_dtype_{};
    DataType             recurrent_state_dtype_{};
    std::vector<Tensor>  linear_conv_states_;
    std::vector<Tensor>  linear_recurrent_states_;
    std::vector<int>     free_linear_state_slots_;
    bool                 warned_linear_state_pool_exhausted_{false};
    bool                 warned_linear_state_pool_oom_{false};
};

}  // namespace turbomind
