// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

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
};

class BlockTrie {
public:
    explicit BlockTrie(size_t block_len, std::shared_ptr<BlockManager> block_manager);

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
    std::tuple<BlockIds, UniqueIds> Match(const Sequence& seq);

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
    void DFS(std::shared_ptr<TrieNode>& node);

private:
    size_t block_seq_len_;

    std::shared_ptr<BlockManager> block_manager_;

    std::shared_ptr<TrieNode> root_;
};

}  // namespace turbomind
