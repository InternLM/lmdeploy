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
};

class BlockTrie {
public:
    explicit BlockTrie(size_t block_len, std::shared_ptr<BlockManager> block_manager);

    /** @brief Attempt to match cached key-value (KV) blocks for a given sequence.
    *
    * This function iterates the tokens of the sequence and attempts
    * to match them with the cached KV blocks. If the max prefix match is found,
    * it returns the IDs, unique IDs, and hash keys of the matched blocks.
    *
    * @param seq The sequence whose tokens are to be matched against the cached KV blocks.
    * @return A tuple containing the following:
    *         - BlockIds: A list of IDs of the matched blocks.
    *         - UniqueIds: A list of unique IDs of the matched blocks.
    *         - std::vector<std::shared_ptr<TrieNode>>: A list of matched node
    *
    * @note If no blocks are matched, all containers in the returned tuple will be empty.
    */
   std::tuple<BlockIds, UniqueIds, std::vector<std::shared_ptr<TrieNode>>> match(const Sequence& seq);

    /**
     * @brief Cache the key-value (KV) blocks of a given sequence.
     *
     * This function caches the KV blocks of the specified sequence. Only valid blocks
     * of a sequence whose status is NOT `Sequence::kCached` are considered
     * for caching.
     *
     * @param seq The sequence whose KV blocks are to be cached.
     * @param tokens The token list that the quence's KV blocks map
     * @return A pair of vectors containing the IDs and unique IDs of the successfully
              cached blocks. If no blocks are cached, a pair of empty vectors is returned.
     *
     * @note Only valid blocks of a non-kCached sequence are processed for caching.
     */
    std::pair<BlockIds, UniqueIds> cache(const Sequence& seq, const std::vector<int>& tokens);

private:
    size_t block_seq_len_;

    std::shared_ptr<BlockManager> block_manager_;

    std::shared_ptr<TrieNode> root_;
};

}  // namespace turbomind
