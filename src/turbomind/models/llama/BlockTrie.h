// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/BlockManager.h"
#include <algorithm>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
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
    std::shared_ptr<TrieNode>                             parent;
};

class BlockTrie {
public:
    explicit BlockTrie(size_t block_len_, std::shared_ptr<BlockManager> block_manager, bool enable_prefix_caching);
    ~BlockTrie();

    bool enabled()
    {
        return enable_prefix_caching_;
    }

    // get cached blocks for sequence
    void match(Sequence& seq);

    // cache computed blocks for sequence
    void cache(const Sequence& seq);

    // evict unused nodes
    int evict(const int num);

private:
    bool   enable_prefix_caching_;
    int    cached_block_num_;
    size_t block_seq_len_;

    std::shared_ptr<BlockManager> block_manager_;

    std::unordered_set<std::shared_ptr<TrieNode>> leaves_;
    std::shared_ptr<TrieNode>                     root_;
};

}  // namespace turbomind
