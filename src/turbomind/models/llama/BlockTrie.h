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
    explicit BlockTrie(size_t block_len_, std::shared_ptr<BlockManager> block_manager, bool enable_prefix_caching);

    bool enabled()
    {
        return enable_prefix_caching_;
    }

    // get cached blocks for sequence
    void match(Sequence& seq);

    // cache computed blocks for sequence
    void cache(const Sequence& seq);

    // remove invalid nodes, return valid count
    int verify();

private:
    int verify_traverse(std::shared_ptr<TrieNode>& node);

private:
    bool   enable_prefix_caching_;
    size_t block_seq_len_;

    std::shared_ptr<BlockManager> block_manager_;

    std::shared_ptr<TrieNode> root_;
};

}  // namespace turbomind
