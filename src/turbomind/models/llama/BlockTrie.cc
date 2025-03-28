// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/BlockTrie.h"
#include "src/turbomind/models/llama/SequenceManager.h"

namespace turbomind {

size_t hash(const std::vector<int>& vec)
{
    size_t seed = vec.size();
    for (const auto& i : vec) {
        seed ^= std::hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

BlockTrie::BlockTrie(size_t block_len): block_seq_len_(block_len)
{
    root_ = std::make_shared<TrieNode>();
}

std::tuple<BlockIds, UniqueIds, std::vector<std::shared_ptr<TrieNode>>> BlockTrie::Match(const Sequence& seq) const
{
    BlockIds                               matched_blocks;
    UniqueIds                              matched_unique_ids;
    std::vector<std::shared_ptr<TrieNode>> matched_nodes;

    std::shared_ptr<TrieNode> curr_node   = root_;
    int                       num_matched = 0;

    while (num_matched + block_seq_len_ < seq.prompt.size()) {
        std::vector<int> curr_tokens(seq.prompt.begin() + num_matched,
                                     seq.prompt.begin() + num_matched + block_seq_len_);
        size_t           hash_key = hash(curr_tokens);

        auto it = curr_node->children.find(hash_key);
        if (it == curr_node->children.end()) {
            break;
        }
        if (curr_tokens != it->second->tokens) {
            TM_LOG_WARNING("hash key cache hit, but tokens are not the same");
            break;
        }

        matched_blocks.emplace_back(it->second->block_id);
        matched_unique_ids.emplace_back(it->second->block_unique_id);
        matched_nodes.emplace_back(it->second);
        curr_node = it->second;
        num_matched += block_seq_len_;
    }
    return std::make_tuple(matched_blocks, matched_unique_ids, matched_nodes);
}

std::tuple<BlockIds, UniqueIds, std::vector<std::shared_ptr<TrieNode>>> BlockTrie::Cache(const Sequence&         seq,
                                                                                         const std::vector<int>& tokens)
{
    TM_LOG_INFO("[BlockTrie][cache] session %llu, seq.blocks %d, tokens %d", seq.id, seq.blocks.size(), tokens.size());
    FT_CHECK(seq.status != Sequence::kCached);
    FT_CHECK(tokens.size() <= seq.blocks.size() * block_seq_len_);

    std::shared_ptr<TrieNode> curr_node = root_;
    int                       idx       = 0;

    BlockIds                               cache_block_ids;
    UniqueIds                              cache_block_unique_ids;
    std::vector<std::shared_ptr<TrieNode>> cache_nodes;

    // We don't cache the last block of the sequence, since it might not be full
    // TODO(lvhan): determine wether the last block is full or not. It is not trivial
    // considering chunk prefill
    for (int idx = 0; idx < seq.blocks.size() - 1; ++idx) {
        auto start = tokens.begin() + idx * block_seq_len_;
        auto end   = start + block_seq_len_;

        std::vector<int> curr_tokens(start, end);
        // TODO(lvhan): add salt to ensure the hash security
        size_t hash_key = hash(curr_tokens);

        int      block_id        = seq.blocks[idx];
        uint64_t block_unique_id = seq.block_unique_ids[idx];

        auto it = curr_node->children.find(hash_key);
        if (it != curr_node->children.end()) {
            if (curr_tokens != it->second->tokens) {
                TM_LOG_WARNING("[BlockTrie][cache] hash key cache hit, but tokens are not the same");
                break;
            }
            curr_node                  = it->second;
            curr_node->block_id        = block_id;
            curr_node->block_unique_id = block_unique_id;
        }
        else {
            // insert new node
            std::shared_ptr<TrieNode> node = std::make_shared<TrieNode>();
            node->hash_key                 = hash_key;
            node->tokens                   = curr_tokens;
            node->block_id                 = block_id;
            node->block_unique_id          = block_unique_id;
            curr_node->children[hash_key]  = node;
            curr_node                      = node;
        }
        cache_block_ids.emplace_back(block_id);
        cache_block_unique_ids.emplace_back(block_unique_id);
        cache_nodes.emplace_back(curr_node);
    }

    return std::make_tuple(cache_block_ids, cache_block_unique_ids, cache_nodes);
}

void BlockTrie::Remove(const std::vector<std::shared_ptr<TrieNode>>& nodes, int valid_size)
{
    if (nodes.empty() || valid_size < 1) {
        return;
    }
    // visit nodes in reverse order
    for (int idx = nodes.size() - 1; idx >= valid_size; --idx) {
        auto child  = nodes[idx];
        auto parent = nodes[idx - 1];
        auto it     = parent->children.find(child->hash_key);
        FT_CHECK(it != parent->children.end());
        FT_CHECK(it->second->tokens == child->tokens);
        parent->children.erase(it);
    }
}

void BlockTrie::Prune(ValidBlockChecker checker)
{
    return DFSPrune(root_, checker);
}

void BlockTrie::DFSPrune(std::shared_ptr<TrieNode>& node, ValidBlockChecker checker)
{
    for (auto it = node->children.begin(); it != node->children.end();) {
        if (!checker(it->second->block_id, it->second->block_unique_id)) {
            // child invalid
            it = node->children.erase(it);
        }
        else {
            DFSPrune(it->second, checker);
            it++;
        }
    }
}

}  // namespace turbomind
