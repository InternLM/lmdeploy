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

BlockTrie::BlockTrie(size_t block_len, std::shared_ptr<BlockManager> block_manager):
    block_seq_len_(block_len), block_manager_(block_manager)
{
    root_ = std::make_shared<TrieNode>();
}

std::tuple<BlockIds, UniqueIds> BlockTrie::Match(const Sequence& seq)
{
    BlockIds  matched_blocks;
    UniqueIds matched_unique_ids;

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
        curr_node = it->second;
        num_matched += block_seq_len_;
    }
    return std::make_tuple(matched_blocks, matched_unique_ids);
}

std::tuple<BlockIds, UniqueIds> BlockTrie::Cache(const Sequence& seq, const std::vector<int>& tokens)
{
    FT_CHECK(seq.status != Sequence::kCached);
    FT_CHECK(tokens.size() <= seq.blocks.size() * block_seq_len_);

    std::shared_ptr<TrieNode> curr_node = root_;
    int                       idx       = 0;

    BlockIds  cache_block_ids;
    UniqueIds cache_block_unique_ids;

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
    }

    return std::make_tuple(cache_block_ids, cache_block_unique_ids);
}

int BlockTrie::Verify()
{
    return DFS(root_);
}

int BlockTrie::DFS(std::shared_ptr<TrieNode>& node)
{
    int invalid_count = 0;
    for (auto it = node->children.begin(); it != node->children.end();) {
        if (block_manager_->unique_id(it->second->block_id) != it->second->block_unique_id) {
            // child invalid
            it = node->children.erase(it);
            ++invalid_count;
        }
        else {
            invalid_count += DFS(it->second);
            it++;
        }
    }
    return invalid_count;
}

}  // namespace turbomind
