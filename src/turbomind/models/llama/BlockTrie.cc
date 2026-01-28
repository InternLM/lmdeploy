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
    BlockIds  block_ids;
    UniqueIds unique_ids;

    auto node  = root_;
    auto first = seq.prompt.begin();

    // Warning: Do not use "<=" operator even when seq.prompt length is evenly
    // divisible by block_seq_len_. The model needs at least one input token to generate output.
    while (first + block_seq_len_ < seq.prompt.end()) {
        const std::vector<int> segment{first, first + block_seq_len_};
        const size_t           hash_key = hash(segment);
        if (const auto it = node->children.find(hash_key); it != node->children.end()) {
            if (segment == it->second->tokens) {
                block_ids.push_back(it->second->block_id);
                unique_ids.push_back(it->second->block_unique_id);
                node = it->second;
                first += block_seq_len_;
            }
            else {
                TM_LOG_WARNING("hash collision detected");
                break;
            }
        }
        else {
            break;
        }
    }

    return std::make_tuple(block_ids, unique_ids);
}

std::tuple<BlockIds, UniqueIds> BlockTrie::Cache(const Sequence& seq, const std::vector<int>& tokens)
{
    // Ensure the seq is active or locked so that all cache blocks must be valid
    TM_CHECK_NE(seq.status, Sequence::kCached);
    TM_CHECK_LE(seq.cache_len, seq.blocks.size() * block_seq_len_);

    auto node = root_;

    BlockIds  cache_block_ids;
    UniqueIds cache_block_unique_ids;

    const int n_blocks = std::min(seq.cache_len, (int)tokens.size()) / block_seq_len_;

    int new_cached = 0;

    for (int idx = 0; idx < n_blocks; ++idx) {
        auto start = tokens.begin() + idx * block_seq_len_;
        auto end   = start + block_seq_len_;

        const std::vector<int> segment(start, end);
        const size_t           hash_key = hash(segment);  // TODO(lvhan): add salt to ensure the hash security

        int      block_id        = seq.blocks[idx];
        uint64_t block_unique_id = seq.block_unique_ids[idx];

        if (auto it = node->children.find(hash_key); it != node->children.end()) {
            if (segment == it->second->tokens) {  // fast-forward
                node                  = it->second;
                node->block_id        = block_id;
                node->block_unique_id = block_unique_id;
            }
            else {
                TM_LOG_WARNING("[BlockTrie][cache] Hash collision detected");
                break;
            }
        }
        else {
            // insert new node
            node                  = node->children.emplace_hint(it, hash_key, std::make_shared<TrieNode>())->second;
            node->hash_key        = hash_key;
            node->tokens          = segment;
            node->block_id        = block_id;
            node->block_unique_id = block_unique_id;
            new_cached += block_seq_len_;
        }
        cache_block_ids.emplace_back(block_id);
        cache_block_unique_ids.emplace_back(block_unique_id);
    }

    TM_LOG_INFO("[BlockTrie][cache] %d new tokens cached", new_cached);

    return std::make_tuple(cache_block_ids, cache_block_unique_ids);
}

void BlockTrie::Verify()
{
    DFS(root_);
}

void BlockTrie::DFS(std::shared_ptr<TrieNode>& node)
{
    for (auto it = node->children.begin(); it != node->children.end();) {
        if (block_manager_->unique_id(it->second->block_id) != it->second->block_unique_id) {
            // child invalid
            it = node->children.erase(it);
        }
        else {
            DFS(it->second);
            it++;
        }
    }
}

}  // namespace turbomind
