// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/BlockTrie.h"
#include "src/turbomind/models/llama/SequenceManager.h"

namespace turbomind {

size_t hash(const std::vector<int>& vec) {
   size_t seed = vec.size();
   for (const auto& i : vec) {
       seed ^= std::hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
   }
   return seed;
}

BlockTrie::BlockTrie(size_t block_seq_len, std::shared_ptr<BlockManager> block_manager, bool enable_prefix_caching):
    block_seq_len_(block_seq_len), block_manager_(block_manager), enable_prefix_caching_(enable_prefix_caching)
{
    root_ = std::make_shared<TrieNode>();
}

void BlockTrie::match(Sequence& seq)
{
    BlockIds matched_blocks;
    UniqueIds matched_unique_ids;

    std::shared_ptr<TrieNode> curr_node = root_;
    int num_matched = 0;

    while (num_matched + block_seq_len_ < seq.prompt.size()) {
        std::vector<int> curr_tokens(seq.prompt.begin() + num_matched, seq.prompt.begin() + num_matched + block_seq_len_);
        size_t hash_key = hash(curr_tokens);

        auto it = curr_node->children.find(hash_key);

        if (it == curr_node->children.end()) {
            break;
        }

        if (curr_tokens != it->second->tokens) {
            break;
        }

        matched_blocks.push_back(it->second->block_id);
        matched_unique_ids.push_back(it->second->block_unique_id);
        curr_node = it->second;
        num_matched += block_seq_len_;
    }

    if (matched_blocks.size() > 0) {
        // add use count
        block_manager_->Lock(matched_blocks);
        block_manager_->Touch(matched_blocks);
        // only consider no history blocks
        seq.blocks.insert(seq.blocks.end(), matched_blocks.begin(), matched_blocks.end());
        seq.block_unique_ids.insert(seq.block_unique_ids.end(), matched_unique_ids.begin(), matched_unique_ids.end());
    }
}

void BlockTrie::cache(const Sequence& seq)
{
    std::shared_ptr<TrieNode> curr_node = root_;
    int num_matched = 0;
    int idx = 0;
    BlockIds cached_blocks;

    while (num_matched + block_seq_len_ <= seq.prompt.size()) {
        std::vector<int> curr_tokens(seq.prompt.begin() + num_matched, seq.prompt.begin() + num_matched + block_seq_len_);
        size_t hash_key = hash(curr_tokens);

        auto it = curr_node->children.find(hash_key);

        int block_id = seq.blocks[idx];
        uint64_t block_unique_id = seq.block_unique_ids[idx];

        if (it != curr_node->children.end()) {
            if (curr_tokens != it->second->tokens) {
                break;
            }
            curr_node = it->second;
            curr_node->block_id = block_id;
            curr_node->block_unique_id = block_unique_id;
        } else {
            // insert new node
            std::shared_ptr<TrieNode> node = std::make_shared<TrieNode>();
            node->hash_key = hash_key;
            node->tokens = curr_tokens;
            node->block_id = block_id;
            node->block_unique_id = block_unique_id;
            node->num_matched = num_matched + block_seq_len_;
            curr_node->children[hash_key] = node;
            curr_node = node;
        }

        cached_blocks.push_back(curr_node->block_id);
        num_matched += block_seq_len_;
        idx++;
    }

    block_manager_->Touch(cached_blocks);
}

int BlockTrie::verify()
{
    return verify_traverse(root_);
}

int BlockTrie::verify_traverse(std::shared_ptr<TrieNode>& node)
{
    int valid_count = 1;
    for (auto it = node->children.begin(); it != node->children.end();) {
        if (block_manager_->unique_id(it->second->block_id) != it->second->block_unique_id) {
            // child invalid
            it = node->children.erase(it);
        } else {
            valid_count += verify_traverse(it->second);
            it++;
        }
    }
    return valid_count;
}

} // namespace turbomind
