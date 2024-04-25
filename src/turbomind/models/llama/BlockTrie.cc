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

BlockTrie::~BlockTrie()
{
    leaves_.clear();
}

void BlockTrie::match(Sequence& seq)
{
    BlockIds matched_blocks;
    UniqueIds matched_unique_ids;

    std::shared_ptr<TrieNode> curr_node;
    if (seq.last_matched_node == nullptr) {
        curr_node = root_;
    } else {
        curr_node = seq.last_matched_node;
    }

    int num_matched = curr_node->num_matched;

    while (num_matched + block_seq_len_ < seq.tokens.size()) {
        std::vector<int> curr_tokens(seq.tokens.begin() + num_matched, seq.tokens.begin() + num_matched + block_seq_len_);
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
        // add ref count
        block_manager_->Lock(matched_blocks);
        block_manager_->Touch(matched_blocks);
        // only consider no history blocks
        seq.blocks.insert(seq.blocks.end(), matched_blocks.begin(), matched_blocks.end());
        seq.block_unique_ids.insert(seq.block_unique_ids.end(), matched_unique_ids.begin(), matched_unique_ids.end());
    }

    seq.last_matched_node = curr_node;
}

void BlockTrie::cache(const Sequence& seq)
{
    std::shared_ptr<TrieNode> curr_node;
    if (seq.last_matched_node == nullptr) {
        curr_node = root_;
    } else {
        curr_node = seq.last_matched_node;
    }

    int num_matched = curr_node->num_matched;

    if (num_matched + block_seq_len_ > seq.cache_len) {
        return;
    }

    int idx = num_matched / block_seq_len_;
    BlockIds cached_blocks;

    while (num_matched + block_seq_len_ <= seq.cache_len) {
        std::vector<int> curr_tokens(seq.tokens.begin() + num_matched, seq.tokens.begin() + num_matched + block_seq_len_);
        size_t hash_key = hash(curr_tokens);

        auto it = curr_node->children.find(hash_key);

        if (it != curr_node->children.end()) {
            if (curr_tokens != it->second->tokens) {
                break;
            }
            curr_node = it->second;
        } else {
            int block_id = seq.blocks[idx];
            uint64_t block_unique_id = seq.block_unique_ids[idx];
            // insert new node
            std::shared_ptr<TrieNode> node = std::make_shared<TrieNode>();
            node->hash_key = hash_key;
            node->tokens = curr_tokens;
            node->block_id = block_id;
            node->block_unique_id = block_unique_id;
            node->parent = curr_node;
            node->num_matched = num_matched + block_seq_len_;
            if (curr_node->children.empty()) {
                leaves_.erase(curr_node);
            }
            curr_node->children[hash_key] = node;
            curr_node = node;
            cached_blocks.push_back(block_id);
            cached_block_num_++;
        }

        num_matched += block_seq_len_;
        idx++;
    }

    seq.last_matched_node = curr_node;
    block_manager_->Lock(cached_blocks);
    block_manager_->Touch(cached_blocks);
    if (curr_node->parent != nullptr && curr_node->children.empty()) {
        leaves_.insert(curr_node);
    }
}

int BlockTrie::evict(const int num)
{
    if (num == 0) {
        return 0;
    }
    int num_evicted = 0;
    BlockIds evicted_blocks;

    auto compare = [&](const std::shared_ptr<TrieNode>& a, const std::shared_ptr<TrieNode>& b) -> bool {
        const Block& block_a = block_manager_->block(a->block_id);
        const Block& block_b = block_manager_->block(b->block_id);
        if (block_a.use_count != block_b.use_count) {
            return block_a.use_count > block_b.use_count;
        }
        return block_a.timestamp > block_b.timestamp;
    };

    std::priority_queue<std::shared_ptr<TrieNode>, std::vector<std::shared_ptr<TrieNode>>, decltype(compare)> leaves_queue(compare);
    for (auto& leaf : leaves_) {
        leaves_queue.push(leaf);
    }

    while (num_evicted < num) {
        if (leaves_queue.empty()) {
            break;
        }

        auto top = leaves_queue.top();
        const Block& block = block_manager_->block(top->block_id);
        if (block.use_count > 1) {
            break;
        }

        // use_count == 1 means only block trie use it, can be evicted
        evicted_blocks.push_back(top->block_id);
        top->parent->children.erase(top->hash_key);
        leaves_.erase(top);
        leaves_queue.pop();

        if (top->parent->parent != nullptr && top->parent->children.empty()) {
            leaves_.insert(top->parent);
            leaves_queue.push(top->parent);
        }

        num_evicted++;
        cached_block_num_--;
    }

    block_manager_->Unlock(evicted_blocks);

    return num_evicted;
}

} // namespace turbomind
