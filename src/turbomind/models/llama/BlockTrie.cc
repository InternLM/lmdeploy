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

BlockTrie::BlockTrie(size_t                        block_len,
                     std::shared_ptr<BlockManager> block_manager,
                     int                           linear_prefix_cache_interval_blocks,
                     int                           linear_state_slot_capacity,
                     std::vector<ssize_t>          conv_state_shape,
                     DataType                      conv_state_dtype,
                     std::vector<ssize_t>          recurrent_state_shape,
                     DataType                      recurrent_state_dtype):
    block_seq_len_(block_len),
    block_manager_(block_manager),
    linear_prefix_cache_interval_blocks_(linear_prefix_cache_interval_blocks),
    linear_prefix_cache_interval_tokens_(linear_prefix_cache_interval_blocks * block_len),
    linear_state_slot_capacity_(linear_state_slot_capacity),
    conv_state_shape_(std::move(conv_state_shape)),
    recurrent_state_shape_(std::move(recurrent_state_shape)),
    conv_state_dtype_(conv_state_dtype),
    recurrent_state_dtype_(recurrent_state_dtype)
{
    root_ = std::make_shared<TrieNode>();

    if (linear_state_slot_capacity_ > 0) {
        TM_CHECK_GT(linear_prefix_cache_interval_blocks_, 0);
        TM_CHECK(!conv_state_shape_.empty());
        TM_CHECK(!recurrent_state_shape_.empty());
        linear_conv_states_.resize(linear_state_slot_capacity_);
        linear_recurrent_states_.resize(linear_state_slot_capacity_);
        free_linear_state_slots_.reserve(linear_state_slot_capacity_);
        for (int slot = linear_state_slot_capacity_ - 1; slot >= 0; --slot) {
            free_linear_state_slots_.push_back(slot);
        }
    }
}

BlockTrieMatch BlockTrie::Match(const Sequence& seq)
{
    BlockTrieMatch match;

    auto node                = root_;
    auto first               = seq.prompt.begin();
    auto linear_prefix_state = root_;

    // Warning: Do not use "<=" operator even when seq.prompt length is evenly
    // divisible by block_seq_len_. The model needs at least one input token to generate output.
    while (first + block_seq_len_ < seq.prompt.end()) {
        const std::vector<int> segment{first, first + block_seq_len_};
        const size_t           hash_key = hash(segment);
        if (const auto it = node->children.find(hash_key); it != node->children.end()) {
            if (segment == it->second->tokens) {
                match.block_ids.push_back(it->second->block_id);
                match.unique_ids.push_back(it->second->block_unique_id);
                node = it->second;
                if (node->linear_state_slot >= 0) {
                    linear_prefix_state = node;
                }
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

    if (linear_prefix_state != root_) {
        const int slot         = linear_prefix_state->linear_state_slot;
        match.linear_cache_len = linear_prefix_state->num_matched;
        match.conv_states      = LinearConvState(slot);
        match.recurrent_states = LinearRecurrentState(slot);
    }

    return match;
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

    int       new_cached      = 0;
    const int checkpoint_base = linear_prefix_cache_interval_tokens_ ?
                                    seq.pending_linear_prefix_capture_base_len / linear_prefix_cache_interval_tokens_ :
                                    0;

    for (int idx = 0; idx < n_blocks; ++idx) {
        auto start = tokens.begin() + idx * block_seq_len_;
        auto end   = start + block_seq_len_;

        const std::vector<int> segment(start, end);
        const size_t           hash_key = hash(segment);  // TODO(lvhan): add salt to ensure the hash security

        int       block_id        = seq.blocks[idx];
        uint64_t  block_unique_id = seq.block_unique_ids[idx];
        const int num_matched     = (idx + 1) * block_seq_len_;

        if (auto it = node->children.find(hash_key); it != node->children.end()) {
            if (segment == it->second->tokens) {  // fast-forward
                node                  = it->second;
                node->block_id        = block_id;
                node->block_unique_id = block_unique_id;
                node->num_matched     = num_matched;
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
            node->num_matched     = num_matched;
            new_cached += block_seq_len_;
        }
        if (IsLinearCheckpointNode(num_matched)) {
            const int checkpoint_idx = num_matched / linear_prefix_cache_interval_tokens_ - checkpoint_base - 1;
            if (checkpoint_idx >= 0 && checkpoint_idx < seq.pending_linear_prefix_capture_count
                && seq.pending_linear_prefix_conv_states && seq.pending_linear_prefix_recurrent_states) {
                if (node->linear_state_slot < 0) {
                    node->linear_state_slot = AcquireLinearStateSlot();
                }
                if (node->linear_state_slot >= 0) {
                    Copy(seq.pending_linear_prefix_conv_states.slice(checkpoint_idx).squeeze(0),
                         LinearConvState(node->linear_state_slot));
                    Copy(seq.pending_linear_prefix_recurrent_states.slice(checkpoint_idx).squeeze(0),
                         LinearRecurrentState(node->linear_state_slot));
                }
            }
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
            ReleaseLinearPrefixState(it->second);
            it = node->children.erase(it);
        }
        else {
            DFS(it->second);
            it++;
        }
    }
}

void BlockTrie::ReleaseLinearPrefixState(std::shared_ptr<TrieNode>& node)
{
    if (!node) {
        return;
    }
    for (auto& [_, child] : node->children) {
        ReleaseLinearPrefixState(child);
    }
    ReleaseLinearStateSlot(node->linear_state_slot);
    node->linear_state_slot = -1;
}

bool BlockTrie::IsLinearCheckpointNode(int num_matched) const
{
    return linear_prefix_cache_interval_tokens_ > 0 && num_matched > 0
           && num_matched % linear_prefix_cache_interval_tokens_ == 0;
}

int BlockTrie::AcquireLinearStateSlot()
{
    if (free_linear_state_slots_.empty()) {
        if (!warned_linear_state_pool_exhausted_) {
            TM_LOG_WARNING("[BlockTrie] linear prefix checkpoint pool exhausted; deeper hybrid prefix checkpoints "
                           "will be skipped until cached entries are evicted");
            warned_linear_state_pool_exhausted_ = true;
        }
        return -1;
    }
    const int slot = free_linear_state_slots_.back();
    free_linear_state_slots_.pop_back();
    try {
        if (!linear_conv_states_[slot]) {
            linear_conv_states_[slot] = {conv_state_shape_, conv_state_dtype_, kDEVICE};
        }
        if (!linear_recurrent_states_[slot]) {
            linear_recurrent_states_[slot] = {recurrent_state_shape_, recurrent_state_dtype_, kDEVICE};
        }
    }
    catch (const std::exception& e) {
        free_linear_state_slots_.push_back(slot);
        if (!warned_linear_state_pool_oom_) {
            TM_LOG_WARNING("[BlockTrie] failed to allocate hybrid prefix checkpoint state: %s. "
                           "Further GDN prefix checkpoints will be skipped until memory is freed.",
                           e.what());
            warned_linear_state_pool_oom_ = true;
        }
        return -1;
    }
    return slot;
}

void BlockTrie::ReleaseLinearStateSlot(int slot)
{
    if (slot >= 0) {
        linear_conv_states_[slot]      = {};
        linear_recurrent_states_[slot] = {};
        free_linear_state_slots_.push_back(slot);
    }
}

Tensor BlockTrie::LinearConvState(int slot) const
{
    TM_CHECK_GE(slot, 0);
    TM_CHECK_LT(slot, (int)linear_conv_states_.size());
    return linear_conv_states_[slot];
}

Tensor BlockTrie::LinearRecurrentState(int slot) const
{
    TM_CHECK_GE(slot, 0);
    TM_CHECK_LT(slot, (int)linear_recurrent_states_.size());
    return linear_recurrent_states_[slot];
}

}  // namespace turbomind
