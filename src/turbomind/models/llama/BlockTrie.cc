// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/BlockTrie.h"
#include "src/turbomind/models/llama/SequenceManager.h"

namespace turbomind {

namespace {

// Mix block length into the key so unrelated segments with identical token runs at different granularities
// do not collide (and to reduce collision risk vs. a naive size-seeded hash).
size_t HashSegmentTokens(const std::vector<int>& vec, size_t block_len)
{
    size_t seed = vec.size() ^ (block_len * 0x9e3779b97f4a7c15ULL);
    for (const auto& t : vec) {
        seed ^= std::hash<int>{}(t) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

}  // namespace

BlockTrie::BlockTrie(size_t                        block_len,
                     std::shared_ptr<BlockManager> block_manager,
                     SnapshotPublisher             snapshot_publisher,
                     SnapshotValidator             snapshot_validator,
                     SnapshotReleaser              snapshot_releaser):
    block_seq_len_(block_len),
    block_manager_(block_manager),
    snapshot_publisher_(std::move(snapshot_publisher)),
    snapshot_validator_(std::move(snapshot_validator)),
    snapshot_releaser_(std::move(snapshot_releaser))
{
    root_ = std::make_shared<TrieNode>();
}

BlockTrie::MatchResult BlockTrie::Match(const Sequence& seq)
{
    MatchResult result;

    auto node  = root_;
    auto first = seq.prompt.begin();

    // Warning: Do not use "<=" operator even when seq.prompt length is evenly
    // divisible by block_seq_len_. The model needs at least one input token to generate output.
    while (first + block_seq_len_ < seq.prompt.end()) {
        const std::vector<int> segment{first, first + block_seq_len_};
        const size_t           hash_key = HashSegmentTokens(segment, block_seq_len_);
        if (const auto it = node->children.find(hash_key); it != node->children.end()) {
            if (segment == it->second->tokens) {
                const auto& child = *it->second;
                if (snapshot_validator_) {
                    if (child.snapshot_slot < 0
                        || !snapshot_validator_(child.snapshot_slot, child.snapshot_unique_id)) {
                        break;
                    }
                    result.snapshot_slot      = child.snapshot_slot;
                    result.snapshot_unique_id = child.snapshot_unique_id;
                }
                result.block_ids.push_back(child.block_id);
                result.unique_ids.push_back(child.block_unique_id);
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

    result.matched_block_count = result.block_ids.size();
    return result;
}

BlockTrie::CacheResult BlockTrie::Cache(const Sequence& seq, const std::vector<int>& tokens)
{
    // Ensure the seq is active or locked so that all cache blocks must be valid
    TM_CHECK_NE(seq.status, Sequence::kCached);
    TM_CHECK_LE(seq.cache_len, seq.blocks.size() * block_seq_len_);

    auto node = root_;

    CacheResult result;

    const int n_blocks = std::min(seq.cache_len, (int)tokens.size()) / block_seq_len_;

    int new_cached = 0;

    for (int idx = 0; idx < n_blocks; ++idx) {
        auto start = tokens.begin() + idx * block_seq_len_;
        auto end   = start + block_seq_len_;

        const std::vector<int> segment(start, end);
        const size_t           hash_key = HashSegmentTokens(segment, block_seq_len_);

        int      block_id        = seq.blocks[idx];
        uint64_t block_unique_id = seq.block_unique_ids[idx];

        auto it = node->children.find(hash_key);
        if (it != node->children.end()) {
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
        if (snapshot_publisher_) {
            std::tie(node->snapshot_slot, node->snapshot_unique_id) =
                snapshot_publisher_(seq, idx, node->snapshot_slot);
        }
        result.block_ids.emplace_back(block_id);
        result.unique_ids.emplace_back(block_unique_id);
    }

    TM_LOG_INFO("[BlockTrie][cache] %d new tokens cached", new_cached);

    return result;
}

void BlockTrie::Verify()
{
    DFS(root_);
}

void BlockTrie::ReleaseSnapshots(const std::shared_ptr<TrieNode>& node)
{
    if (!node) {
        return;
    }
    if (snapshot_releaser_ && node->snapshot_slot >= 0 && snapshot_validator_
        && snapshot_validator_(node->snapshot_slot, node->snapshot_unique_id)) {
        snapshot_releaser_(node->snapshot_slot, node->snapshot_unique_id);
    }
    for (const auto& [_, child] : node->children) {
        ReleaseSnapshots(child);
    }
}

void BlockTrie::DFS(std::shared_ptr<TrieNode>& node)
{
    for (auto it = node->children.begin(); it != node->children.end();) {
        const bool block_valid = block_manager_->unique_id(it->second->block_id) == it->second->block_unique_id;
        const bool snapshot_valid =
            !snapshot_validator_
            || (it->second->snapshot_slot >= 0
                && snapshot_validator_(it->second->snapshot_slot, it->second->snapshot_unique_id));
        if (!block_valid || !snapshot_valid) {
            // Drop the full invalid subtree so descendant snapshot slots do not leak.
            ReleaseSnapshots(it->second);
            it = node->children.erase(it);
        }
        else {
            DFS(it->second);
            it++;
        }
    }
}

}  // namespace turbomind
