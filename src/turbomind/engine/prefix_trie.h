#pragma once

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "src/turbomind/core/check.h"
#include "src/turbomind/engine/block.h"
#include "src/turbomind/engine/prefix_key.h"

namespace turbomind {

// Prefix trie index over logical block nodes. Holds raw (weak) LogicalBlock*
// kept consistent by Erase on recycle, so a lookup never returns a dead node.
class PrefixTrie {
public:
    explicit PrefixTrie(int block_size): block_size_{block_size} {}

    // Exact full lookup: hash, parent, length, and token identity must match.
    LogicalBlock* Find(const LogicalBlock* parent, const PrefixKey& key, TokenSpan tokens) const
    {
        if (auto it = index_.find(key); it != index_.end()) {
            LogicalBlock* b = it->second;
            if (b->parent == parent && b->size == tokens.size
                && std::equal(tokens.begin(), tokens.end(), b->tokens.begin())) {
                return b;
            }
        }
        return nullptr;
    }

    // Longest partial match within one block (never the full block). On a hit,
    // `key` is replaced with the matched node's key.
    LogicalBlock* Search(const LogicalBlock* parent, PrefixKey& key, TokenSpan tokens) const
    {
        std::vector<PrefixKey> prefixes;
        PrefixKey              k = key;
        for (const int* it = tokens.begin(); it != tokens.end(); ++it) {
            k.hash = HashCombine(k.hash, static_cast<size_t>(*it));
            ++k.length;
            prefixes.push_back(k);
        }
        if (static_cast<int>(prefixes.size()) == block_size_) {
            prefixes.pop_back();  // enforce a partial match
        }
        for (int i = static_cast<int>(prefixes.size()); i > 0; --i) {
            if (LogicalBlock* b = Find(parent, prefixes[i - 1], TokenSpan{tokens.begin(), i})) {
                key = prefixes[i - 1];
                return b;
            }
        }
        return nullptr;
    }

    // First-wins insertion; reads node.key/parent/tokens already set.
    bool Insert(LogicalBlock& b)
    {
        TM_CHECK(!b.indexed);
        TM_CHECK(static_cast<bool>(b.key));
        if (auto it = index_.find(b.key); it == index_.end()) {
            index_.emplace_hint(it, b.key, &b);
            b.indexed = true;
            return true;
        }
        return false;
    }

    // Fired by the pool recycle hook.
    void Erase(LogicalBlock& b)
    {
        if (b.indexed) {
            auto it = index_.find(b.key);
            TM_CHECK(it != index_.end());
            TM_CHECK_EQ(it->second, &b);
            index_.erase(it);
        }
    }

private:
    int                                                         block_size_;
    std::unordered_map<PrefixKey, LogicalBlock*, PrefixKeyHash> index_;
};

}  // namespace turbomind
