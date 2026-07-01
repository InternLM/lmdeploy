#pragma once

#include <cstddef>
#include <vector>

#include "src/turbomind/core/check.h"
#include "src/turbomind/engine/fingerprint.h"

namespace turbomind {

struct TokenSpan {
    const int* data{};
    int        size{};

    const int* begin() const noexcept
    {
        return data;
    }

    const int* end() const noexcept
    {
        return size == 0 ? data : data + size;
    }
};

inline TokenSpan MakeTokenSpan(const std::vector<int>& tokens) noexcept
{
    return TokenSpan{tokens.data(), static_cast<int>(tokens.size())};
}

inline TokenSpan MakeTokenSpan(const int* data, int size) noexcept
{
    return TokenSpan{data, size};
}

inline size_t HashCombine(size_t seed, size_t value) noexcept
{
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

inline size_t HashCombine(size_t seed, const Fingerprint& fp) noexcept
{
    for (uint64_t w : fp.words) {
        seed = HashCombine(seed, static_cast<size_t>(w));
    }
    return seed;
}

struct PrefixKey {
    int    length{};
    size_t hash{};

    explicit operator bool() const noexcept
    {
        return hash || length;
    }

    friend bool operator==(const PrefixKey& a, const PrefixKey& b) noexcept
    {
        return a.length == b.length && a.hash == b.hash;
    }
};

// PrefixKey::hash is already a cumulative hash; using it directly lets the
// trie key be extended incrementally token by token.
struct PrefixKeyHash {
    size_t operator()(const PrefixKey& key) const noexcept
    {
        return key.hash;
    }
};

inline PrefixKey ExtendPrefixKey(PrefixKey key, TokenSpan tokens)
{
    TM_CHECK_GE(tokens.size, 0);
    for (const int* it = tokens.begin(); it != tokens.end(); ++it) {
        key.hash = HashCombine(key.hash, static_cast<size_t>(*it));
        ++key.length;
    }
    return key;
}

inline PrefixKey ExtendPrefixKey(PrefixKey key, TokenSpan tokens, const std::vector<Fingerprint>& fps)
{
    key = ExtendPrefixKey(key, tokens);  // existing token fold
    for (const Fingerprint& fp : fps) {
        key.hash = HashCombine(key.hash, fp);
    }
    return key;
}

}  // namespace turbomind
