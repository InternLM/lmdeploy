#pragma once
#include <array>
#include <cstdint>

namespace turbomind {

// 256-bit (SHA-256) opaque multimodal content identity, stored as four 64-bit
// words. All-zero is the reserved "empty" sentinel; an empty fingerprint never
// compares equal to anything -- including another empty fingerprint.
struct Fingerprint {
    std::array<uint64_t, 4> words{};

    bool empty() const noexcept
    {
        return words == std::array<uint64_t, 4>{};
    }

    friend bool operator==(const Fingerprint& a, const Fingerprint& b) noexcept
    {
        if (a.empty() || b.empty()) {
            return false;
        }
        return a.words == b.words;
    }
    friend bool operator!=(const Fingerprint& a, const Fingerprint& b) noexcept
    {
        return !(a == b);
    }
};

}  // namespace turbomind
