#pragma once

#include <cstddef>
#include <stdexcept>

namespace turbomind::lmcache {

struct TransferGeometry {
    int         heads{1};
    int         slices{1};
    std::size_t slice_bytes{};
    bool        strided{};
};

// The MP kernel launches one CTA per (K/V, block, layer), one warp per
// transport head. Native objects can contain many layers; flattening them to a
// single layer starves the device. Expose contiguous byte slices as strided
// MLA views to distribute the object across CTAs. This changes only the
// transport view, never the cache allocation or model head count.
inline TransferGeometry SelectTransferGeometry(std::size_t part_bytes, int max_threads, int blocks_per_chunk = 1)
{
    if (!part_bytes || part_bytes % 4 || max_threads < 32) {
        throw std::invalid_argument("unsupported LMCache transfer alignment/device limits");
    }
    if (part_bytes >= 256 * 1024 && part_bytes % 4096 == 0) {
        const bool use_large_tiles =
            blocks_per_chunk > 1 && part_bytes * blocks_per_chunk >= (24UL << 20) && part_bytes % 8192 == 0;
        const std::size_t tile = use_large_tiles ? 8192 : 4096;
        return {1, static_cast<int>(part_bytes / tile), tile, true};
    }
    int heads = 1;
    while (heads < 32 && 64 * heads <= max_threads && part_bytes % (64 * heads) == 0
           && part_bytes / (4 * heads) >= 2048) {
        heads *= 2;
    }
    return {heads, 1, part_bytes, false};
}

}  // namespace turbomind::lmcache
