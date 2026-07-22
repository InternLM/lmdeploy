#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace turbomind {

struct PageStats {
    int              pages;  // real pages in the region
    size_t           page_size;
    int              free_pages;  // sum over orders of (count << order)
    int              used_pages;  // pages - free_pages
    int              max_order;
    std::vector<int> free_blocks_by_order;  // index = buddy order, value = # free blocks
};

struct SlabStats {
    size_t object_size;   // bytes per object (the size class)
    size_t slab_size;     // bytes per slab
    size_t object_count;  // objects per slab
    int    n_full;
    int    n_partial;
    int    n_empty;
    int    n_slabs;        // full + partial + empty
    size_t total_objects;  // n_slabs * object_count
    size_t used_objects;
    size_t free_objects;
};

struct MemoryStats {
    PageStats              page;
    std::vector<SlabStats> slabs;
    size_t                 live_allocations;  // pool_.size() - free_.size()
    size_t                 live_bytes;        // sum of aligned bytes held by live object parts
    size_t                 region_bytes;
};

// Renders MemoryStats as a verbose multi-line string (defined in stats.cc).
std::string FormatMemoryStats(const MemoryStats& s);

}  // namespace turbomind
