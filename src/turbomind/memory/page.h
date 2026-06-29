#pragma once

#include "src/turbomind/core/check.h"
#include "src/turbomind/memory/stats.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

namespace turbomind {

inline int ceil_log2(int x)
{
    if (x <= 1)
        return 0;
    // TODO: MSVC compatibility
    return 32 - __builtin_clz(x - 1);
}

inline int ceil_pow2(int x)
{
    if (x <= 1)
        return 1;
    return 1 << ceil_log2(x);
}

class PageAllocator {
public:
    PageAllocator(void* base, size_t size, size_t page_size):
        base_{align_base(base, page_size)},
        size_{align_size(base, base_, size)},
        page_size_{page_size},
        page_size_log2_{ceil_log2(page_size)},
        pages_{static_cast<int>(size_ / page_size)},  //
        max_order_{ceil_log2(pages_)},
        max_pages_{ceil_pow2(pages_)},
        nodes_(max_pages_ + max_order_ + 1)
    {
        TM_CHECK_NOTNULL(base_);
        for (int k = 0; k <= max_order_; ++k) {
            list_init(k);
        }
        build();
    }

    PageAllocator(const PageAllocator&)            = default;
    PageAllocator& operator=(const PageAllocator&) = default;

    void* allocate(size_t size)
    {
        const int order = ceil_log2(get_pages(size));

        for (int k = order; k <= max_order_; ++k) {
            const int idx = head(k);
            if (idx != sentinel(k)) {
                erase(idx);
                split(idx, k, order);
                return get_pointer(idx);
            }
        }

        return nullptr;
    }

    void deallocate(void* addr, int size)
    {
        add(get_page_idx(addr), ceil_log2(get_pages(size)));
    }

    PageStats stats() const
    {
        PageStats s;
        s.pages     = pages_;
        s.page_size = page_size_;
        s.max_order = max_order_;
        s.free_blocks_by_order.assign(max_order_ + 1, 0);
        int free_pages = 0;
        for (int k = 0; k <= max_order_; ++k) {
            int count = 0;
            for (int idx = head(k); idx != sentinel(k); idx = nodes_[idx].next) {
                ++count;
            }
            s.free_blocks_by_order[k] = count;
            free_pages += count << k;
        }
        s.free_pages = free_pages;
        s.used_pages = pages_ - free_pages;
        return s;
    }

private:
    // Round `base` up to the next multiple of `page_size`. Also validates that
    // `page_size` is a power of two so that the bit-mask below is well-defined.
    static char* align_base(void* base, size_t page_size)
    {
        TM_CHECK_EQ(page_size, ceil_pow2(page_size));
        const auto p    = reinterpret_cast<uintptr_t>(base);
        const auto mask = static_cast<uintptr_t>(page_size) - 1;
        return reinterpret_cast<char*>((p + mask) & ~mask);
    }

    // Subtract the padding consumed by aligning `base` from the usable region size.
    static size_t align_size(void* base, void* aligned, size_t size)
    {
        const size_t pad = static_cast<char*>(aligned) - static_cast<char*>(base);
        TM_CHECK_LE(pad, size);
        return size - pad;
    }

    void build()
    {
        for (int i = 0; i < pages_; ++i) {
            add(i, 0);
        }
    }

    int get_page_idx(void* addr)
    {
        auto offset = static_cast<char*>(addr) - base_;
        TM_CHECK(0 <= offset && offset < size_);
        return static_cast<int>(offset >> page_size_log2_);
    }

    void* get_pointer(int idx)
    {
        return base_ + (static_cast<size_t>(idx) << page_size_log2_);
    }

    int get_pages(size_t size)
    {
        return static_cast<int>((size + page_size_ - 1) >> page_size_log2_);
    }

    // Sentinel header node index for the order-k free list. Lives at
    // `max_pages_ + k` in nodes_, just past the real-and-ghost-page region.
    // Sentinels are reached via prev/next links only; their `order` field
    // is unread and stays at the default -1.
    int sentinel(int k) const noexcept
    {
        return max_pages_ + k;
    }

    int head(int k) const noexcept
    {
        return nodes_[sentinel(k)].next;
    }

    void list_init(int k) noexcept
    {
        const int s    = sentinel(k);
        nodes_[s].prev = s;
        nodes_[s].next = s;
    }

    // Branchless splice. Also writes nodes_[idx].order = k -- the link
    // helpers own the order field's "in which list" semantics, which add()
    // and split() rely on (see erase()).
    void push_front(int k, int idx) noexcept
    {
        const int s       = sentinel(k);
        const int next    = nodes_[s].next;
        nodes_[next].prev = idx;
        nodes_[idx].prev  = s;
        nodes_[idx].next  = next;
        nodes_[idx].order = k;
        nodes_[s].next    = idx;
    }

    // Branchless detach. Resets nodes_[idx].order = -1 so that add()'s
    // `nodes_[buddy].order == k` coalesce check and split()'s
    // `nodes_[sibling].order == -1` assertion remain correct. The idx
    // node's prev/next are left dangling -- nothing reaches it via link
    // chains once it is out of every list.
    void erase(int idx) noexcept
    {
        const int prev    = nodes_[idx].prev;
        const int next    = nodes_[idx].next;
        nodes_[prev].next = next;
        nodes_[next].prev = prev;
        nodes_[idx].order = -1;
    }

    void split(int idx, int k, int order)
    {
        if (k > order) {
            const int sibling = idx ^ (1 << (k - 1));
            TM_CHECK(nodes_[sibling].order == -1);
            push_front(k - 1, sibling);
            return split(idx, k - 1, order);
        }
    }

    int add(int idx, int k)
    {
        if (k < max_order_) {
            const int buddy = idx ^ (1 << k);
            if (nodes_[buddy].order == k) {
                erase(buddy);
                return add(idx & ~(1 << k), k + 1);
            }
        }
        push_front(k, idx);
        return k;
    }

private:
    char*  base_;
    size_t size_;
    size_t page_size_;

    int page_size_log2_;

    int pages_;
    int max_order_;
    int max_pages_;

    struct Node {
        int prev  = -1;
        int next  = -1;
        int order = -1;  // -1 = not in any free list
    };

    // Layout:
    //   [0, pages_)                            real pages (may be in a free list)
    //   [pages_, max_pages_)                   ghost pages (never added; order stays -1)
    //   [max_pages_, max_pages_ + max_order_]  sentinel header nodes (one per order)
    std::vector<Node> nodes_;
};

}  // namespace turbomind
