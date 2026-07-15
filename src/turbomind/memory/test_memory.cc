// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/memory/object.h"
#include "src/turbomind/memory/page.h"
#include "src/turbomind/memory/slab.h"

#include "src/turbomind/core/core.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

using namespace turbomind;

namespace {

// RAII for a heap region whose base is aligned to `alignment` bytes.
struct AlignedRegion {
    AlignedRegion(size_t alignment, size_t bytes): bytes_{bytes}
    {
        // std::aligned_alloc requires bytes to be a multiple of alignment.
        size_t rounded = (bytes + alignment - 1) / alignment * alignment;
        ptr_           = std::aligned_alloc(alignment, rounded);
        TM_CHECK_NOTNULL(ptr_);
    }
    ~AlignedRegion()
    {
        std::free(ptr_);
    }
    AlignedRegion(const AlignedRegion&) = delete;
    AlignedRegion& operator=(const AlignedRegion&) = delete;

    void* data() const
    {
        return ptr_;
    }
    size_t size() const
    {
        return bytes_;
    }

private:
    void*  ptr_;
    size_t bytes_;
};

inline bool ranges_overlap(const void* a, size_t na, const void* b, size_t nb)
{
    auto pa = reinterpret_cast<uintptr_t>(a);
    auto pb = reinterpret_cast<uintptr_t>(b);
    return pa < pb + nb && pb < pa + na;
}

// Bytes the buddy allocator actually reserves for a request of `size` bytes given `page_size`.
inline size_t reserved_bytes(size_t size, size_t page_size)
{
    int pages = static_cast<int>((size + page_size - 1) / page_size);
    return static_cast<size_t>(ceil_pow2(pages)) * page_size;
}

constexpr size_t kObjectAllocatorPageBytes = 32UL << 20;  // mirrors MemoryState::kPageSize/kMinSlabSize

}  // namespace

TEST_CASE("memory test binary builds", "[memory][smoke]")
{
    REQUIRE(true);
}

TEST_CASE("ceil_log2 / ceil_pow2", "[memory][utils]")
{
    REQUIRE(ceil_log2(0) == 0);
    REQUIRE(ceil_log2(1) == 0);
    REQUIRE(ceil_log2(2) == 1);
    REQUIRE(ceil_log2(3) == 2);
    REQUIRE(ceil_log2(4) == 2);
    REQUIRE(ceil_log2(5) == 3);
    REQUIRE(ceil_log2(7) == 3);
    REQUIRE(ceil_log2(8) == 3);
    REQUIRE(ceil_log2(16) == 4);

    REQUIRE(ceil_pow2(0) == 1);
    REQUIRE(ceil_pow2(1) == 1);
    REQUIRE(ceil_pow2(2) == 2);
    REQUIRE(ceil_pow2(3) == 4);
    REQUIRE(ceil_pow2(4) == 4);
    REQUIRE(ceil_pow2(5) == 8);
    REQUIRE(ceil_pow2(16) == 16);
    REQUIRE(ceil_pow2(17) == 32);

    for (int x = 1; x <= 1024; ++x) {
        int p = ceil_pow2(x);
        REQUIRE(p >= x);
        REQUIRE((p & (p - 1)) == 0);  // power of two
    }
}

TEST_CASE("PageAllocator basic", "[memory][page]")
{
    constexpr size_t kPage = 4096;
    constexpr size_t kSize = 16 * kPage;

    AlignedRegion region{kPage, kSize};
    PageAllocator alloc{region.data(), kSize, kPage};

    void* p = alloc.allocate(kPage);
    REQUIRE(p != nullptr);
    REQUIRE(p >= region.data());
    REQUIRE((char*)p + kPage <= (char*)region.data() + kSize);

    alloc.deallocate(p, kPage);

    void* q = alloc.allocate(kPage);
    REQUIRE(q != nullptr);
    REQUIRE(reinterpret_cast<uintptr_t>(q) % kPage == 0);

    alloc.deallocate(q, kPage);
}

TEST_CASE("PageAllocator alignment", "[memory][page]")
{
    constexpr size_t kPage = 4096;
    constexpr size_t kSize = 16 * kPage;

    AlignedRegion region{kPage, kSize};
    PageAllocator alloc{region.data(), kSize, kPage};

    // Even sub-page requests come back page-aligned.
    for (size_t req : {size_t{1}, size_t{17}, size_t{kPage / 2}, size_t{kPage}, size_t{kPage + 1}}) {
        void* p = alloc.allocate(req);
        REQUIRE(p != nullptr);
        REQUIRE(reinterpret_cast<uintptr_t>(p) % kPage == 0);
        alloc.deallocate(p, req);
    }
}

TEST_CASE("PageAllocator misaligned base", "[memory][page]")
{
    constexpr size_t kPage = 4096;
    // Allocate enough to absorb a 3-byte preamble and still expose 16 full pages.
    AlignedRegion region{kPage, 17 * kPage};

    void*         shifted_base = static_cast<char*>(region.data()) + 3;
    size_t        shifted_size = 16 * kPage + (kPage - 3);  // base+3 .. region_end
    PageAllocator alloc{shifted_base, shifted_size, kPage};

    std::vector<void*> ptrs;
    while (void* p = alloc.allocate(kPage)) {
        REQUIRE(reinterpret_cast<uintptr_t>(p) % kPage == 0);
        ptrs.push_back(p);
    }
    REQUIRE(ptrs.size() == 16);  // (shifted_size - pad) / kPage == 16
    for (void* p : ptrs) {
        alloc.deallocate(p, kPage);
    }
}

TEST_CASE("PageAllocator coalescing", "[memory][page]")
{
    constexpr size_t kPage  = 4096;
    constexpr size_t kPages = 16;
    constexpr size_t kSize  = kPages * kPage;

    AlignedRegion region{kPage, kSize};

    auto run = [&](bool reverse_free) {
        PageAllocator      alloc{region.data(), kSize, kPage};
        std::vector<void*> ptrs;
        for (size_t i = 0; i < kPages; ++i) {
            void* p = alloc.allocate(kPage);
            REQUIRE(p != nullptr);
            ptrs.push_back(p);
        }
        if (reverse_free) {
            std::reverse(ptrs.begin(), ptrs.end());
        }
        for (void* p : ptrs) {
            alloc.deallocate(p, kPage);
        }
        // After all pages are freed, the whole region must be allocatable in one go.
        void* big = alloc.allocate(kSize);
        REQUIRE(big != nullptr);
        REQUIRE(big == region.data());  // (only valid because base is page-aligned)
        alloc.deallocate(big, kSize);
    };

    run(/*reverse_free=*/false);
    run(/*reverse_free=*/true);
}

TEST_CASE("PageAllocator exhaustion", "[memory][page]")
{
    constexpr size_t kPage  = 4096;
    constexpr size_t kPages = 16;
    constexpr size_t kSize  = kPages * kPage;

    AlignedRegion region{kPage, kSize};
    PageAllocator alloc{region.data(), kSize, kPage};

    std::vector<void*> ptrs;
    while (void* p = alloc.allocate(kPage)) {
        ptrs.push_back(p);
    }
    REQUIRE(ptrs.size() == kPages);

    for (void* p : ptrs) {
        alloc.deallocate(p, kPage);
    }
    void* big = alloc.allocate(kSize);
    REQUIRE(big != nullptr);
    alloc.deallocate(big, kSize);
}

TEST_CASE("PageAllocator non-power-of-2 region", "[memory][page]")
{
    constexpr size_t kPage  = 4096;
    constexpr size_t kPages = 12;  // not a power of two
    constexpr size_t kSize  = kPages * kPage;

    AlignedRegion region{kPage, kSize};
    PageAllocator alloc{region.data(), kSize, kPage};

    // 12 sequential single-page allocations must all succeed.
    std::vector<void*> ptrs;
    for (size_t i = 0; i < kPages; ++i) {
        void* p = alloc.allocate(kPage);
        REQUIRE(p != nullptr);
        REQUIRE(reinterpret_cast<uintptr_t>(p) % kPage == 0);
        // Pointer must lie within the valid region.
        REQUIRE(p >= region.data());
        REQUIRE(static_cast<char*>(p) < static_cast<char*>(region.data()) + kSize);
        ptrs.push_back(p);
    }
    REQUIRE(alloc.allocate(kPage) == nullptr);
    for (void* p : ptrs) {
        alloc.deallocate(p, kPage);
    }
}

TEST_CASE("PageAllocator stress (random)", "[memory][page][stress]")
{
    constexpr size_t kPage  = 4096;
    constexpr size_t kPages = 64;
    constexpr size_t kSize  = kPages * kPage;
    constexpr int    kOps   = 5000;

    AlignedRegion region{kPage, kSize};
    PageAllocator alloc{region.data(), kSize, kPage};

    std::mt19937 rng{0xdeadbeefU};

    // live[addr] = requested_size
    std::map<void*, size_t> live;
    size_t                  live_reserved = 0;

    auto check_no_overlap = [&](void* p, size_t req) {
        size_t res = reserved_bytes(req, kPage);
        for (auto& [addr, sz] : live) {
            REQUIRE_FALSE(ranges_overlap(p, res, addr, reserved_bytes(sz, kPage)));
        }
    };

    for (int op = 0; op < kOps; ++op) {
        bool do_alloc = live.empty() ? true : (live.size() >= kPages ? false : (rng() & 1));
        if (do_alloc) {
            // Random size 1..kSize bytes (allocator rounds up internally).
            size_t req = std::uniform_int_distribution<size_t>{1, kSize}(rng);
            void*  p   = alloc.allocate(req);
            if (!p) {
                continue;  // capacity-bounded failure is OK
            }
            REQUIRE(p >= region.data());
            REQUIRE((char*)p + reserved_bytes(req, kPage) <= (char*)region.data() + kSize);
            REQUIRE(reinterpret_cast<uintptr_t>(p) % kPage == 0);
            check_no_overlap(p, req);
            live[p] = req;
            live_reserved += reserved_bytes(req, kPage);
            REQUIRE(live_reserved <= kSize);
        }
        else {
            // Pick a random live entry and free it.
            auto it = live.begin();
            std::advance(it, std::uniform_int_distribution<size_t>{0, live.size() - 1}(rng));
            alloc.deallocate(it->first, it->second);
            live_reserved -= reserved_bytes(it->second, kPage);
            live.erase(it);
        }
    }

    // Free everything; whole region should be allocatable again.
    for (auto& [addr, sz] : live) {
        alloc.deallocate(addr, sz);
    }
    void* big = alloc.allocate(kSize);
    REQUIRE(big != nullptr);
    alloc.deallocate(big, kSize);
}

TEST_CASE("SlabAllocator min_size < object_size", "[memory][slab]")
{
    constexpr size_t kPage = 4096;
    constexpr size_t kSize = 16 * kPage;

    AlignedRegion region{kPage, kSize};
    PageAllocator pages{region.data(), kSize, kPage};

    // min_size (64) is smaller than object_size (1024). Pre-fix this loops forever.
    SlabAllocator slab{/*object_size=*/1024,
                       /*min_size=*/64,
                       /*max_size=*/64 * 1024,
                       /*threshold=*/0.0f,
                       /*max_empty_slabs=*/2};

    SlabSlot out[1] = {};
    int      n      = slab.allocate(out, 1, pages);
    REQUIRE(n == 1);
    char* addr = slab.AddressOf(out[0]);
    REQUIRE(addr != nullptr);
    REQUIRE(reinterpret_cast<uintptr_t>(addr) >= reinterpret_cast<uintptr_t>(region.data()));
    REQUIRE(reinterpret_cast<uintptr_t>(addr) + 1024 <= reinterpret_cast<uintptr_t>(region.data()) + kSize);
    slab.deallocate(out, 1, pages);
}

TEST_CASE("SlabAllocator slot_owner reverse link", "[memory][slab][owner]")
{
    constexpr size_t kPage = 4096;
    constexpr size_t kSize = 16 * kPage;
    constexpr size_t kObj  = 64;

    AlignedRegion region{kPage, kSize};
    PageAllocator pages{region.data(), kSize, kPage};

    // max_empty_slabs = 2 keeps the single slab resident after the free below,
    // so owner() reads a live (non-reclaimed) slot.
    SlabAllocator slab{kObj, kPage, kPage, 0.0f, /*max_empty_slabs=*/2};

    SlabSlot s[1] = {};
    REQUIRE(slab.allocate(s, 1, pages) == 1);

    // The slab carries a bidirectional link slot -> owning token. A fresh slot
    // starts unowned (default token); set_owner records the owner; deallocate
    // clears the link.
    REQUIRE(slab.owner(s[0]) == object_alloc_t{});

    const object_alloc_t tok{reinterpret_cast<const Allocation*>(0x1000)};
    slab.set_owner(s[0], tok);
    REQUIRE(slab.owner(s[0]) == tok);

    slab.deallocate(s, 1, pages);
    REQUIRE(slab.owner(s[0]) == object_alloc_t{});  // cleared on free
}

TEST_CASE("SlabAllocator empty-slab reclamation", "[memory][slab]")
{
    constexpr size_t kPage = 4096;
    constexpr size_t kSize = 16 * kPage;  // 64 KiB

    AlignedRegion region{kPage, kSize};
    PageAllocator pages{region.data(), kSize, kPage};

    constexpr size_t kObj = 64;
    SlabAllocator    slab{/*object_size=*/kObj,
                       /*min_size=*/kPage,
                       /*max_size=*/kPage,
                       /*threshold=*/0.0f,
                       /*max_empty_slabs=*/0};

    constexpr int         kCount = 64 * 3 + 5;
    std::vector<SlabSlot> outs(kCount);
    int                   got = slab.allocate(outs.data(), kCount, pages);
    REQUIRE(got == kCount);

    // All addresses are distinct (Bug E sanity check on multi-slab path).
    std::vector<char*> addrs;
    for (auto& o : outs) {
        addrs.push_back(slab.AddressOf(o));
    }
    std::sort(addrs.begin(), addrs.end());
    REQUIRE(std::unique(addrs.begin(), addrs.end()) == addrs.end());

    // Free everything; with max_empty_slabs=0, the slab allocator must
    // return its pages to the underlying PageAllocator.
    slab.deallocate(outs.data(), kCount, pages);

    void* big = pages.allocate(kSize);
    REQUIRE(big != nullptr);
    pages.deallocate(big, kSize);
}

TEST_CASE("SlabAllocator single slab", "[memory][slab]")
{
    constexpr size_t kPage = 4096;
    constexpr size_t kSize = 16 * kPage;
    constexpr size_t kObj  = 64;

    AlignedRegion region{kPage, kSize};
    PageAllocator pages{region.data(), kSize, kPage};

    SlabAllocator slab{kObj, kPage, kPage, 0.0f, 2};

    constexpr int         kPerSlab = kPage / kObj;  // 64
    std::vector<SlabSlot> outs(kPerSlab);
    REQUIRE(slab.allocate(outs.data(), kPerSlab, pages) == kPerSlab);

    // Distinctness + per-slab object alignment.
    std::vector<char*> addrs;
    for (auto& o : outs) {
        char* addr = slab.AddressOf(o);
        REQUIRE(addr != nullptr);
        addrs.push_back(addr);
    }
    std::sort(addrs.begin(), addrs.end());
    REQUIRE(std::unique(addrs.begin(), addrs.end()) == addrs.end());
    for (size_t i = 1; i < addrs.size(); ++i) {
        REQUIRE(static_cast<size_t>(addrs[i] - addrs[i - 1]) == kObj);
    }

    // One more allocation should still succeed because the region has
    // pages free for another slab.
    SlabSlot extra[1] = {};
    REQUIRE(slab.allocate(extra, 1, pages) == 1);

    slab.deallocate(extra, 1, pages);
    slab.deallocate(outs.data(), kPerSlab, pages);

    // Round-trip: refill the same slab.
    REQUIRE(slab.allocate(outs.data(), kPerSlab, pages) == kPerSlab);
    slab.deallocate(outs.data(), kPerSlab, pages);
}

TEST_CASE("SlabAllocator spans multiple slabs", "[memory][slab]")
{
    constexpr size_t kPage    = 4096;
    constexpr size_t kSize    = 16 * kPage;
    constexpr size_t kObj     = 64;
    constexpr int    kPerSlab = kPage / kObj;

    AlignedRegion region{kPage, kSize};
    PageAllocator pages{region.data(), kSize, kPage};

    SlabAllocator slab{kObj, kPage, kPage, 0.0f, 2};

    constexpr int         kRequest = kPerSlab * 2 + 3;
    std::vector<SlabSlot> outs(kRequest);
    int                   got = slab.allocate(outs.data(), kRequest, pages);
    REQUIRE(got == kRequest);

    std::vector<char*> addrs;
    for (int i = 0; i < got; ++i) {
        char* addr = slab.AddressOf(outs[i]);
        REQUIRE(addr != nullptr);
        REQUIRE(addr >= region.data());
        REQUIRE(addr + kObj <= (char*)region.data() + kSize);
        addrs.push_back(addr);
    }
    std::sort(addrs.begin(), addrs.end());
    REQUIRE(std::unique(addrs.begin(), addrs.end()) == addrs.end());

    slab.deallocate(outs.data(), got, pages);
}

TEST_CASE("SlabAllocator round-trip", "[memory][slab]")
{
    constexpr size_t kPage = 4096;
    constexpr size_t kSize = 16 * kPage;
    constexpr size_t kObj  = 64;

    AlignedRegion region{kPage, kSize};
    PageAllocator pages{region.data(), kSize, kPage};

    SlabAllocator slab{kObj, kPage, kPage, 0.0f, 2};

    for (int round = 0; round < 5; ++round) {
        constexpr int         kBatch = 200;
        std::vector<SlabSlot> a(kBatch);
        std::vector<SlabSlot> b(kBatch);
        REQUIRE(slab.allocate(a.data(), kBatch, pages) == kBatch);
        REQUIRE(slab.allocate(b.data(), kBatch, pages) == kBatch);

        // a and b live concurrently; their object addresses must not collide.
        std::vector<char*> all;
        for (auto& o : a) {
            all.push_back(slab.AddressOf(o));
        }
        for (auto& o : b) {
            all.push_back(slab.AddressOf(o));
        }
        std::sort(all.begin(), all.end());
        REQUIRE(std::unique(all.begin(), all.end()) == all.end());

        slab.deallocate(a.data(), kBatch, pages);
        slab.deallocate(b.data(), kBatch, pages);
    }
}

TEST_CASE("SlabAllocator stress (random)", "[memory][slab][stress]")
{
    constexpr size_t kPage = 4096;
    constexpr size_t kSize = 32 * kPage;
    constexpr size_t kObj  = 32;
    constexpr int    kOps  = 5000;

    AlignedRegion region{kPage, kSize};
    PageAllocator pages{region.data(), kSize, kPage};

    SlabAllocator slab{kObj, kPage, kPage, 0.0f, 2};

    std::mt19937 rng{0xc0ffee01U};

    std::vector<SlabSlot> live;

    for (int op = 0; op < kOps; ++op) {
        bool do_alloc = live.empty() ? true : (rng() & 1);
        if (do_alloc) {
            int                   batch = std::uniform_int_distribution<int>{1, 16}(rng);
            std::vector<SlabSlot> out(batch);
            int                   got = slab.allocate(out.data(), batch, pages);
            for (int i = 0; i < got; ++i) {
                char* addr = slab.AddressOf(out[i]);
                REQUIRE(addr >= region.data());
                REQUIRE(addr + kObj <= (char*)region.data() + kSize);
                live.push_back(out[i]);
            }
        }
        else {
            int                   batch = std::uniform_int_distribution<int>{1, std::min<int>(16, live.size())}(rng);
            std::vector<SlabSlot> to_free;
            for (int i = 0; i < batch; ++i) {
                size_t idx = std::uniform_int_distribution<size_t>{0, live.size() - 1}(rng);
                to_free.push_back(live[idx]);
                live[idx] = live.back();
                live.pop_back();
            }
            slab.deallocate(to_free.data(), to_free.size(), pages);
        }

        // Live objects are pairwise distinct.
        if ((op & 0xff) == 0 && !live.empty()) {
            std::vector<char*> snap;
            for (auto& o : live) {
                snap.push_back(slab.AddressOf(o));
            }
            std::sort(snap.begin(), snap.end());
            REQUIRE(std::unique(snap.begin(), snap.end()) == snap.end());
        }
    }

    if (!live.empty()) {
        slab.deallocate(live.data(), live.size(), pages);
    }
}

TEST_CASE("ObjectAllocator construct + Register", "[memory][object]")
{
    using core::Allocator;
    using core::Buffer;

    // The hard-coded kPageSize/kMinSlabSize inside ObjectAllocator::Impl is 32 MiB,
    // so the buffer must be large enough to hold at least a couple of slabs.
    constexpr size_t kBytes = 64UL << 20;  // 64 MiB

    Allocator alloc{kCPU};
    Buffer    buf{kBytes, data_type_v<int8_t>, alloc};

    ObjectAllocator obj{buf};

    int idx0 = obj.Register(/*size=*/24, /*alignment=*/8);
    int idx1 = obj.Register(/*size=*/96, /*alignment=*/16);
    REQUIRE(idx0 == 0);
    REQUIRE(idx1 == 1);
    REQUIRE(idx0 != idx1);
}

TEST_CASE("ObjectAllocator Allocate / Deallocate", "[memory][object]")
{
    using core::Allocator;
    using core::Buffer;

    constexpr size_t kBytes = 64UL << 20;
    // Use an object size that yields 512 objects per 32 MiB slab; tiny
    // sizes here would force Slab::initialize to push hundreds of
    // thousands of list nodes, dominating test runtime.
    constexpr size_t kSize  = 65536;  // 64 KiB
    constexpr size_t kAlign = 64;

    Allocator alloc{kCPU};
    Buffer    buf{kBytes, data_type_v<int8_t>, alloc};

    ObjectAllocator obj{buf};
    int             idx = obj.Register(kSize, kAlign);

    constexpr size_t            kBatch = 8;
    std::vector<object_alloc_t> outs(kBatch);
    size_t                      got = obj.Allocate(idx, outs.data(), kBatch);
    REQUIRE(got == kBatch);

    char* base = static_cast<char*>(buf.raw_data());
    for (size_t i = 0; i < got; ++i) {
        char* addr = outs[i]->base(0);
        REQUIRE(addr >= base);
        REQUIRE(addr + kSize <= base + kBytes);
        REQUIRE(reinterpret_cast<uintptr_t>(addr) % kAlign == 0);
    }

    // Single-part objects store part 0 inline (no heap vectors) and read it back
    // through base(0).
    REQUIRE(outs[0]->part_count() == 1);
    REQUIRE(outs[0]->base(0) == outs[0]->base0);
    REQUIRE(outs[0]->bases.empty());

    obj.Deallocate(idx, outs.data(), got);

    got = obj.Allocate(idx, outs.data(), kBatch);
    REQUIRE(got == kBatch);
    obj.Deallocate(idx, outs.data(), got);
}

TEST_CASE("ObjectAllocator multiple registrations", "[memory][object]")
{
    using core::Allocator;
    using core::Buffer;

    // Raw core::Buffer allocations are not guaranteed to be 32 MiB-aligned; one
    // extra page guarantees two usable ObjectAllocator pages after align_base padding.
    constexpr size_t kBytes = 3 * kObjectAllocatorPageBytes;
    constexpr size_t kSize0 = 65536;   // 64 KiB
    constexpr size_t kSize1 = 131072;  // 128 KiB

    Allocator alloc{kCPU};
    Buffer    buf{kBytes, data_type_v<int8_t>, alloc};

    ObjectAllocator obj{buf};

    int idx0 = obj.Register(kSize0, 64);
    int idx1 = obj.Register(kSize1, 128);

    constexpr size_t            kBatch = 8;
    std::vector<object_alloc_t> a(kBatch);
    std::vector<object_alloc_t> b(kBatch);
    REQUIRE(obj.Allocate(idx0, a.data(), kBatch) == kBatch);
    REQUIRE(obj.Allocate(idx1, b.data(), kBatch) == kBatch);

    for (auto& x : a) {
        for (auto& y : b) {
            REQUIRE_FALSE(ranges_overlap(x->base(0), kSize0, y->base(0), kSize1));
        }
    }

    obj.Deallocate(idx0, a.data(), kBatch);
    obj.Deallocate(idx1, b.data(), kBatch);
}

TEST_CASE("ScratchAllocator leaves source untouched", "[memory][object][scratch]")
{
    using core::Allocator;
    using core::Buffer;

    constexpr size_t kBytes = 64UL << 20;
    constexpr size_t kSize  = 65536;
    constexpr size_t kAlign = 64;

    Allocator alloc{kCPU};
    Buffer    buf{kBytes, data_type_v<int8_t>, alloc};

    ObjectAllocator obj{buf};
    const int       idx = obj.Register(kSize, kAlign);

    constexpr size_t            kBatch = 16;
    std::vector<object_alloc_t> live(kBatch);
    REQUIRE(obj.Allocate(idx, live.data(), kBatch) == kBatch);

    std::vector<char*>    addrs(kBatch);
    std::vector<uint64_t> keys(kBatch);
    for (size_t i = 0; i < kBatch; ++i) {
        addrs[i] = live[i]->base(0);
        keys[i]  = live[i]->key;
    }

    // A scratch probe shares the backing region (same addresses) but its
    // alloc/evict must not perturb the source's live allocations.
    {
        ScratchAllocator scratch{obj};
        scratch.Evict(idx, live[0].a);   // free a committed allocation in the copy
        REQUIRE(scratch.Allocate(idx));  // reuse freed capacity in the copy
    }

    for (size_t i = 0; i < kBatch; ++i) {
        REQUIRE(obj.IsValid(live[i], keys[i]));
        REQUIRE(live[i]->base(0) == addrs[i]);
    }
    obj.Deallocate(idx, live.data(), kBatch);
}

TEST_CASE("ObjectAllocator IsValid sentinel and lifecycle", "[memory][object][isvalid]")
{
    using core::Allocator;
    using core::Buffer;

    constexpr size_t kBytes = 64UL << 20;
    constexpr size_t kSize  = 65536;
    constexpr size_t kAlign = 64;

    Allocator alloc{kCPU};
    Buffer    buf{kBytes, data_type_v<int8_t>, alloc};

    ObjectAllocator obj{buf};
    int             idx = obj.Register(kSize, kAlign);

    // Null handle is never valid, for any snapshot key.
    object_alloc_t zero{};
    REQUIRE(obj.IsValid(zero, 0) == false);
    REQUIRE(obj.IsValid(zero, 12345) == false);

    // Live allocations: valid against the key snapshotted at alloc time.
    constexpr size_t            kBatch = 8;
    std::vector<object_alloc_t> live(kBatch);
    REQUIRE(obj.Allocate(idx, live.data(), kBatch) == kBatch);
    std::vector<uint64_t> keys(kBatch);
    for (size_t i = 0; i < kBatch; ++i) {
        keys[i] = live[i]->key;
        REQUIRE(obj.IsValid(live[i], keys[i]) == true);
    }

    // Deallocate: the key is zeroed, so every snapshot stops matching.
    obj.Deallocate(idx, live.data(), kBatch);
    for (size_t i = 0; i < kBatch; ++i) {
        REQUIRE(obj.IsValid(live[i], keys[i]) == false);
    }

    // Re-allocate: even if an Allocation* is recycled into live2, the monotonic
    // key advanced, so the OLD snapshot keys still fail (ABA-proof).
    std::vector<object_alloc_t> live2(kBatch);
    REQUIRE(obj.Allocate(idx, live2.data(), kBatch) == kBatch);
    for (size_t i = 0; i < kBatch; ++i) {
        REQUIRE(obj.IsValid(live2[i], live2[i]->key) == true);
        REQUIRE(obj.IsValid(live[i], keys[i]) == false);
    }

    obj.Deallocate(idx, live2.data(), kBatch);
}

TEST_CASE("ObjectAllocator IsValid across slab reclaim", "[memory][object][isvalid]")
{
    using core::Allocator;
    using core::Buffer;

    // 64 KiB objects yield 512 objects per 32 MiB slab; 768 fills two slabs.
    // kMaxEmptySlabs is zero, so freeing all allocations reclaims both slabs.
    constexpr size_t kBytes = 3 * kObjectAllocatorPageBytes;
    constexpr size_t kSize  = 65536;
    constexpr size_t kAlign = 64;
    constexpr size_t kAlloc = 768;

    Allocator alloc{kCPU};
    Buffer    buf{kBytes, data_type_v<int8_t>, alloc};

    ObjectAllocator obj{buf};
    int             idx = obj.Register(kSize, kAlign);

    std::vector<object_alloc_t> handles(kAlloc);
    REQUIRE(obj.Allocate(idx, handles.data(), kAlloc) == kAlloc);
    std::vector<uint64_t> keys(kAlloc);
    for (size_t i = 0; i < kAlloc; ++i) {
        keys[i] = handles[i]->key;
    }

    obj.Deallocate(idx, handles.data(), kAlloc);
    for (size_t i = 0; i < kAlloc; ++i) {
        REQUIRE(obj.IsValid(handles[i], keys[i]) == false);  // key zeroed on free
    }

    // Churn (slab reclaim/reuse underneath): fresh keys are monotonic, so the
    // stale snapshots never match again.
    std::vector<object_alloc_t> refill(kAlloc);
    REQUIRE(obj.Allocate(idx, refill.data(), kAlloc) == kAlloc);
    for (size_t i = 0; i < kAlloc; ++i) {
        REQUIRE(obj.IsValid(handles[i], keys[i]) == false);
        REQUIRE(obj.IsValid(refill[i], refill[i]->key) == true);
    }

    obj.Deallocate(idx, refill.data(), kAlloc);
}

TEST_CASE("object_alloc_t opacity", "[memory][object][opacity]")
{
    object_alloc_t a{};
    object_alloc_t b{};
    REQUIRE(a == b);
    REQUIRE_FALSE(a != b);
    REQUIRE_FALSE(a < b);

    object_alloc_t c = a;
    REQUIRE(c == a);

    // A non-null sentinel compares unequal to the null handle.
    object_alloc_t s{reinterpret_cast<const Allocation*>(0x1000)};
    REQUIRE(s != a);
    REQUIRE(((a < s) || (s < a)));

    std::vector<object_alloc_t> v{a, b, c, s};
    std::sort(v.begin(), v.end());

    static_assert(sizeof(object_alloc_t) == 8);
}

TEST_CASE("ObjectAllocator composite lifecycle", "[memory][object][composite]")
{
    using core::Allocator;
    using core::Buffer;

    constexpr size_t kBytes = 3 * kObjectAllocatorPageBytes;
    constexpr size_t kRec   = 65536;   // recurrent part size
    constexpr size_t kConv  = 131072;  // conv part size (distinct slab class)
    constexpr int    kN     = 4;       // recurrent parts

    Allocator alloc{kCPU};
    Buffer    buf{kBytes, data_type_v<int8_t>, alloc};

    ObjectAllocator obj{buf};
    // recurrent parts 1..N (count kN) + conv accumulation part 0 (count 1)
    const int idx = obj.Register({{kRec, 1, kN}, {kConv, 1, 1}});

    REQUIRE(obj.PartCount(idx) == kN + 1);
    for (int p = 0; p < kN; ++p) {
        REQUIRE(obj.PartBytes(idx, p) == kRec);
    }
    REQUIRE(obj.PartBytes(idx, kN) == kConv);

    object_alloc_t h{};
    REQUIRE(obj.Allocate(idx, &h, 1) == 1);
    const uint64_t h_key = h->key;
    REQUIRE(obj.IsValid(h, h_key) == true);

    REQUIRE(h->part_count() == kN + 1);
    REQUIRE(h->base(0) == h->bases[0]);  // composite uses the bases vector

    // All part bases distinct and inside the buffer.
    char*                    base = static_cast<char*>(buf.raw_data());
    std::vector<const char*> addrs;
    for (int p = 0; p < kN + 1; ++p) {
        REQUIRE(h->base(p) >= base);
        REQUIRE(h->base(p) + obj.PartBytes(idx, p) <= base + kBytes);
        addrs.push_back(h->base(p));
    }
    std::sort(addrs.begin(), addrs.end());
    REQUIRE(std::unique(addrs.begin(), addrs.end()) == addrs.end());

    obj.Deallocate(idx, &h, 1);
    REQUIRE(obj.IsValid(h, h_key) == false);

    // Re-allocate succeeds (pages reclaimed); the freed snapshot stays invalid.
    object_alloc_t h2{};
    REQUIRE(obj.Allocate(idx, &h2, 1) == 1);
    REQUIRE(obj.IsValid(h2, h2->key) == true);
    REQUIRE(obj.IsValid(h, h_key) == false);
    obj.Deallocate(idx, &h2, 1);
}

TEST_CASE("ObjectAllocator composite stale handle after recycle", "[memory][object][composite]")
{
    using core::Allocator;
    using core::Buffer;

    constexpr size_t kBytes = 3 * kObjectAllocatorPageBytes;
    Allocator        alloc{kCPU};
    Buffer           buf{kBytes, data_type_v<int8_t>, alloc};

    ObjectAllocator obj{buf};
    const int       idx = obj.Register({{65536, 1, 2}, {131072, 1, 1}});

    object_alloc_t a{};
    REQUIRE(obj.Allocate(idx, &a, 1) == 1);
    const uint64_t a_key = a->key;
    obj.Deallocate(idx, &a, 1);  // frees the allocation; its key is zeroed

    object_alloc_t b{};
    REQUIRE(obj.Allocate(idx, &b, 1) == 1);  // fresh, monotonic key

    // `a`'s snapshot key no longer matches (freed, or recycled into `b` with a
    // new key) -> stale/invalid.
    REQUIRE(obj.IsValid(a, a_key) == false);
    REQUIRE(obj.IsValid(b, b->key) == true);
    obj.Deallocate(idx, &b, 1);
}

TEST_CASE("ObjectAllocator composite atomic rollback on partial OOM", "[memory][object][composite]")
{
    using core::Buffer;
    using core::Device;

    // 128 MiB / 32 MiB pages = 4 single-page slots. A 32 MiB object => 1 object
    // per 32 MiB slab (one page each). The region base must be 32 MiB-aligned so
    // PageAllocator does not lose a page to align_base padding.
    constexpr size_t kPage  = kObjectAllocatorPageBytes;
    constexpr size_t kBytes = 4 * kPage;
    constexpr size_t kObj   = kPage;  // 32 MiB -> slab_size 32 MiB, 1 obj/slab

    AlignedRegion region{kPage, kBytes};
    Buffer        buf{region.data(), static_cast<ssize_t>(region.size()), data_type_v<int8_t>, Device{kCPU}};

    ObjectAllocator obj{buf};
    // member 0: 3 parts (3 pages, fits); member 1: 2 parts (needs 2 pages, only 1
    // left) -> partial OOM on member 1 -> roll back all 3 of member 0.
    const int idx = obj.Register({{kObj, 1, 3}, {kObj, 1, 2}});

    object_alloc_t h{};
    REQUIRE(obj.Allocate(idx, &h, 1) == 0);  // not placed
    REQUIRE(h.a == nullptr);                 // batch leaves the slot null on OOM
    REQUIRE(obj.IsValid(h, 0) == false);

    // No leak: a simple 32 MiB object (same aligned size -> shared slab class)
    // can fully allocate all 4 pages.
    const int                   sidx = obj.Register(kObj, 1);
    std::vector<object_alloc_t> live(4);
    REQUIRE(obj.Allocate(sidx, live.data(), 4) == 4);
    obj.Deallocate(sidx, live.data(), 4);
}

TEST_CASE("ObjectAllocator size dedup", "[memory][object][dedup]")
{
    using core::Allocator;
    using core::Buffer;

    constexpr size_t kBytes = 64UL << 20;
    Allocator        alloc{kCPU};
    Buffer           buf{kBytes, data_type_v<int8_t>, alloc};

    ObjectAllocator obj{buf};

    // Equal aligned size (65536) -> SAME object id.
    const int a = obj.Register(65500, 64);  // aligns up to 65536
    const int b = obj.Register(65536, 64);
    REQUIRE(a == b);

    // Different aligned size -> different id.
    const int c = obj.Register(131072, 64);
    REQUIRE(c != a);

    // Composite single entry count 1 collapses to the simple deduped id.
    const int d = obj.Register({{65536, 64, 1}});
    REQUIRE(d == a);
}

TEST_CASE("ScratchAllocator composite source independence", "[memory][object][composite][scratch]")
{
    using core::Allocator;
    using core::Buffer;

    constexpr size_t kBytes = 3 * kObjectAllocatorPageBytes;
    Allocator        alloc{kCPU};
    Buffer           buf{kBytes, data_type_v<int8_t>, alloc};

    ObjectAllocator obj{buf};
    const int       idx = obj.Register({{65536, 1, 3}, {131072, 1, 1}});

    object_alloc_t a{};
    REQUIRE(obj.Allocate(idx, &a, 1) == 1);
    const uint64_t a_key   = a->key;
    char*          a_part0 = a->base(0);

    {
        ScratchAllocator scratch{obj};
        scratch.Evict(idx, a.a);
        // After freeing `a`'s slots in the copy, the copy can place another.
        REQUIRE(scratch.Allocate(idx));
    }

    REQUIRE(obj.IsValid(a, a_key));
    REQUIRE(a->base(0) == a_part0);
    obj.Deallocate(idx, &a, 1);
}

TEST_CASE("ObjectAllocator resolve gives part bases", "[memory][object][resolve]")
{
    using core::Allocator;
    using core::Buffer;

    constexpr size_t kBytes = 3 * kObjectAllocatorPageBytes;
    Allocator        alloc{kCPU};
    Buffer           buf{kBytes, data_type_v<int8_t>, alloc};

    ObjectAllocator obj{buf};
    const int       idx = obj.Register({{65536, 1, 2}, {131072, 1, 1}});

    object_alloc_t h{};
    REQUIRE(obj.Allocate(idx, &h, 1) == 1);

    const Allocation* a = h.a;  // the handle IS the resolved Allocation pointer
    REQUIRE(a != nullptr);
    REQUIRE(h->part_count() == obj.PartCount(idx));
    REQUIRE(static_cast<int>(a->bases.size()) == obj.PartCount(idx));  // composite: all parts in bases
    for (int p = 0; p < obj.PartCount(idx); ++p) {
        REQUIRE(a->bases[p] == h->base(p));
    }
    obj.Deallocate(idx, &h, 1);
}

TEST_CASE("ScratchAllocator reports OOM at capacity", "[memory][object][scratch]")
{
    using core::Buffer;
    using core::Device;

    // 128 MiB / 32 MiB pages = 4 single-page slots. A 32 MiB object => slab_size
    // 32 MiB, 1 obj/slab, so the region holds exactly 4 such objects.
    constexpr size_t kPage  = kObjectAllocatorPageBytes;
    constexpr size_t kBytes = 4 * kPage;
    constexpr size_t kObj   = kPage;

    AlignedRegion region{kPage, kBytes};
    Buffer        buf{region.data(), static_cast<ssize_t>(region.size()), data_type_v<int8_t>, Device{kCPU}};

    ObjectAllocator obj{buf};
    const int       idx = obj.Register(kObj, 1);

    // One live allocation in the source; snapshot its resolved address.
    object_alloc_t live{};
    REQUIRE(obj.Allocate(idx, &live, 1) == 1);
    const uint64_t live_key  = live->key;
    char*          live_addr = live->base(0);

    // A scratch probe inherits the source's committed capacity (1 of 4 slots
    // used). It can place the remaining 3, then the 4th must report OOM.
    {
        ScratchAllocator scratch{obj};
        REQUIRE(scratch.Allocate(idx));        // slot 2
        REQUIRE(scratch.Allocate(idx));        // slot 3
        REQUIRE(scratch.Allocate(idx));        // slot 4 -> capacity now full
        REQUIRE_FALSE(scratch.Allocate(idx));  // capacity exceeded -> false
    }

    // Source untouched: its live handle still resolves to the same address.
    REQUIRE(obj.IsValid(live, live_key));
    REQUIRE(live->base(0) == live_addr);
    obj.Deallocate(idx, &live, 1);
}

TEST_CASE("ObjectAllocator Allocation pointer is stable across churn", "[memory][object][resolve][stability]")
{
    using core::Allocator;
    using core::Buffer;

    constexpr size_t kBytes = 64UL << 20;
    constexpr size_t kSize  = 65536;
    constexpr size_t kAlign = 64;

    Allocator alloc{kCPU};
    Buffer    buf{kBytes, data_type_v<int8_t>, alloc};

    ObjectAllocator obj{buf};
    const int       idx = obj.Register(kSize, kAlign);

    // Pin one allocation and snapshot its durable entry (single-part: base0).
    object_alloc_t h0{};
    REQUIRE(obj.Allocate(idx, &h0, 1) == 1);
    const Allocation* const a0    = h0.a;
    char* const             base0 = h0->base(0);
    REQUIRE(h0->part_count() == 1);
    REQUIRE(h0->bases.empty());  // inline storage, no heap parts

    // Churn the table around h0: allocate a batch, free half, allocate another
    // batch -- all WITHOUT touching h0.
    constexpr size_t            kBatch = 32;
    std::vector<object_alloc_t> first(kBatch);
    REQUIRE(obj.Allocate(idx, first.data(), kBatch) == kBatch);
    obj.Deallocate(idx, first.data(), kBatch / 2);  // free the first half
    std::vector<object_alloc_t> second(kBatch);
    REQUIRE(obj.Allocate(idx, second.data(), kBatch) == kBatch);

    // h0's durable Allocation* and its resolved base are unchanged by the churn:
    // this is the invariant CacheBlock relies on when it caches the handle.
    REQUIRE(h0.a == a0);
    REQUIRE(h0->base(0) == base0);

    obj.Deallocate(idx, &h0, 1);
    obj.Deallocate(idx, first.data() + kBatch / 2, kBatch / 2);  // free the still-live half
    obj.Deallocate(idx, second.data(), kBatch);
}

TEST_CASE("ObjectAllocator simple/composite share one slab class", "[memory][object][dedup][interchange]")
{
    using core::Buffer;
    using core::Device;

    // Buffer == exactly ONE ObjectAllocator page (MemoryState::kPageSize, 32 MiB).
    // 32 MiB-aligned base so PageAllocator does not lose the only page to padding.
    // With a single page there is no second page to reclaim, so a partial donor
    // slab cannot help a *separate* slab class -- sharing is the only way through.
    constexpr size_t kPageBytes = 32UL << 20;  // == MemoryState::kPageSize
    constexpr size_t kObj       = 65536;       // 64 KiB -> slab holds many slots

    AlignedRegion region{kPageBytes, kPageBytes};
    Buffer        buf{region.data(), static_cast<ssize_t>(region.size()), data_type_v<int8_t>, Device{kCPU}};

    ObjectAllocator obj{buf};

    // Simple object of size kObj, and a composite whose single member is two
    // parts of the SAME aligned size kObj. count=2 (!=1) keeps it a real
    // composite (does not collapse to the simple id).
    const int sidx = obj.Register(kObj, 1);
    const int cidx = obj.Register({{kObj, 1, 2}});
    REQUIRE(sidx != cidx);              // distinct objects ...
    REQUIRE(obj.PartCount(cidx) == 2);  // ... but the composite has 2 parts

    // One simple alloc creates the single slab on the only page and leaves it
    // PARTIAL (one slot used, the rest free, zero free pages remain).
    object_alloc_t s0{};
    REQUIRE(obj.Allocate(sidx, &s0, 1) == 1);

    // The composite's two parts must come from that same partial slab. If the
    // composite had its own slab class it would need a fresh page (none left)
    // and fail. Success here == simple and composite share one slab class.
    object_alloc_t comp{};
    REQUIRE(obj.Allocate(cidx, &comp, 1) == 1);
    REQUIRE(comp.a != nullptr);
    REQUIRE(comp->part_count() == 2);

    obj.Deallocate(cidx, &comp, 1);
    obj.Deallocate(sidx, &s0, 1);
}
